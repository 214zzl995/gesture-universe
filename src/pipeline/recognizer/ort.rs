use std::{
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use super::{
    HandposeEngine, RecognizerBackend,
    common::{self, HandposeOutput},
    palm::{PalmDetector, PalmDetectorConfig, crop_from_palm, pick_primary_region},
    run_worker_loop,
};
use crate::{
    model_download::{ensure_handpose_estimator_model_ready, ensure_palm_detector_model_ready},
    types::{Frame, RecognizedFrame},
};

pub fn start_worker(
    backend: RecognizerBackend,
    frame_rx: Receiver<Frame>,
    result_tx: Sender<RecognizedFrame>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let handpose_estimator_model_path = backend.handpose_estimator_model_path();
        let palm_detector_model_path = backend.palm_detector_model_path();

        if let Err(err) =
            ensure_handpose_estimator_model_ready(&handpose_estimator_model_path, |_evt| {})
        {
            log::error!(
                "failed to prepare handpose model at {}: {err:?}",
                handpose_estimator_model_path.display()
            );
            return;
        }

        if let Err(err) = ensure_palm_detector_model_ready(&palm_detector_model_path, |_evt| {}) {
            log::error!(
                "failed to prepare palm detector model at {}: {err:?}",
                palm_detector_model_path.display()
            );
            return;
        }

        let engine = match OrtEngine::new(&handpose_estimator_model_path, &palm_detector_model_path)
        {
            Ok(engine) => {
                log::info!(
                    "handpose ORT backend ready using {} and palm detector {}",
                    handpose_estimator_model_path.display(),
                    palm_detector_model_path.display()
                );
                engine
            }
            Err(err) => {
                log::error!("failed to load ORT handpose model: {err:?}");
                return;
            }
        };

        run_worker_loop(engine, frame_rx, result_tx);
    })
}

struct OrtEngine {
    handpose: Session,
    palm_detector: PalmDetector,
    tracker: HandTracker,
}

impl OrtEngine {
    fn new(model_path: &PathBuf, palm_detector_model_path: &PathBuf) -> Result<Self> {
        let handpose = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .commit_from_file(model_path)
            .with_context(|| format!("failed to load ORT session from {}", model_path.display()))?;

        let palm_detector =
            PalmDetector::new(palm_detector_model_path, PalmDetectorConfig::default())?;

        Ok(Self {
            handpose,
            palm_detector,
            tracker: HandTracker::new(),
        })
    }
}

impl HandposeEngine for OrtEngine {
    fn infer(&mut self, frame: &Frame) -> Result<HandposeOutput> {
        let now = frame.timestamp;
        let palm_regions = self.palm_detector.detect(frame).unwrap_or_else(|err| {
            log::warn!("palm detection failed: {err:?}");
            Vec::new()
        });

        let mut used_tracking_fallback = false;
        let (center, side, angle, prior_score) = if let Some(selected) =
            pick_primary_region(&palm_regions).or_else(|| palm_regions.get(0))
        {
            let (center, side, angle) = crop_from_palm(selected);
            (center, side, angle, selected.score)
        } else if let Some((tracked, score)) = self.tracker.estimate_roi(now) {
            used_tracking_fallback = true;
            (tracked.0, tracked.1, tracked.2, score)
        } else {
            return Ok(HandposeOutput {
                raw_landmarks: Vec::new(),
                projected_landmarks: Vec::new(),
                confidence: 0.0,
                handedness: 0.0,
                palm_regions,
            });
        };

        let (input, transform) =
            common::prepare_rotated_crop(frame, center, side, angle, common::INPUT_SIZE)?;
        let tensor = Tensor::from_array(input)?;
        let outputs = self
            .handpose
            .run(ort::inputs![tensor])
            .context("failed to run ORT session")?;

        if outputs.len() < 1 {
            return Err(anyhow!("model returned no outputs"));
        }

        let coords = outputs[0].try_extract_array::<f32>()?;
        let flattened: Vec<f32> = coords.iter().copied().collect();
        let landmarks = common::decode_landmarks(&flattened)?;

        let confidence = if outputs.len() > 1 {
            outputs[1]
                .try_extract_array::<f32>()
                .ok()
                .and_then(|arr| arr.iter().next().copied())
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let handedness = if outputs.len() > 2 {
            outputs[2]
                .try_extract_array::<f32>()
                .ok()
                .and_then(|arr| arr.iter().next().copied())
                .unwrap_or(0.0)
        } else {
            0.0
        };

        let projected = common::project_landmarks_with_transform(&landmarks, &transform);
        let mut confidence = (confidence * prior_score).clamp(0.0, 1.0);
        if used_tracking_fallback {
            confidence *= 0.9;
        }

        if !landmarks.is_empty() {
            self.tracker.update(&transform, &projected, confidence, now);
        }

        Ok(HandposeOutput {
            raw_landmarks: landmarks,
            projected_landmarks: projected,
            confidence,
            handedness,
            palm_regions,
        })
    }
}

// Keep a short-lived track so the hand does not disappear immediately when palm
// detection drops (e.g. back-of-hand rotations).
const TRACK_MAX_AGE: Duration = Duration::from_millis(450);
const TRACK_MIN_CONF: f32 = 0.15;

struct TrackedHand {
    transform: common::CropTransform,
    projected: Vec<(f32, f32)>,
    confidence: f32,
    last_seen: Instant,
}

impl TrackedHand {
    fn is_stale(&self, now: Instant) -> bool {
        now.duration_since(self.last_seen) > TRACK_MAX_AGE || self.confidence < TRACK_MIN_CONF
    }

    fn estimate_roi(&self) -> Option<((f32, f32), f32, f32)> {
        if self.projected.len() < 3 {
            return None;
        }

        let (min_x, max_x, min_y, max_y) = self
            .projected
            .iter()
            .fold((f32::MAX, f32::MIN, f32::MAX, f32::MIN), |acc, (x, y)| {
                (acc.0.min(*x), acc.1.max(*x), acc.2.min(*y), acc.3.max(*y))
            });

        if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
            return None;
        }

        let span = (max_x - min_x).max(max_y - min_y).max(1.0);
        let expanded = span * 1.8;
        let side = expanded
            .max(self.transform.side * 0.7)
            .min(self.transform.side * 2.5)
            .max(80.0);

        let center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
        let angle =
            estimate_orientation_from_landmarks(&self.projected).unwrap_or(self.transform.angle);

        Some((center, side, angle))
    }
}

struct HandTracker {
    last: Option<TrackedHand>,
}

impl HandTracker {
    fn new() -> Self {
        Self { last: None }
    }

    fn update(
        &mut self,
        transform: &common::CropTransform,
        projected: &[(f32, f32)],
        confidence: f32,
        now: Instant,
    ) {
        if projected.is_empty() {
            self.last = None;
            return;
        }

        self.last = Some(TrackedHand {
            transform: transform.clone(),
            projected: projected.to_vec(),
            confidence,
            last_seen: now,
        });
    }

    fn estimate_roi(&self, now: Instant) -> Option<(((f32, f32), f32, f32), f32)> {
        let tracked = self.last.as_ref()?;
        if tracked.is_stale(now) {
            return None;
        }
        tracked.estimate_roi().map(|roi| (roi, tracked.confidence))
    }
}

fn estimate_orientation_from_landmarks(points: &[(f32, f32)]) -> Option<f32> {
    use std::f32::consts::PI;

    if points.len() <= 17 {
        return None;
    }

    let wrist = points[0];
    let index = points[5];
    let pinky = points[17];
    let axis_x = ((index.0 + pinky.0) * 0.5) - wrist.0;
    let axis_y = ((index.1 + pinky.1) * 0.5) - wrist.1;

    if axis_x.abs() < f32::EPSILON && axis_y.abs() < f32::EPSILON {
        return None;
    }

    let radians = PI / 2.0 - (-(axis_y)).atan2(axis_x);
    let two_pi = 2.0 * PI;
    Some(radians - two_pi * ((radians + PI) / two_pi).floor())
}
