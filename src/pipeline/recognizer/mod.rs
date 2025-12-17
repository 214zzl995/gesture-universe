mod common;
mod ort;
pub(crate) mod palm;

use std::{path::PathBuf, thread};

use crossbeam_channel::{Receiver, Sender};

use crate::{
    gesture::GestureClassifier,
    model_download::{default_handpose_estimator_model_path, default_palm_detector_model_path},
    types::{Frame, GestureResult, RecognizedFrame},
};

use self::common::HandposeOutput;

pub(crate) trait HandposeEngine: Send + 'static {
    fn infer(&mut self, frame: &Frame) -> anyhow::Result<HandposeOutput>;
}

fn run_worker_loop<E: HandposeEngine>(
    mut engine: E,
    frame_rx: Receiver<Frame>,
    result_tx: Sender<RecognizedFrame>,
) {
    let mut classifier = GestureClassifier::new();

    while let Some(frame) = recv_latest_frame(&frame_rx) {
        match engine.infer(&frame) {
            Ok(output) => {
                let gesture = build_gesture_result(output, &frame, &mut classifier);
                let recognized = RecognizedFrame {
                    frame,
                    result: gesture,
                };
                let _ = result_tx.try_send(recognized);
            }
            Err(err) => {
                log::warn!("handpose inference failed: {err:?}");
            }
        }
    }
}

fn recv_latest_frame(frame_rx: &Receiver<Frame>) -> Option<Frame> {
    let mut frame = frame_rx.recv().ok()?;
    while let Ok(newer) = frame_rx.try_recv() {
        frame = newer;
    }
    Some(frame)
}

#[derive(Clone, Debug)]
pub struct RecognizerBackend {
    handpose_estimator_model_path: PathBuf,
    palm_detector_model_path: PathBuf,
}

impl RecognizerBackend {
    pub fn handpose_estimator_model_path(&self) -> PathBuf {
        self.handpose_estimator_model_path.clone()
    }

    pub fn palm_detector_model_path(&self) -> PathBuf {
        self.palm_detector_model_path.clone()
    }

    pub fn backend_label(&self) -> &'static str {
        "ort"
    }
}

impl Default for RecognizerBackend {
    fn default() -> Self {
        RecognizerBackend {
            handpose_estimator_model_path: default_handpose_estimator_model_path(),
            palm_detector_model_path: default_palm_detector_model_path(),
        }
    }
}

pub fn start_recognizer(
    backend: RecognizerBackend,
    frame_rx: Receiver<Frame>,
    result_tx: Sender<RecognizedFrame>,
) -> thread::JoinHandle<()> {
    log::info!("starting handpose backend: {}", backend.backend_label());

    ort::start_worker(backend, frame_rx, result_tx)
}

pub(crate) fn build_gesture_result(
    output: HandposeOutput,
    frame: &Frame,
    classifier: &mut GestureClassifier,
) -> GestureResult {
    let has_detection = output.confidence >= 0.2;
    let detail = if has_detection {
        classifier.classify(
            &output.raw_landmarks,
            &output.projected_landmarks,
            output.confidence,
            output.handedness,
            frame.timestamp,
        )
    } else {
        None
    };

    let label = detail
        .as_ref()
        .map(|d| format!("{}{}", d.primary.emoji(), d.primary.display_name()))
        .unwrap_or_else(|| {
            if has_detection {
                "检测到手".to_string()
            } else {
                "未检测到手".to_string()
            }
        });

    GestureResult {
        label,
        confidence: output.confidence,
        timestamp: frame.timestamp,
        landmarks: if has_detection {
            Some(output.projected_landmarks)
        } else {
            None
        },
        detail,
        palm_regions: output.palm_regions,
    }
}
