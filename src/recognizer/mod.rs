#[cfg(all(feature = "backend-tract", feature = "backend-ort"))]
compile_error!("Enable exactly one backend feature: choose backend-tract or backend-ort.");

#[cfg(not(any(feature = "backend-tract", feature = "backend-ort")))]
compile_error!("Enable at least one backend feature: backend-tract or backend-ort.");

mod common;
#[cfg(feature = "backend-ort")]
mod ort;
#[cfg(feature = "backend-tract")]
mod tract;

use std::{path::PathBuf, thread};

use crossbeam_channel::{Receiver, Sender};

use crate::{
    gesture::GestureClassifier,
    model_download::default_model_path,
    types::{Frame, GestureResult},
};

use self::common::HandposeOutput;

pub(crate) trait HandposeEngine: Send + 'static {
    fn infer(&mut self, frame: &Frame) -> anyhow::Result<HandposeOutput>;
}

fn run_worker_loop<E: HandposeEngine>(
    mut engine: E,
    frame_rx: Receiver<Frame>,
    result_tx: Sender<GestureResult>,
) {
    let mut classifier = GestureClassifier::new();

    while let Some(frame) = recv_latest_frame(&frame_rx) {
        match engine.infer(&frame) {
            Ok(output) => {
                let gesture = build_gesture_result(output, &frame, &mut classifier);
                let _ = result_tx.try_send(gesture);
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
pub enum RecognizerBackend {
    #[cfg(feature = "backend-tract")]
    Tract { model_path: PathBuf },
    #[cfg(feature = "backend-ort")]
    Ort { model_path: PathBuf },
}

impl RecognizerBackend {
    pub fn model_path(&self) -> PathBuf {
        match self {
            #[cfg(feature = "backend-tract")]
            RecognizerBackend::Tract { model_path } => model_path.clone(),
            #[cfg(feature = "backend-ort")]
            RecognizerBackend::Ort { model_path } => model_path.clone(),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            #[cfg(feature = "backend-tract")]
            RecognizerBackend::Tract { .. } => "tract",
            #[cfg(feature = "backend-ort")]
            RecognizerBackend::Ort { .. } => "ort",
        }
    }
}

#[cfg(feature = "backend-tract")]
impl Default for RecognizerBackend {
    fn default() -> Self {
        RecognizerBackend::Tract {
            model_path: default_model_path(),
        }
    }
}

#[cfg(all(not(feature = "backend-tract"), feature = "backend-ort"))]
impl Default for RecognizerBackend {
    fn default() -> Self {
        RecognizerBackend::Ort {
            model_path: default_model_path(),
        }
    }
}

pub fn start_recognizer(
    backend: RecognizerBackend,
    frame_rx: Receiver<Frame>,
    result_tx: Sender<GestureResult>,
) -> thread::JoinHandle<()> {
    log::info!("starting handpose backend: {}", backend.label());

    match backend {
        #[cfg(feature = "backend-tract")]
        RecognizerBackend::Tract { model_path } => {
            tract::start_worker(model_path, frame_rx, result_tx)
        }
        #[cfg(feature = "backend-ort")]
        RecognizerBackend::Ort { model_path } => ort::start_worker(model_path, frame_rx, result_tx),
    }
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
    }
}
