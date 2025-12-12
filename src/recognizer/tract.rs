use std::{path::PathBuf, thread};

use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender};
use tract_onnx::prelude::*;

use super::{
    HandposeEngine,
    common::{self, HandposeOutput},
    run_worker_loop,
};
use crate::{
    model_download::ensure_model_available,
    types::{Frame, GestureResult},
};

pub fn start_worker(
    model_path: PathBuf,
    frame_rx: Receiver<Frame>,
    result_tx: Sender<GestureResult>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        if let Err(err) = ensure_model_available(&model_path) {
            log::error!(
                "failed to prepare handpose model at {}: {err:?}",
                model_path.display()
            );
            return;
        }

        let engine = match TractEngine::new(&model_path) {
            Ok(engine) => {
                log::info!(
                    "handpose tract backend ready ({} nodes) using {}",
                    engine.node_count,
                    model_path.display()
                );
                engine
            }
            Err(err) => {
                log::error!("failed to load handpose model: {err:?}");
                return;
            }
        };

        run_worker_loop(engine, frame_rx, result_tx);
    })
}

struct TractEngine {
    model: TypedRunnableModel<TypedModel>,
    pub node_count: usize,
}

impl TractEngine {
    fn new(model_path: &PathBuf) -> Result<Self> {
        let mut model = tract_onnx::onnx().model_for_path(model_path)?;
        model.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![
                    1.to_dim(),
                    (common::INPUT_SIZE as usize).to_dim(),
                    (common::INPUT_SIZE as usize).to_dim(),
                    3.to_dim()
                ],
            ),
        )?;

        let node_count = model.nodes().len();
        let model = model.into_optimized()?.into_runnable()?;

        Ok(Self { model, node_count })
    }
}

impl HandposeEngine for TractEngine {
    fn infer(&mut self, frame: &Frame) -> Result<HandposeOutput> {
        let (input, letterbox) = common::prepare_frame(frame)?;
        let outputs = self
            .model
            .run(tvec![input.into_tensor().into()])
            .context("failed to run handpose model")?;

        let coords = outputs
            .get(0)
            .ok_or_else(|| anyhow!("model returned no landmarks output"))?
            .to_array_view::<f32>()?;
        let flattened: Vec<f32> = coords.iter().copied().collect();
        let landmarks = common::decode_landmarks(&flattened)?;

        let confidence = outputs
            .get(1)
            .and_then(|t| t.to_array_view::<f32>().ok())
            .and_then(|v| v.iter().next().copied())
            .unwrap_or(0.0);
        let handedness = outputs
            .get(2)
            .and_then(|t| t.to_array_view::<f32>().ok())
            .and_then(|v| v.iter().next().copied())
            .unwrap_or(0.0);

        let projected = common::project_landmarks(&landmarks, &letterbox);

        Ok(HandposeOutput {
            raw_landmarks: landmarks,
            projected_landmarks: projected,
            confidence: confidence.clamp(0.0, 1.0),
            handedness,
        })
    }
}
