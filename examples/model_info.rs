#[allow(dead_code)]
#[path = "../src/model_download.rs"]
mod model_download;

use anyhow::Result;
use model_download::{
    default_handpose_estimator_model_path, default_palm_detector_model_path,
    ensure_handpose_estimator_model_ready, ensure_palm_detector_model_ready,
};
use std::path::PathBuf;

use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::ValueType,
};

fn main() -> Result<()> {
    env_logger::init();

    let handpose_estimator_model = default_handpose_estimator_model_path();

    println!("Loading model: {}", handpose_estimator_model.display());
    ensure_handpose_estimator_model_ready(&handpose_estimator_model, |_evt| {})?;
    print_model_info(&handpose_estimator_model)?;

    let palm_detector_model = default_palm_detector_model_path();
    println!(
        "Loading model: {}",
        default_palm_detector_model_path().display()
    );
    ensure_palm_detector_model_ready(&palm_detector_model, |_evt| {})?;
    print_model_info(&palm_detector_model)?;

    Ok(())
}

fn print_model_info(model_path: &PathBuf) -> Result<()> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(2)?
        .commit_from_file(model_path)?;

    println!("Inputs:");
    for (idx, input) in session.inputs.iter().enumerate() {
        println!(
            "  {}: name=\"{}\" type={:?}",
            idx, input.name, input.input_type
        );
        if let ValueType::Tensor { shape, .. } = &input.input_type {
            println!("     shape={:?}", shape);
        }
    }

    println!("Outputs:");
    for (idx, output) in session.outputs.iter().enumerate() {
        println!(
            "  {}: name=\"{}\" type={:?}",
            idx, output.name, output.output_type
        );
        if let ValueType::Tensor { shape, .. } = &output.output_type {
            println!("     shape={:?}", shape);
        }
    }

    Ok(())
}
