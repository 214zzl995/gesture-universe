#[cfg(all(feature = "backend-tract", feature = "backend-ort"))]
compile_error!("Enable exactly one backend feature: backend-tract or backend-ort.");
#[cfg(not(any(feature = "backend-tract", feature = "backend-ort")))]
compile_error!("Enable at least one backend feature: backend-tract or backend-ort.");

#[path = "../src/model_download.rs"]
mod model_download;

use anyhow::Result;
use model_download::{default_model_path, ensure_model_available};
use std::path::PathBuf;

#[cfg(feature = "backend-ort")]
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::ValueType,
};
#[cfg(feature = "backend-tract")]
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    env_logger::init();

    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_model_path);

    println!("Loading model: {}", model_path.display());
    ensure_model_available(&model_path)?;
    print_model_info(&model_path)?;

    Ok(())
}

#[cfg(feature = "backend-tract")]
fn print_model_info(model_path: &PathBuf) -> Result<()> {
    let mut model = tract_onnx::onnx().model_for_path(model_path)?;

    // The ONNX graph leaves some dims symbolic; seed the expected input shape so we
    // can infer output shapes.
    model.set_input_fact(
        0,
        InferenceFact::dt_shape(
            f32::datum_type(),
            tvec![1.to_dim(), 224.to_dim(), 224.to_dim(), 3.to_dim()],
        ),
    )?;

    let model = model.into_optimized()?;

    println!("Nodes: {}", model.nodes().len());
    println!("Inputs:");
    for (idx, outlet) in model.input_outlets()?.iter().enumerate() {
        let fact = model.outlet_fact(*outlet)?;
        println!(
            "  {}: name=\"{}\" type={:?} shape={:?}",
            idx,
            model.node(outlet.node).name,
            fact.datum_type,
            fact.shape
        );
    }

    println!("Outputs:");
    for (idx, outlet) in model.output_outlets()?.iter().enumerate() {
        let fact = model.outlet_fact(*outlet)?;
        println!(
            "  {}: name=\"{}\" type={:?} shape={:?}",
            idx,
            model.node(outlet.node).name,
            fact.datum_type,
            fact.shape
        );
    }

    Ok(())
}

#[cfg(feature = "backend-ort")]
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
