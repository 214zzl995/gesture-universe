#[path = "../src/model_download.rs"]
mod model_download;

use model_download::{default_model_path, ensure_model_available};
use std::path::PathBuf;

use anyhow::Result;
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    env_logger::init();

    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_model_path);

    println!("Loading model: {}", model_path.display());
    ensure_model_available(&model_path)?;
    let mut model = tract_onnx::onnx().model_for_path(&model_path)?;

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
