#[path = "../src/model_download.rs"]
mod model_download;

use model_download::{default_model_path, ensure_model_available};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use image::{RgbaImage, imageops::FilterType};
use tract_onnx::prelude::*;

const INPUT_SIZE: u32 = 224;

fn main() -> Result<()> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let input_image = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("demo/image.png"));
    let model_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(default_model_path);
    let duration_secs = args.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(1);

    let (input_tensor, _) = prepare_tensor(&input_image).context("failed to read input image")?;
    let input_value: TValue = input_tensor.into();
    ensure_model_available(&model_path)?;
    let model = load_model(&model_path)?;

    println!(
        "Benchmarking model {} on {} for {}s",
        model_path.display(),
        input_image.display(),
        duration_secs
    );

    // Warm-up once to trigger any lazy initialisation.
    let warmup = model.run(tvec![input_value.clone()])?;
    let warmup_conf = extract_confidence(&warmup);
    println!("Warm-up done (conf {:.3})", warmup_conf);

    let duration = Duration::from_secs(duration_secs.max(1));
    let start = Instant::now();
    let mut iterations: u64 = 0;
    let mut last_conf = warmup_conf;
    while start.elapsed() < duration {
        let outputs = model.run(tvec![input_value.clone()])?;
        last_conf = extract_confidence(&outputs);
        iterations += 1;
    }
    let elapsed = start.elapsed();
    let fps = iterations as f64 / elapsed.as_secs_f64();

    println!(
        "Ran {} inferences in {:.3}s -> {:.1} fps (last conf {:.3})",
        iterations,
        elapsed.as_secs_f64(),
        fps,
        last_conf
    );

    Ok(())
}

fn load_model(model_path: &PathBuf) -> TractResult<TypedRunnableModel<TypedModel>> {
    let mut model = tract_onnx::onnx().model_for_path(model_path)?;
    model.set_input_fact(
        0,
        InferenceFact::dt_shape(
            f32::datum_type(),
            tvec![
                1.to_dim(),
                (INPUT_SIZE as usize).to_dim(),
                (INPUT_SIZE as usize).to_dim(),
                3.to_dim()
            ],
        ),
    )?;

    model.into_optimized()?.into_runnable()
}

fn prepare_tensor(path: &PathBuf) -> Result<(Tensor, (u32, u32))> {
    let image = image::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .to_rgba8();
    let (orig_w, orig_h) = image.dimensions();

    let scale = INPUT_SIZE as f32 / (orig_w.max(orig_h) as f32);
    let new_w = (orig_w as f32 * scale).round().max(1.0) as u32;
    let new_h = (orig_h as f32 * scale).round().max(1.0) as u32;
    let resized = image::imageops::resize(&image, new_w, new_h, FilterType::CatmullRom);

    let pad_x = ((INPUT_SIZE as i64 - new_w as i64) / 2).max(0) as u32;
    let pad_y = ((INPUT_SIZE as i64 - new_h as i64) / 2).max(0) as u32;
    let mut letterboxed =
        RgbaImage::from_pixel(INPUT_SIZE, INPUT_SIZE, image::Rgba([0, 0, 0, 255]));
    for y in 0..new_h {
        for x in 0..new_w {
            let px = *resized.get_pixel(x, y);
            letterboxed.put_pixel(x + pad_x, y + pad_y, px);
        }
    }

    let mut input =
        tract_ndarray::Array4::<f32>::zeros((1, INPUT_SIZE as usize, INPUT_SIZE as usize, 3));
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let pixel = letterboxed.get_pixel(x, y).0;
            input[[0, y as usize, x as usize, 0]] = pixel[0] as f32 / 255.0;
            input[[0, y as usize, x as usize, 1]] = pixel[1] as f32 / 255.0;
            input[[0, y as usize, x as usize, 2]] = pixel[2] as f32 / 255.0;
        }
    }

    Ok((input.into_tensor(), (orig_w, orig_h)))
}

fn extract_confidence(outputs: &[TValue]) -> f32 {
    outputs
        .get(1)
        .and_then(|t| t.to_array_view::<f32>().ok())
        .and_then(|v| v.iter().next().copied())
        .unwrap_or(0.0)
}
