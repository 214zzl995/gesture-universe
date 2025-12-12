#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod gesture;
mod model_download;
mod pipeline;
mod types;
mod ui;

use anyhow::Result;
use crossbeam_channel::bounded;
use gpui::Application;
use gpui_component;
use pipeline::RecognizerBackend;

fn main() -> Result<()> {
    env_logger::init();

    let (frame_to_rec_tx, frame_to_rec_rx) = bounded(1);

    let recognizer_backend = RecognizerBackend::default();

    Application::new()
        .with_assets(gpui_component_assets::Assets)
        .run(move |app| {
            gpui_component::init(app);

            if let Err(err) = ui::launch_ui(
                app,
                frame_to_rec_rx,
                frame_to_rec_tx,
                recognizer_backend.clone(),
            ) {
                eprintln!("failed to launch ui: {err:?}");
            }
        });

    Ok(())
}
