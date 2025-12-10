use std::{thread, time::Instant};

use anyhow::Result;
use crossbeam_channel::Sender;
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    query,
    utils::{ApiBackend, CameraIndex, CameraInfo, RequestedFormat, RequestedFormatType},
};

use crate::types::Frame;

#[derive(Clone, Debug)]
pub struct CameraDevice {
    pub index: CameraIndex,
    pub label: String,
}

pub fn available_cameras() -> Result<Vec<CameraDevice>> {
    let cameras = query(ApiBackend::Auto)?;
    Ok(cameras
        .into_iter()
        .map(|info| CameraDevice {
            index: info.index().clone(),
            label: format_camera_label(&info),
        })
        .collect())
}

fn format_camera_label(info: &CameraInfo) -> String {
    let name = info.human_name();
    let desc = info.description().trim();
    let index = info.index().as_string();
    if desc.is_empty() || desc == "N/A" {
        format!("{name} (#{index})")
    } else {
        format!("{name} ({desc}, #{index})")
    }
}

fn build_camera(index: CameraIndex) -> Result<Camera> {
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
    let mut camera = Camera::new(index, requested)?;
    camera.open_stream()?;
    Ok(camera)
}

pub fn start_camera_stream(
    index: CameraIndex,
    ui_tx: Sender<Frame>,
    recog_tx: Sender<Frame>,
) -> Result<thread::JoinHandle<()>> {
    // Fail fast before spawning the capture thread.
    build_camera(index.clone())?;

    let handle = thread::spawn(move || {
        let mut camera = match build_camera(index) {
            Ok(cam) => cam,
            Err(err) => {
                log::error!("failed to open camera: {err:?}");
                return;
            }
        };

        loop {
            let frame = match camera.frame() {
                Ok(frame) => frame,
                Err(err) => {
                    log::warn!("camera frame read failed: {err:?}");
                    continue;
                }
            };

            let decoded = match frame.decode_image::<RgbFormat>() {
                Ok(img) => img,
                Err(err) => {
                    log::warn!("failed to decode camera frame: {err:?}");
                    continue;
                }
            };

            let (width, height) = decoded.dimensions();
            let rgb = decoded.into_raw();
            if rgb.is_empty() {
                continue;
            }

            // Expand RGB to RGBA for the UI pipeline.
            let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
            for chunk in rgb.chunks_exact(3) {
                rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }

            let frame = Frame {
                rgba,
                width,
                height,
                timestamp: Instant::now(),
            };

            let _ = ui_tx.try_send(frame.clone());
            let _ = recog_tx.try_send(frame);
        }
    });

    Ok(handle)
}
