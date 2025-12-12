use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};

use anyhow::{Result, anyhow};
use crossbeam_channel::Sender;
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    query,
    utils::{
        ApiBackend, CameraIndex, CameraInfo, FrameFormat, RequestedFormat, RequestedFormatType,
    },
};
use rayon::prelude::*;

use crate::types::Frame;

// Prefer pixel formats that are widely supported on macOS (the built-in cameras
// often reject YUYV even though Nokhwa reports it).
const PREFERRED_PIXEL_FORMATS: &[FrameFormat] = &[
    FrameFormat::MJPEG,
    FrameFormat::NV12,
    FrameFormat::RAWRGB,
    FrameFormat::RAWBGR,
];

fn requested_formats() -> [RequestedFormat<'static>; 4] {
    [
        RequestedFormat::with_formats(
            RequestedFormatType::AbsoluteHighestFrameRate,
            PREFERRED_PIXEL_FORMATS,
        ),
        RequestedFormat::with_formats(
            RequestedFormatType::AbsoluteHighestResolution,
            PREFERRED_PIXEL_FORMATS,
        ),
        // Fall back to any format Nokhwa can decode, but prefer higher FPS to
        // avoid very low default rates (e.g. 15 FPS) that some drivers reject.
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
    ]
}

#[derive(Clone, Debug)]
pub struct CameraDevice {
    pub index: CameraIndex,
    pub label: String,
}

#[derive(Debug)]
pub struct CameraStream {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl CameraStream {
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CameraStream {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
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
    info.human_name()
}

fn build_camera(index: CameraIndex) -> Result<Camera> {
    let mut last_err = None;

    for requested in requested_formats() {
        match Camera::new(index.clone(), requested) {
            Ok(mut camera) => match camera.open_stream() {
                Ok(()) => return Ok(camera),
                Err(err) => last_err = Some(err.into()),
            },
            Err(err) => last_err = Some(err.into()),
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("failed to open camera with any supported format")))
}

pub fn start_camera_stream(index: CameraIndex, recog_tx: Sender<Frame>) -> Result<CameraStream> {
    // Fail fast before spawning the capture thread.
    build_camera(index.clone())?;

    let stop = Arc::new(AtomicBool::new(false));
    let stop_flag = stop.clone();

    let handle = thread::spawn(move || {
        let mut camera = match build_camera(index) {
            Ok(cam) => cam,
            Err(err) => {
                log::error!("failed to open camera: {err:?}");
                return;
            }
        };

        while !stop_flag.load(Ordering::Relaxed) {
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
            let pixel_count = rgb.len() / 3;
            let mut rgba_bytes = vec![0u8; pixel_count * 4];
            rgba_bytes
                .par_chunks_mut(4)
                .zip(rgb.par_chunks_exact(3))
                .for_each(|(dst, src)| {
                    dst[0] = src[0];
                    dst[1] = src[1];
                    dst[2] = src[2];
                    dst[3] = 255;
                });

            let frame_timestamp = Instant::now();
            let frame = Frame {
                rgba: rgba_bytes,
                width,
                height,
                timestamp: frame_timestamp,
            };

            // Drop if the worker is busy, otherwise forward every frame.
            let _ = recog_tx.try_send(frame);
        }
    });

    Ok(CameraStream {
        stop,
        handle: Some(handle),
    })
}
