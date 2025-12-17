use std::{
    thread,
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender};

use crate::{
    pipeline::skeleton,
    types::{Frame, GestureResult, RecognizedFrame},
};

const MAX_COMPOSITED_FPS: u64 = 30;
const MIN_COMPOSITED_FPS: u64 = 12;
const SLOWDOWN_FACTOR: f64 = 1.25;
const RECOVERY_FACTOR: f64 = 0.85;
const OVERLAY_CONFIDENCE_THRESHOLD: f32 = 0.5;

#[derive(Clone, Debug)]
pub struct CompositedFrame {
    pub frame: Frame,
    pub result: GestureResult,
}

pub fn start_frame_compositor(
    recognized_rx: Receiver<RecognizedFrame>,
) -> (Receiver<CompositedFrame>, thread::JoinHandle<()>) {
    let (tx, rx) = crossbeam_channel::bounded(1);
    let handle = thread::spawn(move || compositor_loop(recognized_rx, tx));
    (rx, handle)
}

fn compositor_loop(
    recognized_rx: Receiver<RecognizedFrame>,
    composited_tx: Sender<CompositedFrame>,
) {
    let min_interval = Duration::from_millis(1_000 / MAX_COMPOSITED_FPS);
    let max_interval = Duration::from_millis(1_000 / MIN_COMPOSITED_FPS);
    let mut target_interval = min_interval;

    while let Ok(mut recognized) = recognized_rx.recv() {
        while let Ok(newer) = recognized_rx.try_recv() {
            recognized = newer;
        }

        let mut frame = recognized.frame;
        let result = recognized.result;

        let compose_start = Instant::now();
        if !result.palm_regions.is_empty() {
            skeleton::draw_palm_regions(
                &mut frame.rgba,
                frame.width,
                frame.height,
                &result.palm_regions,
            );
        }
        if let Some(points) = overlay_points(&result) {
            skeleton::draw_skeleton(&mut frame.rgba, frame.width, frame.height, points);
        }
        let compose_time = compose_start.elapsed();

        let packet = CompositedFrame {
            frame,
            result: result.clone(),
        };
        let dropped_frame = composited_tx.try_send(packet).is_err();

        target_interval = adjust_interval(
            target_interval,
            compose_time,
            min_interval,
            max_interval,
            dropped_frame,
        );
        if let Some(sleep_for) = target_interval.checked_sub(compose_time) {
            if !sleep_for.is_zero() {
                thread::sleep(sleep_for);
            }
        }
    }
}

fn adjust_interval(
    current: Duration,
    compose_time: Duration,
    min_interval: Duration,
    max_interval: Duration,
    dropped_frame: bool,
) -> Duration {
    let current_secs = current.as_secs_f64();
    let compose_secs = compose_time.as_secs_f64();
    let min_secs = min_interval.as_secs_f64();
    let max_secs = max_interval.as_secs_f64();

    if dropped_frame && current < max_interval {
        Duration::from_secs_f64((current_secs * SLOWDOWN_FACTOR).min(max_secs))
    } else if compose_secs > current_secs && current < max_interval {
        Duration::from_secs_f64((compose_secs * SLOWDOWN_FACTOR).min(max_secs))
    } else if compose_secs * 1.5 < current_secs && current > min_interval {
        Duration::from_secs_f64((current_secs * RECOVERY_FACTOR).max(min_secs))
    } else {
        current
    }
}

fn overlay_points(result: &GestureResult) -> Option<&[(f32, f32)]> {
    if result.confidence >= OVERLAY_CONFIDENCE_THRESHOLD {
        result.landmarks.as_deref()
    } else {
        None
    }
}
