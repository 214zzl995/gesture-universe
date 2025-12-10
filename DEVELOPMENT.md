Gesture Universe — Development Plan
===================================

Scope and goals
- Desktop Rust app with a `gpui` window.
- Live camera feed drawn in the top-left.
- Continuous gesture recognition from camera frames.
- Current gesture label (and optional confidence) shown in the center of the UI.
- Favor native Rust; inference is handled by Candle/Burn/Tract to run the `handpose_estimation_mediapipe` ONNX models.

System prerequisites
- Rust toolchain (edition 2024).
- `handpose_estimation_mediapipe` ONNX models present locally (already checked into `handpose_estimation_mediapipe/`).
- Backend-specific notes:
  - Candle: CPU-only works out of the box; Metal/CUDA backends need the corresponding toolchains/drivers.
  - Burn: WGPU backend needs a working GPU driver (Metal/Vulkan/DX12).
  - Tract: Pure CPU by default; uses `blas` if available for speed.
- GPU drivers adequate for `gpui` (Metal on macOS, Vulkan/DX12 on other platforms).

Crates and dependencies (planned)
- gpui: windowing/rendering and layout.
- nokhwa (or similar) for cross-platform camera capture without OpenCV.
- tract-onnx (baseline) for CPU inference; optional feature flags to swap to candle/candle-onnx or burn-wgpu for acceleration.
- crossbeam-channel (or std sync primitives): frame/result pipelines between threads.
- anyhow + thiserror: ergonomic error handling.
- image: format helpers for RGBA conversion.
- log/env_logger (optional): diagnostics.

High-level architecture
- `app`: starts `gpui::Application`, creates the window, owns app state.
- `camera`: opens the default webcam via `nokhwa` (or chosen backend), converts frames to RGBA, pushes frames onto a channel.
- `recognizer`: consumes frames, runs the MediaPipe handpose model (ONNX) via the selected backend, publishes `GestureResult`.
- `ui/view`: renders the latest frame (top-left) and the latest gesture label (center).
- `state`: shared structures (e.g., `AppState { latest_frame, latest_result }`) guarded by channels or `Arc<Mutex<...>>`.

Data flow
1) Camera thread: capture -> convert frame to RGBA bytes -> send over `frame_tx`.
2) Recognition worker: recv frame -> run palm detection + 21-keypoint hand pose (handpose_estimation_mediapipe) -> map to gesture label -> send `GestureResult` over `result_tx`.
3) UI: subscribes to latest frame/result; on updates, requests a re-render.

UI layout (initial)
- Background: simple solid color.
- Top-left: live camera view (scaled to a fixed size or aspect-preserving fit).
- Center: text showing gesture name (and confidence % if available); fallback to “No gesture detected”.

Concurrency & threading
- Camera capture runs in a dedicated thread (blocking OpenCV read).
- Recognition can share the camera thread or run in another worker; start with one worker for simplicity.
- UI runs on the `gpui` main thread; use channels to deliver updates and `cx.notify()`/state invalidation to trigger repaint.

Error handling
- Fail fast on camera open errors with user-facing message.
- Drop/skip frames on transient read errors; log them.
- If recognition fails for a frame, keep the previous result and continue; surface backend errors with enough context (selected runtime, model path).

Gesture recognition approach (phased)
- Phase 1: placeholder heuristic (brightness/motion) to exercise the pipeline.
- Phase 2: load `handpose_estimation_mediapipe_2023feb.onnx` via `tract-onnx` (CPU), implement preprocessing (crop/resize/normalize to expected input) and postprocessing to decode 21 keypoints; derive simple gesture labels (e.g., 0–9) from keypoints.
- Phase 3: add backend switch + acceleration:
  - Candle: enable `candle-onnx` for Metal/CUDA; ensure tensor layout matches the model.
  - Burn: use `burn-wgpu` for GPU inference; keep a CPU fallback.
- Phase 4 (optional): quantized model evaluation (`*_int8*.onnx`), batching, and performance tuning.

Testing strategy
- Unit: frame conversion utilities (camera frame -> RGBA bytes), gesture result formatting, model preprocessing/postprocessing math.
- Integration (manual): run the app, verify camera feed and gesture text updates; test camera unplug/replug and backend switching (tract vs candle/burn).
- Regression: compare a few canned frames against expected keypoint outputs from the Python demo for consistency.
- Logging: enable `RUST_LOG=info` during manual runs.

Build/run
- `cargo run` to start the app (defaulting to tract/CPU).
- Backend selection (planned via features/env): choose one of `tract` (default), `candle`, or `burn`; set hardware flags (e.g., `CANDLE_USE_CUDA=1`) as needed.
- Models are loaded from `handpose_estimation_mediapipe/*.onnx`; ensure the desired file exists before running.
- If GPU backends fail to init, fall back to CPU and log a warning.

Directory conventions (planned)
- `src/main.rs`: app entry; bootstrap gpui.
- `src/camera.rs`: camera capture utilities (nokhwa backend, RGBA conversion).
- `src/recognizer.rs`: gesture recognition logic (pre/postprocess + backend abstraction).
- `src/ui.rs`: gpui view/components and state wiring.
- `src/types.rs`: shared structs/enums (`GestureResult`, etc.).
- `handpose_estimation_mediapipe/`: ONNX models + reference docs.

Open questions to revisit
- Which gestures and labels are in scope for the first cut?
- Performance targets (FPS, latency) and acceptable CPU/GPU usage?
- Should we record/debug-save frames on errors (disk write policy)?

Next implementation steps
- Add dependencies to `Cargo.toml` (gpui, camera backend, tract-onnx + feature flags for candle/burn, anyhow, thiserror, crossbeam-channel, image, logging).
- Scaffold modules and basic state.
- Implement camera capture loop and RGBA conversion.
- Implement model loader + backend abstraction; run `handpose_estimation_mediapipe_2023feb.onnx` via tract/CPU first.
- Wire gpui view to display the latest frame and gesture label.
- Add placeholder recognizer, then replace with ONNX-backed inference; test end-to-end manually on webcam input.
