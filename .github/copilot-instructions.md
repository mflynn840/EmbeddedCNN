## Purpose

This file gives concise, actionable guidance for an AI coding agent to be productive in the EmbeddedCNN repo. It focuses on the repo layout, the end-to-end model -> firmware flow, developer commands, and project-specific conventions found in code.

## Big picture (what this repo does)
- Python ML pipeline (under `python/ML`) trains a tiny PyTorch CNN (TinyCNN) on MNIST, exports it to ONNX → TensorFlow → quantized TFLite, and generates a C array for embedding.
- Microcontroller firmware (under `microcontroller/`) is an Arduino/PlatformIO ESP32 sketch that expects a TFLite model compiled into C (tflite-micro) and accepts 28x28 grayscale frames over serial for inference.
- Camera tooling (under `python/Camera`) streams frames and demonstrates sending 28x28 grayscale frames to the ESP32.

## Key files and where to look
- Training and model orchestration: `python/ML/train.py`, `python/ML/cnn_module.py`, `python/ML/mnist_module.py`.
- Export pipeline (PyTorch → ONNX → TF → TFLite → C): `python/ML/embed_model.py`, `python/ML/embed_model_utils.py`, `python/ML/quant_utils.py`.
- Representative dataset generator for TFLite quantization: `python/ML/quant_utils.py` (yields NHWC numpy arrays).
- ESP32 firmware entry point: `microcontroller/src/main.cpp` and PlatformIO config: `microcontroller/platformio.ini`.
- Camera demo/stream: `python/Camera/streamCam.py` (shows how 28x28 grayscale frames are prepared and displayed).
- Install dependencies: `python/requirements.txt` (large pinned set used for model conversion and PyTorch/TF tooling).

## Primary workflows & commands
- Local Python environment: from repository root
  - Install Python deps: `pip install -r python/requirements.txt`
  - Train a model (quick): `python -m ML.train` (uses PyTorch Lightning, trainer configured in `python/ML/train.py`).
  - Export an embeddable model: `python -m ML.embed_model` or run `python/ML/embed_model.py` — this runs prune→onnx→tf→tflite→c-array using helper functions in `embed_model_utils.py`.

- PlatformIO / ESP32 firmware:
  - Build: from `microcontroller/` run `platformio run`
  - Upload: `platformio run -t upload -e esp32dev`
  - Serial monitor: `platformio device monitor -b 115200`

- Camera demo (sends frames locally, not uploaded by default): run `python python/Camera/streamCam.py` — it opens an MJPEG stream at `tcp://127.0.0.1:8080/feed.mjpg` and resizes frames to 28x28 bytes.

## Data flow / integration points (important for code edits)
1. Train: `TinyCnnModule` (PyTorch Lightning) saves state_dicts (see `cnn_module.py`).
2. Export: `embed_model_utils.onnx_export` expects a PyTorch model with input shape (1,1,28,28); uses opset 17.
3. TF conversion: `onnx_to_tflow` uses `onnx2tf.convert` with `keep_nchw_or_ncdhw_input_names=True` (PyTorch NCHW->TFLite NHWC handling is performed by `representative_data_gen_from_loader`).
4. Quantization: `tflow_to_quant_tflite` configures the TFLiteConverter for int8 and requires a representative dataset that yields NHWC float32 numpy arrays.
5. Firmware: `tflite_to_c_array` outputs a `.cc` with an unsigned char array and `_len` symbol — include or copy this into `microcontroller/` (there is no automatic sync currently).

## Project-specific conventions & gotchas
- The code assumes MNIST-sized input (28x28 grayscale) and a TinyCNN architecture (`ML/cnn_module.py`) — any new dataset or input size will require editing multiple pipeline steps.
- Representative dataset yields NHWC arrays (images permuted in `quant_utils.py`). Keep this when adding new datasets.
- Export pipeline writes intermediates to `python/ML/models/{onnx,tf,tflite,C}` — the C output needs to be manually moved into the microcontroller project or referenced from there.
- The PlatformIO config includes `lib_deps` for `tflite-micro` and `flatbuffers` and uses `-std=gnu++17` plus several include paths; do not re-order or remove these flags without confirming build impact.
- `microcontroller/src/main.cpp` currently contains TODOs for loading the TFLite model and running inference — the firmware is minimal and intended as a starting point for integration.

## Useful examples for quick edits
- To add a new model name and produce firmware-ready C: run the export flow with `model_name` (see `embed_model.export_as_embeddable`) and then copy `python/ML/models/C/<model>.cc` into `microcontroller/lib/codegen/` or `microcontroller/src/` and include it from `main.cpp`.
- To change input processing on the device, mirror the same preprocessing done in `python/Camera/streamCam.py` (convert to grayscale, resize to 28x28, send raw bytes) so the on-device interpreter receives identical byte ordering.

## Tests, linting, CI
- There are no automated tests or CI config files detectable in the repo root. Use local PlatformIO and the Python environment for validation.

## When in doubt (assumptions for the agent)
- Assume models are trained for MNIST-like single-channel 28x28 inputs.
- The export pipeline is single-machine and relies on installed TF/ONNX/onnx2tf toolings — these are pinned in `python/requirements.txt`.
- The final integration (copying the `.cc` file into `microcontroller/`) is manual — the repo doesn't currently automate that step.

---
If any of the assumptions above are incorrect or you want me to add step-by-step automation (e.g. a Makefile to copy the generated C into the firmware tree and run `platformio run && platformio run -t upload`), tell me which part to automate and I will update the repo.
