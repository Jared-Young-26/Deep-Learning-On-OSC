# Faster R-CNN Object Detection

This module runs image object detection with a local clone of
[trzy/FasterRCNN](https://github.com/trzy/FasterRCNN). The wrapper accepts a
local file, an image URL, or a directory of images and saves annotated outputs
under this module.

The repo-owned scripts in this directory do not modify the upstream detector.
They standardize OSC setup, choose between the supported PyTorch and TF2 paths,
and keep the public CLI and output layout consistent for local review.

## Overview

- `question_3_faster_rcnn/setup_fasterrcnn_osc.sh` clones the upstream project
  into `external/FasterRCNN` and creates a repo-local virtual environment.
- `question_3_faster_rcnn/download_models_fasterrcnn.sh` downloads the supported
  TF2 and PyTorch checkpoints into the upstream clone.
- `question_3_faster_rcnn/demo_fasterrcnn.py` runs inference and writes
  annotated images to `question_3_faster_rcnn/outputs/` by default.

## Setup

Run from the repository root:

```bash
bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh
bash question_3_faster_rcnn/download_models_fasterrcnn.sh
```

The default upstream clone location is `external/FasterRCNN`.
The OSC setup requires Python 3.9.18 and installs both the CUDA PyTorch runtime
and the TF2 fallback by default so the demo can auto-select a usable backend at
run time.

For a lighter GPU-only PyTorch environment, opt out of the TF2 install
explicitly:

```bash
INSTALL_TF2=0 bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh
```

## Run

OSC GPU batch example for the CUDA-backed PyTorch path:

```bash
bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- \
  python3.9 question_3_faster_rcnn/demo_fasterrcnn.py \
  --framework pytorch \
  --mode to-file
```

Default run on the built-in sample URL. This uses `--framework auto`, which
prefers PyTorch on CUDA nodes and falls back to TF2 otherwise:

```bash
python3.9 question_3_faster_rcnn/demo_fasterrcnn.py --mode to-file
```

Run on one local image:

```bash
python3.9 question_3_faster_rcnn/demo_fasterrcnn.py \
  --image question_3_faster_rcnn/inputs/000000001000.jpg \
  --mode to-file
```

Run on a directory recursively:

```bash
python3.9 question_3_faster_rcnn/demo_fasterrcnn.py \
  --image question_3_faster_rcnn/inputs \
  --mode to-file
```

Force the CUDA-backed PyTorch path explicitly when needed:

```bash
python3.9 question_3_faster_rcnn/demo_fasterrcnn.py \
  --framework pytorch \
  --mode to-file
```

Force the CPU-capable TF2 path explicitly when needed:

```bash
python3.9 question_3_faster_rcnn/demo_fasterrcnn.py \
  --framework tf2 \
  --mode to-file
```

## Inputs and Outputs

- `--image` accepts a local image path, an image URL, or a directory processed
  recursively.
- URL inputs are downloaded into
  `question_3_faster_rcnn/inputs/downloaded/`.
- Outputs are written to `question_3_faster_rcnn/outputs/` unless `--output` is
  provided.
- Directory input mirrors the relative folder structure under the output
  directory.
- `--output` can be used for a single output file, a single-image output
  directory, or a batch output root.

## Environment Notes

- The wrapper defaults to `auto` and resolves to `pytorch` when CUDA is
  available in the repo-local environment, otherwise `tf2` when TensorFlow is
  installed.
- On OSC, the PyTorch path should be launched through `bash osc_gpu_batch.sh`
  or from a shell opened by `bash osc_gpu_interactive.sh`.
- Python 3.9.18 is the only supported OSC interpreter for this workflow.
- On OSC Jupyter/login-style CPU sessions, `framework: tf2` is expected because
  those sessions typically do not expose a CUDA device.
- `--mode to-file` is the practical choice for headless systems.
- `--mode viewer` opens an image window and is intended for GUI environments.
- `--framework pytorch` is strict and requires CUDA.
- `--framework tf2` is the supported CPU fallback and requires TensorFlow in the
  repo-local FasterRCNN environment.

## Troubleshooting

- If setup completed on an older revision and the TF2 fallback now fails with
  NumPy or Matplotlib import errors, remove `external/FasterRCNN/.venv` and rerun
  `bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh`.
- If you only need the CUDA-backed PyTorch path on a GPU node, use
  `INSTALL_TF2=0 bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh` for the
  faster GPU-only setup.
