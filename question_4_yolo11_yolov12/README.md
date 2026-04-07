# YOLO11 and YOLOv12 Object Detection

This module provides two object detection workflows built around local clones of
[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) and
[sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12). Each model
has its own setup script, wrapper, input folder, and output folder.

## Overview

- `question_4_yolo11_yolov12/setup_yolo11_osc.sh` clones the Ultralytics
  repository into `external/ultralytics` and creates a repo-local virtual
  environment.
- `question_4_yolo11_yolov12/demo_yolo11.py` runs YOLO11 inference and copies
  annotated images to `question_4_yolo11_yolov12/outputs/yolo11/`.
- `question_4_yolo11_yolov12/setup_yolov12_osc.sh` clones the YOLOv12
  repository into `external/yolov12` and creates its repo-local virtual
  environment.
- `question_4_yolo11_yolov12/demo_yolov12.py` runs YOLOv12 inference and copies
  annotated images to `question_4_yolo11_yolov12/outputs/yolov12/`.

## Setup

Run from the repository root:

```bash
bash question_4_yolo11_yolov12/setup_yolo11_osc.sh
bash question_4_yolo11_yolov12/setup_yolov12_osc.sh
```

Both setup scripts standardize OSC on Python 3.9.18 and reject other Python
versions before rebuilding their repo-local environments.

OSC examples:

```bash
PYTHON_BIN=/path/to/python3.9 bash question_4_yolo11_yolov12/setup_yolov12_osc.sh
bash question_4_yolo11_yolov12/setup_yolov12_osc.sh
```

## Run

YOLO11 on OSC with one requested GPU:

```bash
bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- \
  python3.9 question_4_yolo11_yolov12/demo_yolo11.py \
  --device 0
```

YOLO11 on one local image with CPU inference:

```bash
python3.9 question_4_yolo11_yolov12/demo_yolo11.py \
  --source question_4_yolo11_yolov12/inputs/yolo11/000000001000.jpg \
  --device cpu
```

YOLO11 on a URL with CPU inference:

```bash
python3.9 question_4_yolo11_yolov12/demo_yolo11.py \
  --source https://ultralytics.com/images/bus.jpg \
  --device cpu
```

YOLOv12 on OSC with one requested GPU:

```bash
bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- \
  python3.9 question_4_yolo11_yolov12/demo_yolov12.py \
  --device 0
```

By default, `demo_yolov12.py` expects the repo-local YOLOv12 checkout at
`external/yolov12`. Pass `--repo-dir` if you cloned YOLOv12 somewhere else.

YOLOv12 on one local image with CPU inference:

```bash
python3.9 question_4_yolo11_yolov12/demo_yolov12.py \
  --source question_4_yolo11_yolov12/inputs/yolov12/000000007795.jpg \
  --device cpu
```

YOLOv12 on a URL with CPU inference:

```bash
python3.9 question_4_yolo11_yolov12/demo_yolov12.py \
  --source https://ultralytics.com/images/bus.jpg \
  --device cpu
```

## Inputs and Outputs

- YOLO11 reads from `question_4_yolo11_yolov12/inputs/yolo11/` by default and
  writes curated outputs to `question_4_yolo11_yolov12/outputs/yolo11/`.
- YOLOv12 reads from `question_4_yolo11_yolov12/inputs/yolov12/` by default and
  writes curated outputs to `question_4_yolo11_yolov12/outputs/yolov12/`.
- `--source` accepts a local image path, a directory, or an image URL.
- `--output` is for single-image runs only.
- `--output-dir` changes the destination directory when copying annotated
  outputs.
- Both wrappers use `question_4_yolo11_yolov12/runs/detect/` for the upstream
  run directory before copying final images into `outputs/`.

## Environment Notes

- On OSC, request the GPU node first with `bash osc_gpu_batch.sh` or open one
  with `bash osc_gpu_interactive.sh` before using `--device 0`.
- Use `--device cpu` on CPU-only systems or when you want to avoid GPU
  inference.
- Python 3.9.18 is the supported OSC baseline for both YOLO11 and YOLOv12.
- Supported input suffixes are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.
- `INSTALL_FLASH_ATTN=1 bash question_4_yolo11_yolov12/setup_yolov12_osc.sh`
  is not part of the supported Python 3.9.18 OSC baseline; only revisit it
  after separately revalidating the YOLOv12 CUDA stack outside this enforced
  baseline.
