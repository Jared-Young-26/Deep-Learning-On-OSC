# Faster R-CNN Object Detection

This module runs image object detection with a local clone of
[trzy/FasterRCNN](https://github.com/trzy/FasterRCNN). The wrapper accepts a
local file, an image URL, or a directory of images and saves annotated outputs
under this module.

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

## Run

Default run on the built-in sample URL:

```bash
python3 question_3_faster_rcnn/demo_fasterrcnn.py --mode to-file
```

Run on one local image:

```bash
python3 question_3_faster_rcnn/demo_fasterrcnn.py \
  --image question_3_faster_rcnn/inputs/000000001000.jpg \
  --mode to-file
```

Run on a directory recursively:

```bash
python3 question_3_faster_rcnn/demo_fasterrcnn.py \
  --image question_3_faster_rcnn/inputs \
  --mode to-file
```

Select a framework explicitly when needed:

```bash
python3 question_3_faster_rcnn/demo_fasterrcnn.py \
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

- The wrapper defaults to `tf2` on macOS and `pytorch` on other platforms.
- `--mode to-file` is the practical choice for headless systems.
- `--mode viewer` opens an image window and is intended for GUI environments.

