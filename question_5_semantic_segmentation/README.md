# Semantic Segmentation with YOLO11 and iSAID

This module builds a semantic segmentation pipeline for aerial imagery using
YOLO11 segmentation and the iSAID dataset. The workflow prepares a repo-local
Ultralytics environment, bootstraps the dataset into YOLO segmentation format,
fine-tunes the model, and exports per-pixel semantic outputs for input images.

The repo-owned scripts here intentionally separate three concerns: environment
setup, dataset normalization, and model execution. That keeps the OSC workflow
reusable even though the upstream Ultralytics project expects a different
project layout than this assignment repository.

## Overview

- `question_5_semantic_segmentation/setup_yolo11_osc.sh` prepares the
  repo-local Ultralytics environment and can bootstrap the dataset
  automatically.
- `question_5_semantic_segmentation/bootstrap_isaid_seg.py` normalizes the raw
  iSAID layout, converts COCO annotations into YOLO segmentation labels,
  downloads `yolo11s-seg.pt`, and writes the dataset YAML.
- `question_5_semantic_segmentation/train_isaid_seg.py` fine-tunes the
  segmentation model and copies the reusable checkpoint to
  `question_5_semantic_segmentation/models/isaid_seg/best.pt`.
- `question_5_semantic_segmentation/demo_yolo_segmentation.py` runs inference on
  satellite images and writes overlays, class maps, colorized masks, per-image
  summaries, and a batch index CSV.

## Setup

Repository-root setup command:

```bash
bash question_5_semantic_segmentation/setup_yolo11_osc.sh
```

The setup script standardizes OSC on Python 3.9.18 before it creates or reuses
`external/ultralytics/.venv`.

By default, setup also bootstraps the dataset and downloads the pretrained
`yolo11s-seg.pt` checkpoint. To refresh only the Python environment and skip the
dataset step:

```bash
AUTO_BOOTSTRAP_DATASET=0 bash question_5_semantic_segmentation/setup_yolo11_osc.sh
```

The default upstream clone location is `external/ultralytics`.

## Dataset Bootstrap

Dataset bootstrap command:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/bootstrap_isaid_seg.py \
  --repo-dir external/ultralytics
```

Dataset bootstrap with raw-download step:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/bootstrap_isaid_seg.py \
  --repo-dir external/ultralytics \
  --download-dataset
```

Bootstrap writes these key artifacts:

- `question_5_semantic_segmentation/datasets/isaid_seg.yaml`
- `question_5_semantic_segmentation/datasets/raw/isaid/labels/train/`
- `question_5_semantic_segmentation/datasets/raw/isaid/labels/val/`
- `question_5_semantic_segmentation/models/pretrained/yolo11s-seg.pt`

The normalized raw dataset layout is:

```text
question_5_semantic_segmentation/datasets/raw/isaid/
  images/
    train/
    val/
  annotations/
    instances_train.json
    instances_val.json
```

## Training

Repository-root training command:

```bash
bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 04:00:00 -- \
  external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/train_isaid_seg.py \
  --repo-dir external/ultralytics \
  --device 0 \
  --epochs 20 \
  --imgsz 768 \
  --batch 1 \
  --workers 0 \
  --close-mosaic 20 \
  --exist-ok
```

Default training settings are:

- model: `yolo11s-seg.pt`
- epochs: `20`
- imgsz: `768`
- batch: `1`
- workers: `0`
- close_mosaic: `20`
- per-epoch validation: disabled on OSC; Ultralytics still validates on the final epoch

The reusable checkpoint alias is written to:

- `question_5_semantic_segmentation/models/isaid_seg/best.pt`

The canonical resumable checkpoint is:

- `question_5_semantic_segmentation/runs/segment/train/isaid_yolo11s_seg/weights/last.pt`

Resumed training command from `last.pt`:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/train_isaid_seg.py \
  --repo-dir external/ultralytics \
  --device 0 \
  --resume
```

The repo-root orchestrator follows the same rule automatically: it skips Q5 if
`models/isaid_seg/best.pt` already exists, resumes from `last.pt` when that
alias is missing, and otherwise starts a fresh run.

Short CPU validation command:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/train_isaid_seg.py \
  --repo-dir external/ultralytics \
  --device cpu \
  --epochs 1 \
  --batch 1 \
  --workers 0
```

## Inference

Batch inference on the default input directory:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/demo_yolo_segmentation.py \
  --repo-dir external/ultralytics \
  --device cpu
```

Single-image inference:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/demo_yolo_segmentation.py \
  --repo-dir external/ultralytics \
  --source question_5_semantic_segmentation/inputs/satellite_images/P0362.png \
  --device cpu
```

Filtered-class inference:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/demo_yolo_segmentation.py \
  --repo-dir external/ultralytics \
  --keep-classes "ship,small vehicle" \
  --device cpu
```

Model resolution order is:

1. `question_5_semantic_segmentation/models/isaid_seg/best.pt`
2. `question_5_semantic_segmentation/models/pretrained/yolo11s-seg.pt`

## Inputs and Outputs

Input images go in:

- `question_5_semantic_segmentation/inputs/satellite_images/`

Batch outputs are written to:

- `question_5_semantic_segmentation/outputs/satellite_results/`

Each image gets its own output directory containing:

- `overlay.<original suffix>`
- `class_ids.png`
- `mask.png`
- `summary.json`

The batch run also writes:

- `index.csv`

Example:

```text
question_5_semantic_segmentation/outputs/satellite_results/
  index.csv
  P0362/
    overlay.png
    class_ids.png
    mask.png
    summary.json
```

`class_ids.png` is a one-channel semantic class map where `0` is background and
positive values correspond to the model's class ids. `mask.png` is the
colorized semantic mask, and `overlay.*` blends that mask with the original
image.

## Environment Notes

- Python 3.9.18 is the supported OSC baseline for this workflow.
- On OSC, launch GPU training through `bash osc_gpu_batch.sh` or from a shell
  opened by `bash osc_gpu_interactive.sh`.
- The OSC stability fix is in the Q5 training defaults (`imgsz=768`, `batch=1`,
  `workers=0`, `close_mosaic=20`), reduced validation frequency, and resume
  behavior, not in extra Slurm `--mem` flags.
- The first bootstrap run downloads several gigabytes of dataset assets and can
  take time to extract and convert.
- Setup and inference work on CPU. Full training is best on a CUDA-capable
  system with `--device 0`.
