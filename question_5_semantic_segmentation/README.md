# Question 5: DOTA Satellite Detection MVP

Question 5 is implemented as a small, reproducible DOTA oriented-object
detection pipeline built around YOLO11 OBB.

The intended workflow is:

1. prepare the Ultralytics environment
2. bootstrap the dataset and reusable pretrained OBB checkpoint
3. fine-tune once on OSC
4. run forward-only demos on satellite images

## Canonical Files

- `setup_yolo11_osc.sh`
- `bootstrap_dota_obb.py`
- `train_dota_obb.py`
- `demo_yolo_segmentation.py`
- `q5_obb_common.py`
- `inputs/satellite_images/PUT_SATELLITE_IMAGES_HERE.txt`

Everything else is generated on demand. Datasets, checkpoints, and demo outputs
are intentionally not part of the committed structure.

## What Each Script Does

- `setup_yolo11_osc.sh`: prepares the repo-local Ultralytics environment used by every other Question 5 script
- `bootstrap_dota_obb.py`: downloads DOTAv1 plus the pretrained OBB checkpoint, extracts the raw dataset, tiles it into the split layout, and writes the dataset YAML
- `train_dota_obb.py`: fine-tunes the OBB detector and copies the best checkpoint into one stable alias path
- `demo_yolo_segmentation.py`: runs forward-only inference on satellite images, turns OBB detections into overlays, masks, and JSON summaries, and writes an aggregate CSV for batch runs
- `q5_obb_common.py`: shared path, import, model-resolution, download, and dataset helpers used by the other three scripts

## Setup

From the repo root:

```bash
bash question_5_semantic_segmentation/setup_yolo11_osc.sh
```

From inside `question_5_semantic_segmentation`:

```bash
bash setup_yolo11_osc.sh
```

## Bootstrap

From the repo root:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/bootstrap_dota_obb.py \
  --repo-dir external/ultralytics
```

From inside `question_5_semantic_segmentation`:

```bash
../external/ultralytics/.venv/bin/python \
  bootstrap_dota_obb.py \
  --repo-dir ../external/ultralytics
```

This generates:

- `question_5_semantic_segmentation/datasets/`
- `question_5_semantic_segmentation/datasets/DOTAv1-split.yaml`
- `question_5_semantic_segmentation/models/pretrained/yolo11s-obb.pt`

## Train on OSC

From the repo root:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/train_dota_obb.py \
  --repo-dir external/ultralytics \
  --device 0
```

From inside `question_5_semantic_segmentation`:

```bash
../external/ultralytics/.venv/bin/python \
  train_dota_obb.py \
  --repo-dir ../external/ultralytics \
  --device 0
```

The training script writes the reusable alias checkpoint to:

- `question_5_semantic_segmentation/models/dota_obb/best.pt`

## Forward-Only Demo

Put your satellite images in:

- `question_5_semantic_segmentation/inputs/satellite_images`

For the published repo, this folder keeps only a small curated sample set. Add
your own local images there when running new demos.

The demo never trains implicitly. It resolves models in this order:

1. `question_5_semantic_segmentation/models/dota_obb/best.pt`
2. `question_5_semantic_segmentation/models/pretrained/yolo11s-obb.pt`

From the repo root:

```bash
external/ultralytics/.venv/bin/python \
  question_5_semantic_segmentation/demo_yolo_segmentation.py \
  --repo-dir external/ultralytics \
  --device cpu
```

From inside `question_5_semantic_segmentation`:

```bash
../external/ultralytics/.venv/bin/python \
  demo_yolo_segmentation.py \
  --repo-dir ../external/ultralytics \
  --device cpu
```

Use `--device cpu` locally on a Mac. Use `--device 0` on OSC GPU nodes.
Omit `--keep-classes` to detect all DOTAv1 classes. Add it only when you want
to filter to a smaller subset.

## Input and Output Layout

Input images go here:

- `question_5_semantic_segmentation/inputs/satellite_images`

The batch demo writes results here:

- `question_5_semantic_segmentation/outputs/satellite_results`

Each image gets its own folder containing:

- `overlay.<original suffix>`
- `mask.png`
- `summary.json`

The batch run also writes:

- `index.csv`

Example:

```text
inputs/satellite_images/
  region_a/
    P0000.png

outputs/satellite_results/
  index.csv
  region_a/
    P0000/
      overlay.png
      mask.png
      summary.json
```

## DOTA Classes

The pipeline is aligned to DOTAv1 classes:

- `plane`
- `ship`
- `storage tank`
- `baseball diamond`
- `tennis court`
- `basketball court`
- `ground track field`
- `harbor`
- `bridge`
- `large vehicle`
- `small vehicle`
- `helicopter`
- `roundabout`
- `soccer ball field`
- `swimming pool`

Notes:

- `small vehicle` and `large vehicle` are the correct DOTA vehicle classes
- the pipeline does not guess exact `car` or `truck` labels
- `building` is not a stock DOTAv1 class


