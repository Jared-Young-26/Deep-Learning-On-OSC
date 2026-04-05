# Question 3: Faster R-CNN on OSC

This folder wraps the upstream [trzy/FasterRCNN](https://github.com/trzy/FasterRCNN)
project into a simpler assignment workflow. The upstream repo and pretrained
weights live under `external/`, while this folder keeps the assignment-facing
commands, sample inputs, and saved outputs.

## Canonical Files

- `setup_fasterrcnn_osc.sh`
- `download_models_fasterrcnn.sh`
- `demo_fasterrcnn.py`
- `inputs/`
- `outputs/`

## What Each Script Does

- `setup_fasterrcnn_osc.sh`: clones the upstream repo into `external/FasterRCNN`, builds a repo-local virtual environment, and installs the practical runtime for the current machine
- `download_models_fasterrcnn.sh`: downloads the backbone and detector checkpoints into the upstream repo where both the wrapper and upstream code expect them
- `demo_fasterrcnn.py`: resolves local files, URLs, or directories of images, launches the appropriate Faster R-CNN path, and saves annotated outputs into this folder

## Canonical Workflow

Run Question 3 in this order:

1. set up the upstream repo and environment
2. download the pretrained weights
3. run the demo on one image, one URL, or a directory of images

## Setup

Run from this folder:

```bash
bash setup_fasterrcnn_osc.sh
```

By default, the upstream clone is created in `../external/FasterRCNN`.

Then activate the environment and download the checkpoints:

```bash
source ../external/FasterRCNN/.venv/bin/activate
bash download_models_fasterrcnn.sh
```

## Demo Commands

Single URL image:

```bash
python3 demo_fasterrcnn.py \
  --framework tf2 \
  --weights fasterrcnn_tf2.h5 \
  --image http://trzy.org/files/fasterrcnn/gary.jpg \
  --mode to-file
```

Single local image:

```bash
python3 demo_fasterrcnn.py \
  --framework tf2 \
  --weights fasterrcnn_tf2.h5 \
  --image inputs/000000001000.jpg \
  --mode to-file
```

Recursive directory batch:

```bash
python3 demo_fasterrcnn.py \
  --framework tf2 \
  --weights fasterrcnn_tf2.h5 \
  --image inputs \
  --mode to-file
```

## Input and Output Behavior

- If `--image` is a URL, the wrapper downloads it into `inputs/downloaded/` first
- If `--image` is one local file and `--output` is omitted, the wrapper writes `outputs/<input filename>`
- If `--image` is a directory, the wrapper scans recursively and mirrors the relative folder structure under `outputs/`
- Batch directory mode skips `inputs/downloaded/` so cached URL examples do not mix with manual sample images
- `--mode viewer` only supports a single image input

Example:

- `inputs/example/cars.jpg` becomes `outputs/example/cars.jpg`

## Practical Notes

- `--mode to-file` is the best choice on headless OSC nodes
- On macOS, the practical local path is `--framework tf2`
- On CUDA-backed Linux/OSC systems, the PyTorch path is also available
- On macOS, use `download_models_fasterrcnn.sh` instead of the upstream `download_models.sh` because the upstream script assumes `wget`
- If your OSC environment requires modules, load Python first

## Pipeline Walkthrough

The clean walkthrough story is:

1. the setup script prepares the upstream repo and environment
2. the download script fetches pretrained checkpoints
3. the wrapper chooses the correct inference path
4. the model runs on one image or a batch
5. the wrapper saves the final annotated image into `outputs/`


