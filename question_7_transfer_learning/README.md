# Transfer Learning Object Detection with TLlib

This module wraps TLlib's VOC-to-Clipart domain adaptation example into a
repeatable workflow with environment checks, dataset preparation, source-only
training, D-adapt phases, visualization, and summary generation.

Upstream references:

- TLlib object detection example:
  <https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/object_detection>
- D-adapt overview:
  <https://github.com/thuml/D-adapt>

## Overview

- `question_7_transfer_learning/setup_tllib_osc.sh` clones TLlib, creates the
  repo-local environment, and installs the baseline dependencies.
- `question_7_transfer_learning/run_tllib_osc.sh` runs the default end-to-end
  helper flow: doctor check first, then the full pipeline.
- `question_7_transfer_learning/demo_tllib_object_detection.py` exposes the
  wrapper's individual modes: `doctor`, `prepare-datasets`, `source-only`,
  `d-adapt`, `visualize`, `report`, and `full-pipeline`.

## Setup

Run from the repository root:

```bash
bash question_7_transfer_learning/setup_tllib_osc.sh
```

Optional setup flags:

```bash
INSTALL_TORCH=1 bash question_7_transfer_learning/setup_tllib_osc.sh
INSTALL_DETECTRON2=1 bash question_7_transfer_learning/setup_tllib_osc.sh
```

The default upstream clone location is `external/Transfer-Learning-Library`.

## Doctor Check

Verify the environment before training:

```bash
python3 question_7_transfer_learning/demo_tllib_object_detection.py --mode doctor
```

The doctor mode checks the Python environment, required modules, dataset
locations, and resolved device selection.

## Profiles

- `smoke` uses small local subsets and shortened schedules for quick
  validation.
- `benchmark` uses the full datasets and benchmark output locations.

## Smoke Profile

Run the full smoke pipeline:

```bash
python3 question_7_transfer_learning/demo_tllib_object_detection.py \
  --mode full-pipeline \
  --profile smoke \
  --download-datasets
```

Force a fresh smoke rerun:

```bash
python3 question_7_transfer_learning/demo_tllib_object_detection.py \
  --mode full-pipeline \
  --profile smoke \
  --download-datasets \
  --force
```

## Benchmark Profile

Run the helper script on a prepared environment:

```bash
bash question_7_transfer_learning/run_tllib_osc.sh
```

Run the helper with a smoke profile instead:

```bash
PROFILE=smoke bash question_7_transfer_learning/run_tllib_osc.sh
```

Force a fresh rerun:

```bash
FORCE=1 bash question_7_transfer_learning/run_tllib_osc.sh
```

Allow CPU execution explicitly when CUDA is unavailable:

```bash
ALLOW_CPU=1 PROFILE=smoke bash question_7_transfer_learning/run_tllib_osc.sh
```

## Direct Wrapper Usage

Run the benchmark pipeline directly through the Python wrapper:

```bash
python3 question_7_transfer_learning/demo_tllib_object_detection.py \
  --mode full-pipeline \
  --profile benchmark \
  --download-datasets \
  --device cuda \
  --phase-count 3
```

## Outputs

Smoke runs write to:

- `question_7_transfer_learning/logs/source_only_smoke/`
- `question_7_transfer_learning/logs/d_adapt_smoke/`
- `question_7_transfer_learning/visualizations/voc2clipart_smoke/`
- `question_7_transfer_learning/outputs/voc2clipart_smoke/summary.json`
- `question_7_transfer_learning/outputs/voc2clipart_smoke/summary.md`

Benchmark runs write to:

- `question_7_transfer_learning/logs/source_only_benchmark/`
- `question_7_transfer_learning/logs/d_adapt_benchmark/`
- `question_7_transfer_learning/visualizations/voc2clipart_benchmark/`
- `question_7_transfer_learning/outputs/voc2clipart_benchmark/summary.json`
- `question_7_transfer_learning/outputs/voc2clipart_benchmark/summary.md`

## Environment Notes

- `run_tllib_osc.sh` uses the TLlib virtualenv Python at
  `external/Transfer-Learning-Library/.venv/bin/python` by default.
- Benchmark runs are intended for CUDA-enabled systems; the helper refuses
  benchmark execution on CPU unless `ALLOW_CPU=1` is set.
- `INSTALL_TORCH`, `INSTALL_DETECTRON2`, `PROFILE`, `ALLOW_CPU`, and `FORCE`
  are the primary environment variables exposed by the shell helpers.



