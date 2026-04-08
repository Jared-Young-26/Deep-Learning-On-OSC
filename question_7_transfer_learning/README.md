# Transfer Learning Object Detection with TLlib

This module wraps TLlib's VOC-to-Clipart domain adaptation example into a
repeatable workflow with environment checks, dataset preparation, source-only
training, D-adapt phases, visualization, and summary generation.

The tracked shell and Python entrypoints are responsible for the OSC-specific
stability work in this directory: clone repair, dependency checks, compatibility
patches, profile defaults, and portable summary outputs. The upstream training
logic itself remains in the local TLlib checkout under `external/`.

Upstream references:

- TLlib object detection example:
  <https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/object_detection>
- D-adapt overview:
  <https://github.com/thuml/D-adapt>

## Overview

- `question_7_transfer_learning/setup_tllib_osc.sh` clones TLlib, creates the
  repo-local environment, installs the baseline dependencies, and can also run
  a repair-only compatibility preflight.
- `question_7_transfer_learning/run_tllib_osc.sh` runs the default end-to-end
  helper flow: self-heal the TLlib clone first, run doctor, then the
  smoke-profile full pipeline unless `PROFILE=benchmark` is set.
- `question_7_transfer_learning/demo_tllib_object_detection.py` exposes the
  wrapper's individual modes: `doctor`, `prepare-datasets`, `source-only`,
  `d-adapt`, `visualize`, `report`, and `full-pipeline`.

## Setup

Run from the repository root:

```bash
bash question_7_transfer_learning/setup_tllib_osc.sh
```

This setup script standardizes OSC on Python 3.9.18 and rejects other Python
minor versions before rebuilding `external/Transfer-Learning-Library/.venv`.
It also applies a small TLlib compatibility patch for newer `torchvision`
releases used on OSC.

Optional setup flags:

```bash
INSTALL_TORCH=1 bash question_7_transfer_learning/setup_tllib_osc.sh
INSTALL_DETECTRON2=1 bash question_7_transfer_learning/setup_tllib_osc.sh
REPAIR_ONLY=1 bash question_7_transfer_learning/setup_tllib_osc.sh
```

On Linux/OSC, the setup script now forces Detectron2 to build with GNU
`gcc/g++` by default. If your shell exposes NVHPC-style compilers first, rerun
setup with explicit overrides:

```bash
DETECTRON2_CC=$(command -v gcc) DETECTRON2_CXX=$(command -v g++) \
  INSTALL_TORCH=1 INSTALL_DETECTRON2=1 \
  bash question_7_transfer_learning/setup_tllib_osc.sh
```

The default upstream clone location is `external/Transfer-Learning-Library`.

## Self-Healing Behavior

- `run_tllib_osc.sh` is self-bootstrapping.
- If the TLlib virtualenv is missing, it runs full setup automatically.
- If the virtualenv already exists, it runs `REPAIR_ONLY=1` first so stale
  TLlib source files are patched before doctor or training starts.
- `run_assignment_osc.sh` uses the same repair-first logic during Q7 readiness.

## Doctor Check

Verify the environment before training:

```bash
python3.9 question_7_transfer_learning/demo_tllib_object_detection.py --mode doctor
```

If Q7 fails during readiness, rerun only the TLlib setup step first rather than
the full assignment:

```bash
env INSTALL_TORCH=1 INSTALL_DETECTRON2=1 \
  bash question_7_transfer_learning/setup_tllib_osc.sh
```

Run only the idempotent compatibility repair on an existing clone:

```bash
env REPAIR_ONLY=1 \
  bash question_7_transfer_learning/setup_tllib_osc.sh
```

The doctor mode checks the Python environment, required modules, dataset
locations, and resolved device selection.

If you still see `ImportError: cannot import name 'model_urls'` on OSC, update
the OSC checkout itself before rerunning Q7:

```bash
git pull
env REPAIR_ONLY=1 bash question_7_transfer_learning/setup_tllib_osc.sh
sed -n '1,20p' external/Transfer-Learning-Library/tllib/vision/models/resnet.py
```

The import near the top of `resnet.py` should be:

```python
from torchvision.models.resnet import BasicBlock, Bottleneck
```

## Profiles

- `smoke` uses small local subsets and shortened schedules for quick
  validation. It now validates the source-only stage only and skips D-adapt by
  default because the tiny proposal sets are often too small for stable
  adaptation batches.
- `benchmark` uses the full datasets and benchmark output locations. On OSC it
  now defaults all Detectron2 and D-adapt data-loader worker pools to `0` to
  avoid Slurm host-memory OOMs; raise them manually only if your allocation has
  headroom. The OSC preflight also repairs stale VOC-style annotation sizes in
  TLlib so Clipart examples with mismatched XML `width`/`height` metadata do
  not trip Detectron2 size checks.

## Smoke Profile

Run the full smoke pipeline:

```bash
python3.9 question_7_transfer_learning/demo_tllib_object_detection.py \
  --mode full-pipeline \
  --profile smoke \
  --download-datasets
```

Force a fresh smoke rerun:

```bash
python3.9 question_7_transfer_learning/demo_tllib_object_detection.py \
  --mode full-pipeline \
  --profile smoke \
  --download-datasets \
  --force
```

## Benchmark Profile

Run the helper script on a prepared environment:

```bash
bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- \
  bash question_7_transfer_learning/run_tllib_osc.sh
```

This is the recommended morning command after a plain `git pull`; the helper
now repairs or bootstraps the TLlib clone automatically before running Q7.

Run the helper with the benchmark profile instead:

```bash
bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 04:00:00 -- \
  env PROFILE=benchmark bash question_7_transfer_learning/run_tllib_osc.sh
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
bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 04:00:00 -- \
  python3.9 question_7_transfer_learning/demo_tllib_object_detection.py \
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
- Python 3.9.18 is the only supported OSC interpreter for this workflow.
- On OSC, the benchmark path should run inside a GPU allocation requested by
  `bash osc_gpu_batch.sh` or from a shell opened by `bash osc_gpu_interactive.sh`.
- Benchmark runs are intended for CUDA-enabled systems; the helper refuses
  benchmark execution on CPU unless `ALLOW_CPU=1` is set.
- `INSTALL_TORCH`, `INSTALL_DETECTRON2`, `PROFILE`, `ALLOW_CPU`, and `FORCE`
  are the primary environment variables exposed by the shell helpers.
