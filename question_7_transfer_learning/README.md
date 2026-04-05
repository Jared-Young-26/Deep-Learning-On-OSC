# Question 7: TLlib Transfer Learning Object Detection

This folder packages TLlib's VOC -> Clipart object-detection example into a
repeatable assignment workflow. The wrapper is designed to support both local
validation and a fuller OSC benchmark run.

Upstream references:

- TLlib object detection example:
  <https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/object_detection>
- D-adapt overview:
  <https://github.com/thuml/D-adapt>

## Canonical Files

- `setup_tllib_osc.sh`
- `run_tllib_osc.sh`
- `demo_tllib_object_detection.py`

## What Each Script Does

- `setup_tllib_osc.sh`: clones TLlib, builds the repo-local environment, installs the baseline dependencies, and optionally installs Torch or Detectron2
- `run_tllib_osc.sh`: runs the standard assignment workflow by checking the environment first and then launching the full benchmark pipeline
- `demo_tllib_object_detection.py`: the main wrapper that supports `doctor`, dataset preparation, source-only training, D-adapt phases, visualization, report generation, and the full end-to-end pipeline

## What This Question Demonstrates

The intended demonstration is:

1. start from TLlib's pretrained Faster R-CNN backbone
2. train a `source-only` detector on VOC2007 + VOC2012
3. evaluate that detector on the target domain `Clipart`
4. run one or more D-adapt phases on the same VOC -> Clipart task
5. compare the target-domain results before and after adaptation

The important evidence is:

- the TLlib environment is working on OSC
- the source-only baseline produces target-domain metrics on Clipart
- the adapted detector produces its own target-domain metrics on Clipart
- you can show logs, checkpoints, summary files, and qualitative predictions

## Two Workflows

This folder intentionally supports two workflows:

- `smoke`: local validation only. It uses tiny subsets and very short runs so you can prove the wrapper works on a laptop
- `benchmark`: assignment-grade workflow. It uses the full VOC and Clipart datasets, benchmark output folders, and upstream-style defaults

Use `smoke` to debug locally. Use `benchmark` on OSC for the actual
demonstration.

In practice, the script flow is:

1. `doctor` confirms the environment and dataset roots
2. `source-only` establishes the baseline detector
3. `d-adapt` refines that detector phase by phase
4. `visualize` and `report` turn the run into presentation-friendly evidence

## Setup

Run from the repository root:

```bash
bash question_7_transfer_learning/setup_tllib_osc.sh
```

Optional:

```bash
INSTALL_TORCH=1 bash question_7_transfer_learning/setup_tllib_osc.sh
INSTALL_DETECTRON2=1 bash question_7_transfer_learning/setup_tllib_osc.sh
```

Notes:

- the real target for this question is OSC/Linux with CUDA
- Apple Silicon macOS is useful for local validation, not for the final assignment run
- `setup_tllib_osc.sh` installs TLlib, `timm`, and optionally Detectron2 into `external/Transfer-Learning-Library/.venv`

## Doctor Check

Before training, verify the environment:

```bash
python3 question_7_transfer_learning/demo_tllib_object_detection.py --mode doctor
```

You want to see:

- `detectron2` available
- full `VOC2007`, `VOC2012`, and `Clipart` dataset paths marked `ready`
- on OSC, `resolved_model_device: cuda`

## Local Validation With Smoke Profile

This is the laptop-safe validation run:

```bash
python3 question_7_transfer_learning/demo_tllib_object_detection.py \
  --mode full-pipeline \
  --profile smoke \
  --download-datasets
```

What it proves:

- the wrapper can download data
- source-only training runs
- D-adapt runs through its reduced smoke configuration
- visualization and summary generation work

What it does not prove:

- meaningful transfer performance
- assignment-grade results
- anything close to the upstream VOC -> Clipart benchmark numbers

If a previous smoke run already exists, use `--force` to rebuild it:

```bash
python3 question_7_transfer_learning/demo_tllib_object_detection.py \
  --mode full-pipeline \
  --profile smoke \
  --download-datasets \
  --force
```

## Assignment-Grade OSC Run

This is the intended assignment workflow on an OSC GPU node:

```bash
bash question_7_transfer_learning/run_tllib_osc.sh
```

What `run_tllib_osc.sh` does:

1. uses the TLlib virtualenv Python directly
2. checks whether CUDA is available
3. runs `--mode doctor`
4. runs the full `benchmark` pipeline with full datasets
5. runs the default D-adapt phase sequence

It refuses to run the benchmark flow if CUDA is missing unless you explicitly
override that safeguard with `ALLOW_CPU=1`.

For local CPU-only fallback, prefer smoke mode:

```bash
ALLOW_CPU=1 PROFILE=smoke bash question_7_transfer_learning/run_tllib_osc.sh
```

For a fresh rerun into the benchmark output folders:

```bash
FORCE=1 bash question_7_transfer_learning/run_tllib_osc.sh
```

To pass extra overrides through to the wrapper:

```bash
bash question_7_transfer_learning/run_tllib_osc.sh --n-visualizations 8
```

## Direct Wrapper Command

If you prefer to run the wrapper yourself instead of using the shell helper:

```bash
python3 question_7_transfer_learning/demo_tllib_object_detection.py \
  --mode full-pipeline \
  --profile benchmark \
  --download-datasets \
  --device cuda \
  --phase-count 3
```

Important behavior for `benchmark`:

- it uses the full datasets under `question_7_transfer_learning/datasets/`
- it writes to benchmark-specific output folders, not the smoke folders
- it does not apply the smoke-specific shortened detector settings
- it defaults to a multi-phase D-adapt sequence for VOC -> Clipart

## Benchmark Artifacts

The assignment-grade run writes here by default:

- source-only logs/checkpoints: `question_7_transfer_learning/logs/source_only_benchmark/...`
- D-adapt logs/checkpoints: `question_7_transfer_learning/logs/d_adapt_benchmark/...`
- visualizations: `question_7_transfer_learning/visualizations/voc2clipart_benchmark/...`
- summary:
  - `question_7_transfer_learning/outputs/voc2clipart_benchmark/summary.json`
  - `question_7_transfer_learning/outputs/voc2clipart_benchmark/summary.md`

These are the files you should use to demonstrate the assignment result.

## What To Show In A Presentation

The cleanest demonstration bundle is:

1. `doctor` output showing the environment and `resolved_model_device: cuda`
2. source-only `log.txt` and `model_final.pth`
3. final D-adapt phase `log.txt` and `model_final.pth`
4. `outputs/voc2clipart_benchmark/summary.md`
5. source-only visualization images
6. adapted visualization images

That gives you both:

- quantitative evidence: `AP`, `AP50`, `AP75`
- qualitative evidence: before/after prediction images
