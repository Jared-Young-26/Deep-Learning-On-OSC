#!/usr/bin/env python3
"""Build a private PDF study guide for defending the repo in a professor review."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
TMP_DIR = REPO_ROOT / "tmp" / "pdfs"
OUTPUT_DEFAULT = REPO_ROOT / "output" / "pdf" / "deep_learning_osc_repo_defense_guide.pdf"
MANIFEST_DEFAULT = TMP_DIR / "repo_defense_manifest.json"

QUESTION_ORDER = [
    "question_3_faster_rcnn",
    "question_4_yolo11_yolov12",
    "question_5_semantic_segmentation",
    "question_6_convolution_filters",
    "question_7_transfer_learning",
]

QUESTION_TITLES = {
    "question_3_faster_rcnn": "Q3 Faster R-CNN Object Detection",
    "question_4_yolo11_yolov12": "Q4 YOLO11 and YOLOv12 Object Detection",
    "question_5_semantic_segmentation": "Q5 YOLO11 Semantic Segmentation on iSAID",
    "question_6_convolution_filters": "Q6 Directional Convolution Filters",
    "question_7_transfer_learning": "Q7 TLlib VOC-to-Clipart Transfer Learning",
}

QUESTION_SHORT = {
    "question_3_faster_rcnn": "Q3",
    "question_4_yolo11_yolov12": "Q4",
    "question_5_semantic_segmentation": "Q5",
    "question_6_convolution_filters": "Q6",
    "question_7_transfer_learning": "Q7",
}

FILE_CLASSIFICATION = {
    "question_3_faster_rcnn/demo_fasterrcnn.py": "Repo-owned wrapper/orchestration with embedded compatibility shims.",
    "question_3_faster_rcnn/setup_fasterrcnn_osc.sh": "Repo-owned setup/orchestration shell script.",
    "question_3_faster_rcnn/download_models_fasterrcnn.sh": "Repo-owned checkpoint download helper.",
    "question_4_yolo11_yolov12/demo_yolo11.py": "Repo-owned wrapper/orchestration for Ultralytics YOLO11.",
    "question_4_yolo11_yolov12/demo_yolov12.py": "Repo-owned wrapper/orchestration for the YOLOv12 clone.",
    "question_4_yolo11_yolov12/setup_yolo11_osc.sh": "Repo-owned Ultralytics environment setup script.",
    "question_4_yolo11_yolov12/setup_yolov12_osc.sh": "Repo-owned YOLOv12 environment setup script with pinned dependencies.",
    "question_5_semantic_segmentation/q5_seg_common.py": "Repo-owned shared implementation backbone for Q5 bootstrap, train, and demo flows.",
    "question_5_semantic_segmentation/bootstrap_isaid_seg.py": "Repo-owned dataset bootstrap and normalization wrapper.",
    "question_5_semantic_segmentation/train_isaid_seg.py": "Repo-owned training wrapper around Ultralytics segmentation.",
    "question_5_semantic_segmentation/demo_yolo_segmentation.py": "Repo-owned inference/post-processing/reporting wrapper.",
    "question_5_semantic_segmentation/setup_yolo11_osc.sh": "Repo-owned environment bootstrap and optional dataset auto-bootstrap shell script.",
    "question_6_convolution_filters/convolution_filter_library.py": "Repo-owned implementation of the directional kernels and convolution math.",
    "question_6_convolution_filters/demo_convolution_filters.py": "Repo-owned synthetic demonstration and JSON reporting script.",
    "question_6_convolution_filters/demo_convolution_filters_image.py": "Repo-owned real-image preprocessing, filtering, visualization, and reporting script.",
    "question_7_transfer_learning/demo_tllib_object_detection.py": "Repo-owned high-level TLlib pipeline wrapper and reporting driver.",
    "question_7_transfer_learning/setup_tllib_osc.sh": "Repo-owned setup and compatibility-patch shell script.",
    "question_7_transfer_learning/run_tllib_osc.sh": "Repo-owned execution wrapper that repairs, doctors, and runs the Q7 pipeline.",
    "run_assignment_osc.sh": "Repo-owned repo-level OSC orchestrator for Q3 through Q7.",
    "osc_gpu_batch.sh": "Repo-owned Slurm batch launcher.",
    "osc_gpu_interactive.sh": "Repo-owned Slurm interactive launcher.",
    "osc_gpu_preflight.sh": "Repo-owned shared OSC GPU validation helper.",
}

FILE_PURPOSE = {
    "question_3_faster_rcnn/demo_fasterrcnn.py": "Keeps one stable CLI while switching between CUDA-only PyTorch FasterRCNN and the TF2 fallback, normalizing paths, downloads, batch handling, and output paths.",
    "question_3_faster_rcnn/setup_fasterrcnn_osc.sh": "Prepares the local FasterRCNN clone and one OSC-safe environment that can support both the PyTorch and TF2 code paths.",
    "question_3_faster_rcnn/download_models_fasterrcnn.sh": "Downloads the model weights that the wrapper expects to find under the FasterRCNN clone.",
    "question_4_yolo11_yolov12/demo_yolo11.py": "Resolves paths, validates the runtime, runs Ultralytics YOLO11 through an inline script, then copies curated artifacts out of the upstream run folder.",
    "question_4_yolo11_yolov12/demo_yolov12.py": "Mirrors the YOLO11 wrapper shape but points at the YOLOv12 clone and its dedicated runtime.",
    "question_4_yolo11_yolov12/setup_yolo11_osc.sh": "Clones/reuses Ultralytics, rebuilds the environment, and installs the project in editable mode.",
    "question_4_yolo11_yolov12/setup_yolov12_osc.sh": "Clones/reuses YOLOv12, rebuilds its environment, and pins a narrow dependency set for the supported OSC baseline.",
    "question_5_semantic_segmentation/q5_seg_common.py": "Centralizes path resolution, environment enforcement, data download utilities, iSAID normalization helpers, label-tree replacement, and YAML generation for Q5.",
    "question_5_semantic_segmentation/bootstrap_isaid_seg.py": "Turns raw iSAID and DOTA inputs into a normalized raw layout, YOLO segmentation labels, and a reusable dataset YAML and pretrained checkpoint.",
    "question_5_semantic_segmentation/train_isaid_seg.py": "Launches training with OSC-specific safeguards, resume handling, and stable output aliasing for the fine-tuned checkpoint.",
    "question_5_semantic_segmentation/demo_yolo_segmentation.py": "Runs segmentation inference and converts instance predictions into fused semantic outputs, per-image summaries, and a batch index CSV.",
    "question_5_semantic_segmentation/setup_yolo11_osc.sh": "Ensures the Ultralytics environment is valid and optionally bootstraps the dataset and pretrained checkpoint immediately.",
    "question_6_convolution_filters/convolution_filter_library.py": "Implements the actual kernel derivation, convolution, normalization, and response statistics without relying on an external deep-learning framework.",
    "question_6_convolution_filters/demo_convolution_filters.py": "Builds a synthetic image whose orientations make the kernel behavior easy to justify and compare.",
    "question_6_convolution_filters/demo_convolution_filters_image.py": "Runs the same filter bank on real images and writes human-viewable artifacts that expose each stage of the image pipeline.",
    "question_7_transfer_learning/demo_tllib_object_detection.py": "Normalizes configuration, validates the environment, prepares datasets, builds TLlib commands, controls stage-skipping, and writes portable summaries.",
    "question_7_transfer_learning/setup_tllib_osc.sh": "Builds the TLlib environment and applies local source repairs required by the OSC-tested dependency stack.",
    "question_7_transfer_learning/run_tllib_osc.sh": "Provides the practical end-to-end entrypoint that repairs the environment, runs doctor, resolves the device, and launches the selected full pipeline.",
    "run_assignment_osc.sh": "Chains the per-question setups and demos into one resumable OSC job plan.",
    "osc_gpu_batch.sh": "Turns one repo command into a readable batch script, loads CUDA, and sources the shared GPU preflight helpers.",
    "osc_gpu_interactive.sh": "Requests an interactive allocation and drops the user into a shell that already passed the same preflight rules.",
    "osc_gpu_preflight.sh": "Encapsulates the repository's definition of a valid GPU allocation instead of letting each question folder guess.",
}

UPSTREAM_HANDOFF = {
    "question_3_faster_rcnn/demo_fasterrcnn.py": [
        "Hands off to `external/FasterRCNN/pytorch/*` through the embedded PyTorch script and `external/FasterRCNN/tf2/*` through the embedded TF2 script.",
        "Imports in the embedded script point directly to upstream FasterRCNN modules such as `pytorch.FasterRCNN.*`, `tf2.FasterRCNN.*`, and their visualization/model helpers.",
    ],
    "question_3_faster_rcnn/setup_fasterrcnn_osc.sh": [
        "Clones `https://github.com/trzy/FasterRCNN.git` into `external/FasterRCNN` and installs requirements from `pytorch/requirements.txt` and `tf2/requirements.txt` inside that clone.",
    ],
    "question_3_faster_rcnn/download_models_fasterrcnn.sh": [
        "Downloads checkpoints expected by the upstream FasterRCNN code paths and stores them under the upstream clone root.",
    ],
    "question_4_yolo11_yolov12/demo_yolo11.py": [
        "The inline runtime script imports `ultralytics.YOLO` from `external/ultralytics` and delegates prediction to `model.predict(...)`.",
    ],
    "question_4_yolo11_yolov12/demo_yolov12.py": [
        "The inline runtime script imports `ultralytics.YOLO` from `external/yolov12` and delegates prediction to `model.predict(...)`.",
    ],
    "question_4_yolo11_yolov12/setup_yolo11_osc.sh": [
        "Clones the official Ultralytics repo and installs it editable so imports resolve to the local source tree.",
    ],
    "question_4_yolo11_yolov12/setup_yolov12_osc.sh": [
        "Clones the YOLOv12 repo and installs its local source tree after pinning a repo-owned dependency set.",
    ],
    "question_5_semantic_segmentation/q5_seg_common.py": [
        "Provides the Q5-owned glue code that sits in front of the upstream `external/ultralytics` package.",
        "Uses network/download helpers and then writes artifacts into the layout expected by Ultralytics segmentation training.",
    ],
    "question_5_semantic_segmentation/bootstrap_isaid_seg.py": [
        "Calls `ultralytics.data.converter.convert_coco` inside the local Ultralytics environment to transform normalized COCO annotations into YOLO segmentation labels.",
    ],
    "question_5_semantic_segmentation/train_isaid_seg.py": [
        "Instantiates `ultralytics.YOLO` and subclasses `ultralytics.models.yolo.segment.train.SegmentationTrainer` so the actual optimization loop remains upstream.",
    ],
    "question_5_semantic_segmentation/demo_yolo_segmentation.py": [
        "Instantiates `ultralytics.YOLO` for inference; the repo-owned part starts after the raw result object comes back and turns it into semantic artifacts.",
    ],
    "question_5_semantic_segmentation/setup_yolo11_osc.sh": [
        "Clones/reuses `external/ultralytics` and installs that local source tree; the optional auto-bootstrap then invokes the repo-owned dataset wrapper.",
    ],
    "question_6_convolution_filters/convolution_filter_library.py": [
        "No upstream model backbone is used here; this folder is the clearest example of code that is original to this repository.",
    ],
    "question_6_convolution_filters/demo_convolution_filters.py": [
        "No external backbone beyond the local `convolution_filter_library.py` module.",
    ],
    "question_6_convolution_filters/demo_convolution_filters_image.py": [
        "Uses Pillow for image IO, but the convolution and reporting logic stays in repo-owned code.",
    ],
    "question_7_transfer_learning/demo_tllib_object_detection.py": [
        "Builds commands for `source_only.py`, `d_adapt.py`, and `visualize.py` under `external/Transfer-Learning-Library/examples/domain_adaptation/object_detection`.",
        "The actual detector training/adaptation/visualization is therefore TLlib/Detectron2 code; the repo-owned layer is the controller around it.",
    ],
    "question_7_transfer_learning/setup_tllib_osc.sh": [
        "Clones `thuml/Transfer-Learning-Library` and rewrites specific files under `tllib/...` to keep that checkout compatible with the validated OSC stack.",
    ],
    "question_7_transfer_learning/run_tllib_osc.sh": [
        "Delegates the heavy lifting to `demo_tllib_object_detection.py`, which in turn delegates stage execution to TLlib's upstream scripts.",
    ],
}

INLINE_SCRIPT_NOTES = {
    "question_3_faster_rcnn/demo_fasterrcnn.py::INLINE_PYTORCH_SCRIPT": [
        "Loads the FasterRCNN PyTorch model once, restores Pillow `getsize()` compatibility, iterates over a JSON job list, and calls upstream visualization on each result.",
        "This is important because the outer wrapper never imports upstream PyTorch code directly; it executes this script inside the target runtime with `python -c`.",
    ],
    "question_3_faster_rcnn/demo_fasterrcnn.py::INLINE_TF2_SCRIPT": [
        "Builds the TF2 model with a concrete input shape before loading H5 weights, patches the text renderer for newer Pillow behavior, and then runs inference plus drawing.",
        "The warm-up forward pass before `load_weights(...)` is one of the most scrutiny-worthy lines because it exists purely to satisfy upstream variable naming/layout constraints.",
    ],
    "question_4_yolo11_yolov12/demo_yolo11.py::INLINE_SCRIPT": [
        "This embedded script is the minimal upstream handoff: parse args, instantiate `YOLO`, build `predict_kwargs`, optionally inject `device`, and call `model.predict(...)` once.",
    ],
    "question_4_yolo11_yolov12/demo_yolov12.py::INLINE_SCRIPT": [
        "Same control shape as the YOLO11 inline script, but executed inside the YOLOv12 environment and pointed at its checkpoint names and clone path.",
    ],
}

QUESTION_NOTES = {
    "question_3_faster_rcnn": {
        "what_it_is": [
            "A repo-owned CLI wrapper around the `trzy/FasterRCNN` repository that hides two very different execution paths behind one stable interface: CUDA-only PyTorch and a TF2 fallback.",
            "Its main intellectual content is runtime selection, path normalization, remote-image caching, batch handling, output planning, and compatibility shims that avoid patching vendored source files.",
        ],
        "why_exists": [
            "To make FasterRCNN runnable on OSC with a consistent Python baseline and a headless-friendly `to-file` mode.",
            "To preserve a reviewable artifact layout under `question_3_faster_rcnn/outputs/` instead of leaving results only inside the upstream repo.",
        ],
        "architecture": [
            "CLI args are normalized first: repo path, target interpreter, framework choice, checkpoint path, input selection, and output destinations.",
            "Input handling branches three ways: a URL is downloaded into `inputs/downloaded/`, a single local image becomes one job, and a directory becomes a recursively discovered job list.",
            "Framework resolution probes the target interpreter, not the host shell: PyTorch requires both importability and CUDA visibility; TF2 requires a clean NumPy + Matplotlib + TensorFlow import set.",
            "PyTorch batch mode pushes one JSON job list into `INLINE_PYTORCH_SCRIPT`, which loads the model once and loops over every image; TF2 mode launches one subprocess per image through `INLINE_TF2_SCRIPT`.",
            "The final artifact is always an annotated image in the managed output tree or a viewer window in `viewer` mode.",
        ],
        "deep_focus": [
            "Explain `resolve_framework(...)` carefully because it is where the wrapper decides whether the run is valid at all.",
            "Explain `resolve_input_selection(...)` and `resolve_output_paths(...)` because these are the main repo-owned transformations from user input into deterministic jobs.",
            "Explain why the wrapper keeps compatibility fixes inline instead of patching the upstream clone in place.",
        ],
        "brief_focus": [
            "Mention the low-level FasterRCNN detector math only as upstream backbone behavior unless you also reviewed `external/FasterRCNN/*` in depth.",
            "Mention the download shell helper and default URLs briefly; they are operational, not algorithmically central.",
        ],
        "minimal_run": [
            "Setup: `bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh`",
            "Weights: `bash question_3_faster_rcnn/download_models_fasterrcnn.sh`",
            "CPU-capable default: `python3.9 question_3_faster_rcnn/demo_fasterrcnn.py --mode to-file`",
            "Explicit GPU path: `bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- python3.9 question_3_faster_rcnn/demo_fasterrcnn.py --framework pytorch --mode to-file`",
        ],
        "not_found": [
            "A quantitative evaluation report for Q3 is not found in repo; the tracked outputs are annotated images, not benchmark tables.",
            "A formal per-line diff against upstream FasterRCNN is not found in repo; the wrapper shows the handoff points, but not a provenance ledger for every borrowed idea.",
        ],
    },
    "question_4_yolo11_yolov12": {
        "what_it_is": [
            "A paired set of wrappers that make YOLO11 and YOLOv12 look operationally identical from the assignment side even though they come from different external clones.",
            "The repo-owned logic is mostly about standardizing runtime checks, source normalization, output copying, and OSC guardrails across both models.",
        ],
        "why_exists": [
            "To compare two YOLO-family backbones under one shared repo style.",
            "To keep outputs under dedicated per-model folders while still reusing the same wrapper pattern.",
        ],
        "architecture": [
            "Both wrappers resolve the upstream repo and target interpreter, enforce the Python 3.9.18 baseline, and reject GPU requests when CUDA is unavailable.",
            "The source may be a local file, a local directory, or a URL. The wrapper predicts exactly what filenames the upstream save path should emit before it runs the model.",
            "The inline script is intentionally tiny: instantiate `YOLO`, build `predict_kwargs`, and call `predict` inside the target environment.",
            "After the upstream run, the wrapper copies curated artifacts from `runs/detect/<name>/` into `outputs/yolo11/` or `outputs/yolov12/` and surfaces readable status messages.",
        ],
        "deep_focus": [
            "Explain the mirrored design between `demo_yolo11.py` and `demo_yolov12.py` because that is clear evidence of repo-authored interface shaping rather than copied model code.",
            "Explain `expected_output_names(...)` plus `copy_saved_images(...)`, since those functions are what make the wrapper deterministic and reviewable.",
            "Explain why YOLOv12 setup pins a narrower dependency set than the upstream repo: the repo is asserting a reproducible OSC path.",
        ],
        "brief_focus": [
            "Mention the actual YOLO backbone implementation only as upstream logic unless you inspected the external clones deeply.",
            "Mention the ANSI log cleaning and warning rewriting briefly; it matters operationally but is not the folder's main idea.",
        ],
        "minimal_run": [
            "YOLO11 setup: `bash question_4_yolo11_yolov12/setup_yolo11_osc.sh`",
            "YOLOv12 setup: `bash question_4_yolo11_yolov12/setup_yolov12_osc.sh`",
            "YOLO11 CPU run: `python3.9 question_4_yolo11_yolov12/demo_yolo11.py --device cpu`",
            "YOLOv12 CPU run: `python3.9 question_4_yolo11_yolov12/demo_yolov12.py --device cpu`",
        ],
        "not_found": [
            "A benchmark/evaluation summary comparing YOLO11 and YOLOv12 accuracy is not found in repo; tracked outputs are annotated images, not metrics tables.",
            "A design note explaining why `yolo11n.pt` and `yolov12n.pt` were chosen as defaults is not found in repo.",
        ],
    },
    "question_5_semantic_segmentation": {
        "what_it_is": [
            "The most complete data-engineering pipeline in the repo: dataset discovery/download, raw-layout normalization, annotation rewriting, label conversion, training, inference, semantic fusion, and summary export.",
            "The architectural spine is `q5_seg_common.py`, which means the deepest oral defense should treat that file as the shared substrate behind every other Q5 entrypoint.",
        ],
        "why_exists": [
            "To adapt the general-purpose Ultralytics segmentation workflow to one assignment-specific dataset layout and artifact contract.",
            "To keep the iSAID-to-YOLO conversion logic explicit and reviewable instead of burying it in ad hoc shell commands.",
        ],
        "architecture": [
            "Setup ensures the Ultralytics runtime is importable and optionally auto-bootstraps the dataset and pretrained checkpoint.",
            "Bootstrap resolves repo-local paths, optionally downloads DOTA plus iSAID Google Drive folders, normalizes annotation JSONs to the repo's canonical 15-class scheme, converts COCO to YOLO segmentation labels, and writes `isaid_seg.yaml`.",
            "Training re-execs into the repo environment, verifies CUDA when requested, subclasses the Ultralytics segmentation trainer for OSC-safe validation behavior, and copies the winning checkpoint to a stable alias path under `models/isaid_seg/best.pt`.",
            "Inference loads one checkpoint, resolves any class filter, predicts one image at a time, converts instance masks into a fused semantic class map, writes overlay/class-id/mask artifacts, and serializes JSON/CSV summaries.",
            "The most important repo-owned data manipulations happen before and after the Ultralytics call: JSON normalization before training and semantic fusion/reporting after inference.",
        ],
        "deep_focus": [
            "Explain `normalize_isaid_annotation_json(...)`, `prepare_isaid_raw_layout(...)`, and `write_dataset_yaml(...)` carefully; they are central repo-specific transformations, not generic upstream behavior.",
            "Explain the nested `OSCSafeSegmentationTrainer` class because it shows the repo making a concrete engineering adaptation for OSC memory behavior.",
            "Explain `extract_instances(...)`, `build_semantic_map(...)`, and `render_outputs(...)` because that is where the repo turns instance-style model outputs into the semantic artifacts expected by the assignment.",
        ],
        "brief_focus": [
            "Mention raw Ultralytics optimizer internals briefly unless you reviewed the external code in detail.",
            "Mention packaging-version checks in setup briefly; they are important for reliability but not the conceptual center of Q5.",
        ],
        "minimal_run": [
            "Setup + optional auto-bootstrap: `bash question_5_semantic_segmentation/setup_yolo11_osc.sh`",
            "Manual bootstrap: `external/ultralytics/.venv/bin/python question_5_semantic_segmentation/bootstrap_isaid_seg.py --repo-dir external/ultralytics --download-dataset`",
            "Training: `bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 04:00:00 -- external/ultralytics/.venv/bin/python question_5_semantic_segmentation/train_isaid_seg.py --repo-dir external/ultralytics --device 0 --epochs 20 --imgsz 768 --batch 1 --workers 0 --close-mosaic 20 --exist-ok`",
            "Inference: `external/ultralytics/.venv/bin/python question_5_semantic_segmentation/demo_yolo_segmentation.py --repo-dir external/ultralytics --device cpu`",
        ],
        "not_found": [
            "A tracked benchmark of segmentation quality on iSAID is not found in repo.",
            "A formal explanation for why this exact 15-class palette was chosen is not found in repo beyond the fixed mapping encoded in `q5_seg_common.py`.",
        ],
    },
    "question_6_convolution_filters": {
        "what_it_is": [
            "The most purely original algorithmic folder in the repository: six directional kernels are derived in local code, then applied to both synthetic and real-image inputs.",
            "There is no external deep-learning backbone here; the folder's value is that every step is inspectable, mathematically small, and easy to defend line by line.",
        ],
        "why_exists": [
            "To demonstrate understanding of orientation-sensitive convolution without hiding behind a pretrained model.",
            "To produce both structured numeric output and visual artifacts that make the filter responses defendable.",
        ],
        "architecture": [
            "The library file defines the direction vectors, kernel derivation, same-size zero-padded convolution, absolute-response normalization, and summary statistics.",
            "The synthetic demo builds a test image with horizontal, vertical, and diagonal strokes, runs all kernels, prints the results, and writes a JSON summary.",
            "The real-image demo converts RGB to grayscale, stretches contrast, thresholds to black/white, applies the same filters, normalizes each response to uint8, writes individual maps, and assembles a 2x4 contact sheet.",
        ],
        "deep_focus": [
            "Explain `derive_directional_kernel(...)` and `convolve2d(...)` in detail; they are the core logic most worth defending line by line.",
            "Explain why the preprocessing pipeline is `RGB -> grayscale -> autocontrast -> threshold at 128` before convolution.",
            "Explain how `normalize_matrix_abs_to_uint8(...)` and `matrix_stats(...)` make the feature maps both viewable and comparable.",
        ],
        "brief_focus": [
            "Mention Pillow-only IO helpers briefly; they support the pipeline but are not the algorithmic heart.",
            "Mention CLI argument parsing briefly unless asked about interface design.",
        ],
        "minimal_run": [
            "Synthetic demo: `python3.9 question_6_convolution_filters/demo_convolution_filters.py --size 11`",
            "Image demo: `python3.9 question_6_convolution_filters/demo_convolution_filters_image.py --source question_6_convolution_filters/inputs --output-dir question_6_convolution_filters/outputs`",
        ],
        "not_found": [
            "A citation naming the exact external source for the directional kernel design is not found in repo; the implementation is present, but an academic attribution note is not.",
            "A quantitative comparison against Sobel, Prewitt, or learned filters is not found in repo.",
        ],
    },
    "question_7_transfer_learning": {
        "what_it_is": [
            "A repo-owned controller for TLlib's VOC-to-Clipart domain adaptation example that adds environment doctoring, dataset prep, smoke subsets, stage skipping, profile defaults, compatibility repairs, and summary export.",
            "This folder is not inventing the transfer-learning algorithm; it is engineering a reproducible and inspectable path around TLlib and Detectron2.",
        ],
        "why_exists": [
            "To make a fragile multi-stage domain-adaptation workflow runnable on OSC without manual patching and hand-managed output bookkeeping.",
            "To expose the pipeline as understandable stages instead of one opaque external command.",
        ],
        "architecture": [
            "Setup builds or repairs the TLlib clone and rewrites a handful of upstream files for compatibility with newer torchvision, optional Detectron2 installs, and dataset metadata issues.",
            "The driver script resolves repo paths, profile defaults, interpreter, config files, output directories, environment summary, and device selection before any stage runs.",
            "Dataset preparation ensures full VOC2007/VOC2012/Clipart layouts exist and optionally derives reduced smoke subsets without mutating the full datasets.",
            "Stage execution is explicit: source-only training produces the baseline checkpoint, D-adapt runs sequential phases that consume the previous checkpoint, visualization renders saved predictions, and report/full-pipeline summarize outputs into JSON and Markdown.",
            "The wrapper's value is the controller logic: skip if outputs exist, remove only stage-specific artifacts on `--force`, and keep summaries portable with repo-relative paths.",
        ],
        "deep_focus": [
            "Explain `apply_profile_defaults(...)`, `resolve_dataset_paths(...)`, `build_source_only_command(...)`, and `build_dadapt_command(...)` because they translate the repo's intent into the exact upstream TLlib invocation.",
            "Explain `create_smoke_subset(...)` as a deterministic data-reduction strategy that keeps the dataset contract intact.",
            "Explain `maybe_run_stage(...)`, `summarize_stage(...)`, and `write_summary(...)` because they are where the wrapper becomes resumable and reviewable instead of merely executable.",
            "Explain the compatibility patcher in `setup_tllib_osc.sh`; it is one of the clearest examples of repo-authored engineering around an upstream backbone.",
        ],
        "brief_focus": [
            "Mention Detectron2 and TLlib model internals briefly unless you have separately studied the upstream implementation in `external/Transfer-Learning-Library`.",
            "Mention command-printing and dry-run helpers briefly; they are useful but not the conceptual center.",
        ],
        "minimal_run": [
            "Setup: `bash question_7_transfer_learning/setup_tllib_osc.sh`",
            "Doctor: `python3.9 question_7_transfer_learning/demo_tllib_object_detection.py --mode doctor`",
            "Smoke pipeline: `ALLOW_CPU=1 PROFILE=smoke bash question_7_transfer_learning/run_tllib_osc.sh`",
            "Benchmark pipeline: `bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 04:00:00 -- env PROFILE=benchmark bash question_7_transfer_learning/run_tllib_osc.sh`",
        ],
        "not_found": [
            "A target AP threshold that defines success for Q7 is not found in repo.",
            "A tracked provenance table tying each compatibility patch to a specific upstream TLlib commit or issue is not found in repo, even though the patched files and repair logic are present locally.",
        ],
    },
}

REPO_FRONT_MATTER = {
    "summary": [
        "This repository is primarily a wrapper/orchestration repo, not a repo that reimplements every deep-learning backbone from scratch.",
        "The clearest division is at `external/`: FasterRCNN, Ultralytics, YOLOv12, and TLlib live there; the tracked files in the repo root and `question_*` folders normalize environment setup, path handling, GPU checks, resumability, and artifact layout.",
        "The code gets progressively more wrapper-heavy in Q3, Q4, Q5, and Q7, while Q6 is the most self-contained original implementation.",
    ],
    "shared_patterns": [
        "Python 3.9.18 is enforced repeatedly across question folders and root orchestrators, which means interpreter selection is part of the repo's design, not an incidental detail.",
        "OSC GPU access is treated as an infrastructure concern handled by Slurm wrappers and preflight helpers rather than by the per-question Python scripts.",
        "Most question folders separate three concerns: setup, execution, and artifact management. That pattern is explicit in filenames and comments.",
        "The repo often chooses inline compatibility shims or wrapper subclasses instead of editing upstream projects directly. Q3 and Q5 are the clearest examples.",
    ],
    "interview_risks": [
        "Be precise about what is original repo logic versus upstream backbone code. Over-claiming authorship of TLlib, Ultralytics, YOLOv12, or FasterRCNN internals would be inaccurate.",
        "Be ready to explain why so much code is about validation, path handling, and environment repair: on OSC, reliability is part of the assignment's engineering content.",
        "For the wrapper-heavy folders, defend the repo by walking the data and control flow around the backbone, not by pretending the backbone implementation itself was written here.",
    ],
    "not_found": [
        "A repo-wide architecture diagram is not found in repo.",
        "Question folders for Q1 and Q2 are not found in repo; the study guide therefore covers the actual local scope: Q3 through Q7.",
    ],
}

FUNCTION_NOTES = {
    "question_3_faster_rcnn/demo_fasterrcnn.py::resolve_framework": "This is the real gatekeeper for Q3: it proves whether the prepared environment can legally run PyTorch, must fall back to TF2, or should fail before any upstream code launches.",
    "question_3_faster_rcnn/demo_fasterrcnn.py::resolve_input_selection": "Converts the raw `--image` argument into a normalized job list so the rest of the wrapper can treat single-file, directory, and URL flows uniformly.",
    "question_3_faster_rcnn/demo_fasterrcnn.py::resolve_output_paths": "Maps the normalized input selection into deterministic destination paths and enforces viewer-vs-file behavior.",
    "question_3_faster_rcnn/demo_fasterrcnn.py::build_pytorch_batch_command": "Packages the entire batch into one `python -c` call so the upstream PyTorch model loads once and reuses the same process across all images.",
    "question_3_faster_rcnn/demo_fasterrcnn.py::main": "Binds the whole wrapper together: validate the clone, select framework, resolve weights/inputs/outputs, print the run summary, and then dispatch to the chosen upstream path.",
    "question_4_yolo11_yolov12/demo_yolo11.py::expected_output_names": "Predicts the filenames YOLO should emit before the run, which is what lets the wrapper detect missing artifacts cleanly afterward.",
    "question_4_yolo11_yolov12/demo_yolo11.py::copy_saved_images": "Moves the raw upstream save-dir outputs into the curated assignment output directory while validating that every expected image actually exists.",
    "question_4_yolo11_yolov12/demo_yolo11.py::main": "Shows the full repo-owned contract for YOLO11: repo validation, interpreter checks, source normalization, command construction, upstream run, and curated artifact copy-out.",
    "question_4_yolo11_yolov12/demo_yolov12.py::resolve_python": "Unlike YOLO11, YOLOv12 refuses to proceed if its prepared `.venv` is missing; this is a deliberate stricter contract because the runtime is more tightly pinned.",
    "question_4_yolo11_yolov12/demo_yolov12.py::main": "Mirrors the YOLO11 wrapper shape while surfacing stronger repo-dir error messages that point the user back to the correct setup command.",
    "question_5_semantic_segmentation/q5_seg_common.py::normalize_isaid_annotation_json": "This is one of the most important original Q5 transforms: it rewrites category ids/names into the repo's canonical class space and fixes image filenames/size metadata when needed.",
    "question_5_semantic_segmentation/q5_seg_common.py::prepare_isaid_raw_layout": "Builds the raw dataset contract used by the rest of Q5 by linking/copying DOTA image trees and writing normalized annotation files into one stable layout.",
    "question_5_semantic_segmentation/q5_seg_common.py::load_and_validate_category_maps": "Prevents silent train/val drift by requiring both splits to expose the same contiguous class map before YAML generation or training.",
    "question_5_semantic_segmentation/q5_seg_common.py::write_dataset_yaml": "Writes the exact Ultralytics dataset descriptor consumed by training, so this is the final repo-owned handoff from data engineering into the model stack.",
    "question_5_semantic_segmentation/bootstrap_isaid_seg.py::main": "Owns the full Q5 data bootstrap narrative: path resolution, optional downloads, normalization, label conversion, YAML writing, and pretrained checkpoint fetch.",
    "question_5_semantic_segmentation/train_isaid_seg.py::resolve_training_model": "Fails early when the requested checkpoint is missing, which keeps training errors attributable to setup rather than buried deep in Ultralytics.",
    "question_5_semantic_segmentation/train_isaid_seg.py::main.OSCSafeSegmentationTrainer": "This nested subclass is the core OSC-specific training adaptation: it reduces validation loader pressure and suppresses redundant final validation behavior.",
    "question_5_semantic_segmentation/train_isaid_seg.py::main": "Stages the training run, guards resume behavior, loads the model, injects the custom trainer, and copies the chosen checkpoint to a stable alias path.",
    "question_5_semantic_segmentation/demo_yolo_segmentation.py::extract_instances": "Turns the raw Ultralytics result object into a repo-owned detection list with class ids, confidences, boxes, pixel counts, and boolean masks.",
    "question_5_semantic_segmentation/demo_yolo_segmentation.py::build_semantic_map": "Resolves overlapping instances by per-pixel confidence competition, which is the key step that makes the final output semantic rather than merely instance-level.",
    "question_5_semantic_segmentation/demo_yolo_segmentation.py::render_outputs": "Converts the fused class map into the three visible artifacts: overlay, raw class-id PNG, and colorized semantic mask.",
    "question_5_semantic_segmentation/demo_yolo_segmentation.py::success_summary": "Serializes the semantic result into a defendable JSON record with classes, pixel counts, instance counts, paths, and flattened detection metadata.",
    "question_5_semantic_segmentation/demo_yolo_segmentation.py::run_directory_mode": "Shows how batch execution is made reviewable: each image gets its own output folder and summary, and the whole run is flattened into one index CSV.",
    "question_6_convolution_filters/convolution_filter_library.py::derive_directional_kernel": "The cleanest algorithmic core in the repo: project each kernel cell onto the target direction and reduce that projection to -1, 0, or 1.",
    "question_6_convolution_filters/convolution_filter_library.py::convolve2d": "Implements same-size zero-padded convolution explicitly, making the spatial math defendable without relying on any framework primitive.",
    "question_6_convolution_filters/demo_convolution_filters.py::build_synthetic_image": "Constructs a test pattern that guarantees all six orientations are represented, which is why the response summary is easy to justify.",
    "question_6_convolution_filters/demo_convolution_filters_image.py::process_image": "Owns the end-to-end real-image pipeline: load, preprocess, convolve, normalize, save each feature map, build the contact sheet, and serialize summary JSON.",
    "question_7_transfer_learning/demo_tllib_object_detection.py::apply_profile_defaults": "Makes `smoke` and `benchmark` real operational modes instead of mere labels by filling in missing runtime knobs from repo-owned defaults.",
    "question_7_transfer_learning/demo_tllib_object_detection.py::create_smoke_subset": "Creates deterministic reduced datasets without breaking the VOC-style directory contract, which is central to explaining why smoke mode still exercises the real pipeline shape.",
    "question_7_transfer_learning/demo_tllib_object_detection.py::build_source_only_command": "Translates the repo's stage concept into the exact TLlib CLI for baseline training and evaluation.",
    "question_7_transfer_learning/demo_tllib_object_detection.py::build_dadapt_command": "Builds one sequential adaptation phase, injecting both TLlib adaptor knobs and Detectron2 config overrides.",
    "question_7_transfer_learning/demo_tllib_object_detection.py::maybe_run_stage": "This is the resumability contract: skip if the expected output exists, or surgically clean only the current stage when `--force` is set.",
    "question_7_transfer_learning/demo_tllib_object_detection.py::summarize_stage": "Extracts the minimum facts that make the pipeline defendable after the fact: checkpoints, log existence, metrics, adaptor status, and visualization counts.",
    "question_7_transfer_learning/demo_tllib_object_detection.py::write_summary": "Writes both machine-readable and human-readable reports so the pipeline leaves a portable audit trail.",
    "question_7_transfer_learning/demo_tllib_object_detection.py::main": "Owns the full controller flow: resolve paths/profile/device, prepare datasets, validate the runtime, run stages, and materialize the final report.",
}


@dataclass
class DefinitionInfo:
    qualname: str
    name: str
    kind: str
    lineno: int
    end_lineno: int
    header: str
    docstring: str
    depth: int
    category: str
    summary: str
    raises: list[str] = field(default_factory=list)
    block_walk: list[dict[str, Any]] = field(default_factory=list)


def rich(text: str) -> str:
    """Escape paragraph text and render backticked spans in a monospace font."""
    if not text:
        return ""
    parts = re.split(r"(`[^`]+`)", text)
    rendered = []
    for part in parts:
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            rendered.append(f'<font name="Courier">{escape(part[1:-1])}</font>')
        else:
            rendered.append(escape(part))
    return "".join(rendered).replace("\n", "<br/>")


def code_ref(path: str | Path, start: int, end: int | None = None) -> str:
    label = f"{path}:{start}" if end is None or end == start else f"{path}:{start}-{end}"
    return f"`{label}`"


def git_head(repo_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "not found in repo"


def guess_category(name: str, kind: str, body_text: str) -> str:
    lowered = name.lower()
    if kind == "class":
        return "class/type definition"
    if lowered == "main":
        return "entrypoint orchestration"
    if lowered.startswith("build_") and "command" in lowered:
        return "upstream command construction"
    if lowered.startswith("build_"):
        return "configuration or artifact construction"
    if lowered.startswith("resolve_"):
        return "path/config resolution"
    if lowered.startswith("ensure_"):
        return "validation or readiness gate"
    if lowered.startswith("parse_"):
        return "argument or text parsing"
    if lowered.startswith("write_"):
        return "serialization/output writing"
    if lowered.startswith("load_"):
        return "loading/ingest"
    if lowered.startswith("download_") or lowered.startswith("fetch_"):
        return "download or remote fetch"
    if lowered.startswith("copy_"):
        return "artifact copying or dataset materialization"
    if lowered.startswith("normalize_"):
        return "data normalization"
    if lowered.startswith("summarize_") or lowered.startswith("format_"):
        return "summary/report formatting"
    if lowered.startswith("run_") or lowered.startswith("maybe_run_"):
        return "stage execution"
    if lowered.startswith("render_") or lowered.startswith("visualize"):
        return "visualization/output rendering"
    if lowered.startswith("extract_"):
        return "structured extraction from upstream outputs"
    if lowered.startswith("collect_"):
        return "environment or artifact collection"
    if lowered.startswith("validate_"):
        return "validation or contract enforcement"
    if "subprocess" in body_text or "check=True" in body_text:
        return "subprocess orchestration"
    return "local helper logic"


def guess_summary(relpath: str, def_name: str, kind: str, docstring: str, category: str) -> str:
    key = f"{relpath}::{def_name}"
    if key in FUNCTION_NOTES:
        return FUNCTION_NOTES[key]
    if docstring:
        text = docstring.strip()
        if not text.endswith("."):
            text += "."
        return text
    if kind == "class":
        return f"Defines the `{def_name}` type used by surrounding logic in this file."
    if category == "path/config resolution":
        return "Normalizes one incoming value into the canonical form used by the rest of the wrapper."
    if category == "validation or readiness gate":
        return "Stops the pipeline early when a required invariant is missing, instead of letting the failure happen deeper downstream."
    if category == "upstream command construction":
        return "Builds the exact argv list that will be handed off to the upstream project or runtime."
    if category == "serialization/output writing":
        return "Turns in-memory results into the stable artifact format this repo expects."
    if category == "stage execution":
        return "Owns one execution step rather than one pure transformation."
    if category == "data normalization":
        return "Rewrites data into the normalized representation expected by later stages."
    return f"Implements `{def_name}` as a {category} helper."


def raise_types(node: ast.AST) -> list[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            exc = child.exc
            if isinstance(exc, ast.Call):
                func = exc.func
                if isinstance(func, ast.Name):
                    found.add(func.id)
                elif isinstance(func, ast.Attribute):
                    found.add(func.attr)
            elif isinstance(exc, ast.Name):
                found.add(exc.id)
    return sorted(found)


def leading_indent(text: str) -> int:
    return len(text) - len(text.lstrip(" "))


def extract_python_blocks(lines: list[str], start: int, end: int, base_indent: int) -> list[dict[str, Any]]:
    """Use local comment groups as a compact line-block walkthrough."""
    blocks: list[dict[str, Any]] = []
    pending_title: str | None = None
    pending_comment_line: int | None = None
    current_start: int | None = None
    current_title: str | None = None

    def finalize(block_end: int) -> None:
        nonlocal current_start, current_title
        if current_start is None or current_title is None:
            return
        if block_end < current_start:
            return
        blocks.append(
            {
                "title": current_title,
                "start": current_start,
                "end": block_end,
            }
        )
        current_start = None
        current_title = None

    i = start + 1
    while i <= end:
        line = lines[i - 1]
        stripped = line.strip()
        indent = leading_indent(line)

        if stripped.startswith("#") and indent > base_indent:
            title_parts = [stripped.lstrip("# ").strip()]
            pending_comment_line = i
            j = i + 1
            while j <= end:
                line2 = lines[j - 1]
                stripped2 = line2.strip()
                if stripped2.startswith("#") and leading_indent(line2) == indent:
                    title_parts.append(stripped2.lstrip("# ").strip())
                    j += 1
                    continue
                break
            pending_title = " ".join(part for part in title_parts if part)
            if current_start is not None:
                finalize(pending_comment_line - 1)
            i = j
            continue

        if pending_title and stripped:
            current_title = pending_title
            current_start = i
            pending_title = None
            pending_comment_line = None

        i += 1

    if current_start is not None:
        finalize(end)

    if not blocks and end - start >= 18:
        return [{"title": "Single contiguous control block.", "start": start + 1, "end": end}]

    return blocks[:8]


class DefinitionCollector(ast.NodeVisitor):
    def __init__(self, lines: list[str], relpath: str):
        self.lines = lines
        self.relpath = relpath
        self.stack: list[str] = []
        self.items: list[DefinitionInfo] = []

    def _record(self, node: ast.AST, name: str, kind: str) -> None:
        header = self.lines[node.lineno - 1].strip()
        docstring = ast.get_docstring(node) or ""
        qualname = ".".join([*self.stack, name]) if self.stack else name
        body_text = "\n".join(self.lines[node.lineno - 1 : node.end_lineno])
        category = guess_category(name, kind, body_text)
        summary = guess_summary(self.relpath, qualname, kind, docstring, category)
        base_indent = leading_indent(self.lines[node.lineno - 1])
        blocks = extract_python_blocks(self.lines, node.lineno, node.end_lineno, base_indent)
        self.items.append(
            DefinitionInfo(
                qualname=qualname,
                name=name,
                kind=kind,
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                header=header,
                docstring=docstring,
                depth=len(self.stack),
                category=category,
                summary=summary,
                raises=raise_types(node),
                block_walk=blocks,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node, node.name, "function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node, node.name, "async function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, node.name, "class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def parse_python_file(path: Path) -> dict[str, Any]:
    relpath = path.relative_to(REPO_ROOT).as_posix()
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    collector = DefinitionCollector(lines, relpath)
    collector.visit(tree)

    imports: list[str] = []
    inline_scripts: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(f"{node.module or ''}::{', '.join(alias.name for alias in node.names)}")
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id.startswith("INLINE")
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                script_name = target.id
                inner_source = node.value.value
                inner_lines = inner_source.splitlines()
                try:
                    inner_tree = ast.parse(inner_source)
                    inner_collector = DefinitionCollector(inner_lines, relpath)
                    inner_collector.visit(inner_tree)
                    inner_defs = []
                    for item in inner_collector.items:
                        adjusted = asdict(item)
                        adjusted["lineno"] = node.lineno + item.lineno - 1
                        adjusted["end_lineno"] = node.lineno + item.end_lineno - 1
                        adjusted["outer_anchor"] = code_ref(relpath, adjusted["lineno"], adjusted["end_lineno"])
                        inner_defs.append(adjusted)
                except SyntaxError:
                    inner_defs = []
                inline_scripts.append(
                    {
                        "name": script_name,
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno,
                        "summary": INLINE_SCRIPT_NOTES.get(f"{relpath}::{script_name}", []),
                        "definitions": inner_defs,
                    }
                )

    return {
        "path": relpath,
        "type": "python",
        "line_count": len(lines),
        "classification": FILE_CLASSIFICATION.get(relpath, "Repo-owned Python file."),
        "purpose": FILE_PURPOSE.get(relpath, "Purpose not explicitly summarized; inspect file-level comments and function inventory."),
        "imports": imports,
        "definitions": [asdict(item) for item in collector.items],
        "inline_scripts": inline_scripts,
    }


def extract_shell_functions(lines: list[str]) -> list[dict[str, Any]]:
    pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(\)\s*\{$")
    defs = []
    matches = [(i + 1, pattern.match(line.strip())) for i, line in enumerate(lines)]
    fn_lines = [(lineno, match.group(1)) for lineno, match in matches if match]
    for index, (lineno, name) in enumerate(fn_lines):
        end = len(lines)
        if index + 1 < len(fn_lines):
            end = fn_lines[index + 1][0] - 1
        body = "\n".join(lines[lineno - 1 : end])
        defs.append(
            {
                "qualname": name,
                "name": name,
                "kind": "shell function",
                "lineno": lineno,
                "end_lineno": end,
                "header": lines[lineno - 1].strip(),
                "docstring": "",
                "depth": 0,
                "category": guess_category(name, "function", body),
                "summary": guess_summary("shell", name, "function", "", guess_category(name, "function", body)),
                "raises": [],
                "block_walk": [],
            }
        )
    return defs


def extract_shell_stages(lines: list[str]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    current_title: str | None = None
    current_start: int | None = None

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("# ").strip()
            if title:
                if current_title is not None and current_start is not None:
                    stages.append({"title": current_title, "start": current_start, "end": lineno - 1})
                current_title = title
                current_start = lineno + 1
    if current_title is not None and current_start is not None:
        stages.append({"title": current_title, "start": current_start, "end": len(lines)})
    return stages[:16]


def parse_shell_file(path: Path) -> dict[str, Any]:
    relpath = path.relative_to(REPO_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    commands = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.strip().startswith("#")
        and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*\(\)\s*\{$", line.strip())
        and line.strip() not in {"}", "then", "fi", "done", "do", "else"}
    ]
    representative = commands[:12]
    return {
        "path": relpath,
        "type": "shell",
        "line_count": len(lines),
        "classification": FILE_CLASSIFICATION.get(relpath, "Repo-owned shell script."),
        "purpose": FILE_PURPOSE.get(relpath, "Purpose not explicitly summarized; inspect comments and command blocks."),
        "definitions": extract_shell_functions(lines),
        "stages": extract_shell_stages(lines),
        "representative_commands": representative,
    }


def parse_markdown_file(path: Path) -> dict[str, Any]:
    relpath = path.relative_to(REPO_ROOT).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    headings = [line.strip("# ").strip() for line in lines if line.startswith("#")]
    return {
        "path": relpath,
        "type": "markdown",
        "line_count": len(lines),
        "classification": "Repo-owned documentation/evidence file.",
        "purpose": "Acts as explicit repo evidence for workflow intent, minimal run steps, and ownership boundaries.",
        "headings": headings[:20],
        "definitions": [],
    }


def parse_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".py":
        return parse_python_file(path)
    if path.suffix == ".sh":
        return parse_shell_file(path)
    return parse_markdown_file(path)


def question_files(question_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in question_dir.iterdir()
            if path.is_file() and (path.suffix in {".py", ".sh"} or path.name == "README.md")
        ],
        key=lambda p: (p.suffix != ".md", p.name),
    )


def sample_files(paths: list[Path], limit: int = 8) -> list[str]:
    items = []
    for path in sorted(paths)[:limit]:
        items.append(path.relative_to(REPO_ROOT).as_posix())
    return items


def build_artifact_evidence(question: str) -> list[str]:
    folder = REPO_ROOT / question
    evidence: list[str] = []
    if question == "question_3_faster_rcnn":
        outputs = list((folder / "outputs").glob("*"))
        evidence.append(f"Tracked annotated outputs present: `{len(outputs)}` files under `question_3_faster_rcnn/outputs/`.")
    elif question == "question_4_yolo11_yolov12":
        y11 = list((folder / "outputs" / "yolo11").glob("*"))
        y12 = list((folder / "outputs" / "yolov12").glob("*"))
        evidence.append(f"Tracked curated outputs present: `{len(y11)}` YOLO11 images and `{len(y12)}` YOLOv12 images.")
    elif question == "question_5_semantic_segmentation":
        index_csv = folder / "outputs" / "satellite_results" / "index.csv"
        if index_csv.exists():
            with index_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            statuses = Counter(row.get("status", "") for row in rows)
            evidence.append(
                f"Existing batch evidence at `question_5_semantic_segmentation/outputs/satellite_results/index.csv`: "
                f"{dict(statuses)}. The tracked CSV currently records error rows rather than silent failure."
            )
            if rows:
                first = rows[0]
                if first.get("error_message"):
                    one_line = " ".join(first["error_message"].split())[:180]
                    evidence.append(f"Representative tracked failure message: `{one_line}...`")
        summary_files = list((folder / "outputs" / "satellite_results").rglob("summary.json"))
        evidence.append(f"Per-image summary artifacts present: `{len(summary_files)}` JSON files under `outputs/satellite_results/`.")
    elif question == "question_6_convolution_filters":
        demo_json = folder / "outputs" / "convolution_filter_demo.json"
        if demo_json.exists():
            payload = json.loads(demo_json.read_text(encoding="utf-8"))
            kernels = ", ".join(payload.get("kernels", {}).keys())
            evidence.append(f"Synthetic demo JSON exists and records kernels: `{kernels}`.")
            stats = payload.get("response_stats", {})
            if stats:
                top = next(iter(stats))
                evidence.append(
                    f"Tracked response stats include fields like `min`, `max`, `mean_abs`, and `sum_abs` for `{top}`."
                )
        output_dirs = [path for path in (folder / "outputs").iterdir() if path.is_dir()] if (folder / "outputs").exists() else []
        evidence.append(f"Per-image output directories present: `{len(output_dirs)}`.")
    elif question == "question_7_transfer_learning":
        summary_md = folder / "outputs" / "voc2clipart_smoke" / "summary.md"
        if summary_md.exists():
            text = summary_md.read_text(encoding="utf-8")
            phase_hits = len(re.findall(r"^## ", text, flags=re.MULTILINE))
            evidence.append(
                f"Tracked smoke summary exists at `question_7_transfer_learning/outputs/voc2clipart_smoke/summary.md` with `{phase_hits}` stage sections."
            )
            if "AP=0.0" in text:
                evidence.append("The tracked smoke summary currently shows zero AP metrics, which is useful evidence that the reporting layer works even when results are weak.")
        vis = list((folder / "visualizations").rglob("*.png")) if (folder / "visualizations").exists() else []
        evidence.append(f"Tracked visualization PNGs present: `{len(vis)}`.")
    return evidence


def external_backbone_map() -> list[str]:
    entries = [
        ("external/FasterRCNN", "trzy/FasterRCNN"),
        ("external/ultralytics", "ultralytics/ultralytics"),
        ("external/yolov12", "sunsmarterjie/yolov12"),
        ("external/Transfer-Learning-Library", "thuml/Transfer-Learning-Library"),
    ]
    bullets = []
    for relpath, label in entries:
        repo_path = REPO_ROOT / relpath
        head = git_head(repo_path) if repo_path.exists() else "not found in repo"
        bullets.append(f"`{relpath}` -> `{label}`; local clone head: `{head}`.")
    return bullets


def build_question_manifest(question: str) -> dict[str, Any]:
    question_dir = REPO_ROOT / question
    files = [parse_file(path) for path in question_files(question_dir)]
    total_lines = sum(file_info["line_count"] for file_info in files)
    definition_total = sum(len(file_info.get("definitions", [])) for file_info in files if file_info["type"] == "python")
    shell_def_total = sum(len(file_info.get("definitions", [])) for file_info in files if file_info["type"] == "shell")
    outputs = list((question_dir / "outputs").rglob("*")) if (question_dir / "outputs").exists() else []
    runs = list((question_dir / "runs").rglob("*")) if (question_dir / "runs").exists() else []
    datasets = list((question_dir / "datasets").rglob("*")) if (question_dir / "datasets").exists() else []
    logs = list((question_dir / "logs").rglob("*")) if (question_dir / "logs").exists() else []
    return {
        "question": question,
        "title": QUESTION_TITLES[question],
        "short": QUESTION_SHORT[question],
        "files": files,
        "stats": {
            "file_count": len(files),
            "total_lines": total_lines,
            "python_definition_count": definition_total,
            "shell_function_count": shell_def_total,
            "output_artifact_count": len(outputs),
            "run_artifact_count": len(runs),
            "dataset_artifact_count": len(datasets),
            "log_artifact_count": len(logs),
        },
        "artifact_samples": {
            "outputs": sample_files(outputs, limit=10),
            "runs": sample_files(runs, limit=8),
            "datasets": sample_files(datasets, limit=8),
            "logs": sample_files(logs, limit=8),
        },
        "artifact_evidence": build_artifact_evidence(question),
    }


def build_manifest() -> dict[str, Any]:
    root_files = [
        parse_file(REPO_ROOT / "README.md"),
        parse_file(REPO_ROOT / "run_assignment_osc.sh"),
        parse_file(REPO_ROOT / "osc_gpu_batch.sh"),
        parse_file(REPO_ROOT / "osc_gpu_interactive.sh"),
        parse_file(REPO_ROOT / "osc_gpu_preflight.sh"),
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "repo_front_matter": REPO_FRONT_MATTER,
        "root_files": root_files,
        "external_backbones": external_backbone_map(),
        "questions": [build_question_manifest(question) for question in QUESTION_ORDER],
    }


def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "GuideTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#12263A"),
            spaceAfter=12,
        ),
        "SubTitle": ParagraphStyle(
            "GuideSubTitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#3D5A6C"),
            spaceAfter=12,
        ),
        "H1": ParagraphStyle(
            "GuideH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=colors.HexColor("#0B3C5D"),
            spaceBefore=8,
            spaceAfter=8,
        ),
        "H2": ParagraphStyle(
            "GuideH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#0F5E7A"),
            spaceBefore=7,
            spaceAfter=5,
        ),
        "H3": ParagraphStyle(
            "GuideH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#1A3C40"),
            spaceBefore=6,
            spaceAfter=3,
        ),
        "Body": ParagraphStyle(
            "GuideBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10.5,
            spaceAfter=3,
        ),
        "Bullet": ParagraphStyle(
            "GuideBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=10.2,
            leftIndent=14,
            firstLineIndent=0,
            spaceAfter=2,
        ),
        "Small": ParagraphStyle(
            "GuideSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.3,
            leading=9,
            textColor=colors.HexColor("#36454F"),
            spaceAfter=2,
        ),
    }
    return styles


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(rich(text), style)


def bullet(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(rich(text), style, bulletText="-")


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6C7A89"))
    canvas.drawString(doc.leftMargin, 0.45 * inch, "Private study guide for local use only.")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def add_spacer(story: list[Any], height: float = 0.08) -> None:
    story.append(Spacer(1, height * inch))


def append_definitions(story: list[Any], styles: dict[str, ParagraphStyle], file_info: dict[str, Any]) -> None:
    definitions = file_info.get("definitions", [])
    if not definitions:
        return
    story.append(para("Function-by-function walkthrough", styles["H3"]))
    for definition in definitions:
        anchor = code_ref(file_info["path"], definition["lineno"], definition["end_lineno"])
        header = (
            f"<b>{escape(definition['qualname'])}</b> "
            f"({escape(definition['kind'])}; {rich(anchor)})"
        )
        story.append(Paragraph(header, styles["Body"]))
        story.append(
            para(
                f"Category: `{definition['category']}`. Summary: {definition['summary']}",
                styles["Small"],
            )
        )
        if definition.get("raises"):
            story.append(
                para(
                    "Failure signals raised directly here: "
                    + ", ".join(f"`{name}`" for name in definition["raises"]),
                    styles["Small"],
                )
            )
        if definition.get("block_walk"):
            blocks = "; ".join(
                f"{rich(code_ref(file_info['path'], block['start'], block['end']))}: {escape(block['title'])}"
                for block in definition["block_walk"]
            )
            story.append(Paragraph(f"Line-block walk: {blocks}", styles["Small"]))
        add_spacer(story, 0.03)


def append_inline_scripts(story: list[Any], styles: dict[str, ParagraphStyle], file_info: dict[str, Any]) -> None:
    inline_scripts = file_info.get("inline_scripts", [])
    if not inline_scripts:
        return
    story.append(para("Embedded runtime scripts executed with `python -c`", styles["H3"]))
    for script in inline_scripts:
        story.append(
            para(
                f"`{script['name']}` lives in {code_ref(file_info['path'], script['lineno'], script['end_lineno'])}.",
                styles["Body"],
            )
        )
        for note in script.get("summary", []):
            story.append(bullet(note, styles["Bullet"]))
        for definition in script.get("definitions", []):
            story.append(
                para(
                    f"Inner definition: `{definition['qualname']}` at {definition['outer_anchor']}. "
                    f"Summary: {definition['summary']}",
                    styles["Small"],
                )
            )
        add_spacer(story, 0.03)


def append_shell_details(story: list[Any], styles: dict[str, ParagraphStyle], file_info: dict[str, Any]) -> None:
    stages = file_info.get("stages", [])
    if stages:
        story.append(para("Top-level shell stages", styles["H3"]))
        for stage in stages:
            story.append(
                para(
                    f"{code_ref(file_info['path'], stage['start'], stage['end'])}: {stage['title']}",
                    styles["Small"],
                )
            )
    functions = file_info.get("definitions", [])
    if functions:
        story.append(para("Shell functions", styles["H3"]))
        for definition in functions:
            story.append(
                para(
                    f"`{definition['qualname']}` at {code_ref(file_info['path'], definition['lineno'], definition['end_lineno'])}. "
                    f"Category: `{definition['category']}`. Summary: {definition['summary']}",
                    styles["Small"],
                )
            )
    commands = file_info.get("representative_commands", [])
    if commands:
        story.append(para("Representative commands and side effects", styles["H3"]))
        for command in commands[:8]:
            story.append(para(f"`{command}`", styles["Small"]))


def append_file_section(story: list[Any], styles: dict[str, ParagraphStyle], file_info: dict[str, Any]) -> None:
    relpath = file_info["path"]
    story.append(para(f"File: `{relpath}`", styles["H2"]))
    story.append(
        para(
            f"Classification: {file_info['classification']} "
            f"Evidence anchor: {code_ref(relpath, 1, file_info['line_count'])}",
            styles["Body"],
        )
    )
    story.append(para(f"Purpose: {file_info['purpose']}", styles["Body"]))

    for note in UPSTREAM_HANDOFF.get(relpath, []):
        story.append(bullet(f"Upstream handoff: {note}", styles["Bullet"]))

    if file_info["type"] == "markdown":
        headings = file_info.get("headings", [])
        if headings:
            story.append(
                para(
                    "Key headings present in the tracked README: "
                    + ", ".join(f"`{heading}`" for heading in headings[:10]),
                    styles["Small"],
                )
            )
        return

    if file_info["type"] == "python":
        imports = file_info.get("imports", [])
        if imports:
            sample = ", ".join(f"`{item}`" for item in imports[:8])
            story.append(para(f"Top import evidence: {sample}", styles["Small"]))
        append_inline_scripts(story, styles, file_info)
        append_definitions(story, styles, file_info)
        return

    append_shell_details(story, styles, file_info)


def append_question_section(story: list[Any], styles: dict[str, ParagraphStyle], question: dict[str, Any]) -> None:
    notes = QUESTION_NOTES[question["question"]]
    story.append(PageBreak())
    story.append(para(question["title"], styles["H1"]))
    story.append(
        para(
            f"Tracked scope: `{question['stats']['file_count']}` top-level files, "
            f"`{question['stats']['total_lines']}` lines, "
            f"`{question['stats']['python_definition_count']}` Python definitions, "
            f"`{question['stats']['shell_function_count']}` shell functions.",
            styles["SubTitle"],
        )
    )

    story.append(para("What it is", styles["H2"]))
    for line in notes["what_it_is"]:
        story.append(bullet(line, styles["Bullet"]))

    story.append(para("Why this folder exists in the repo", styles["H2"]))
    for line in notes["why_exists"]:
        story.append(bullet(line, styles["Bullet"]))

    story.append(para("Architecture and data flow", styles["H2"]))
    for line in notes["architecture"]:
        story.append(bullet(line, styles["Bullet"]))

    story.append(para("Tracked files and responsibilities", styles["H2"]))
    for file_info in question["files"]:
        story.append(
            para(
                f"`{file_info['path']}` -> {file_info['classification']} "
                f"({file_info['line_count']} lines).",
                styles["Small"],
            )
        )

    story.append(para("Data transformations and artifact lifecycle", styles["H2"]))
    stats = question["stats"]
    story.append(
        bullet(
            f"Artifact counts now present on disk: outputs `{stats['output_artifact_count']}`, runs `{stats['run_artifact_count']}`, "
            f"datasets `{stats['dataset_artifact_count']}`, logs `{stats['log_artifact_count']}`.",
            styles["Bullet"],
        )
    )
    for line in question["artifact_evidence"]:
        story.append(bullet(line, styles["Bullet"]))
    samples = question["artifact_samples"]
    for label, values in samples.items():
        if values:
            story.append(
                para(
                    f"Sample tracked {label}: " + ", ".join(f"`{value}`" for value in values[:6]),
                    styles["Small"],
                )
            )

    story.append(para("Original repo logic vs upstream backbone logic", styles["H2"]))
    story.append(
        bullet(
            "Repo-owned logic here is mostly the tracked `.py`/`.sh` wrapper layer plus any artifact shaping, environment enforcement, and summary-writing logic called out below.",
            styles["Bullet"],
        )
    )
    handoff_paths = set()
    for file_info in question["files"]:
        for note in UPSTREAM_HANDOFF.get(file_info["path"], []):
            story.append(bullet(note, styles["Bullet"]))
            handoff_paths.add(note)
    if not handoff_paths:
        story.append(bullet("No upstream backbone handoff beyond local helper modules is evidenced in this folder.", styles["Bullet"]))

    story.append(para("What to explain deeply vs what to mention briefly", styles["H2"]))
    for line in notes["deep_focus"]:
        story.append(bullet(f"Explain deeply: {line}", styles["Bullet"]))
    for line in notes["brief_focus"]:
        story.append(bullet(f"Mention briefly: {line}", styles["Bullet"]))

    story.append(para("Minimal run path", styles["H2"]))
    for line in notes["minimal_run"]:
        story.append(bullet(line, styles["Bullet"]))

    story.append(para("Per-file walkthrough", styles["H2"]))
    for file_info in question["files"]:
        append_file_section(story, styles, file_info)

    story.append(para("Not found in repo", styles["H2"]))
    for line in notes["not_found"]:
        story.append(bullet(line, styles["Bullet"]))


def build_story(manifest: dict[str, Any]) -> list[Any]:
    styles = make_styles()
    story: list[Any] = []

    story.append(para("Deep Repo Defense Guide", styles["Title"]))
    story.append(
        para(
            "Private PDF study guide for defending `Deep-Learning-On-OSC` in a professor interview. "
            "This guide is grounded only in the local repository and local external clones present in the workspace.",
            styles["SubTitle"],
        )
    )
    story.append(
        para(
            f"Generated from `{REPO_ROOT}` at `{manifest['generated_at']}`. "
            "Scope covered here: Q3 through Q7 plus repo-level orchestration and OSC helpers.",
            styles["SubTitle"],
        )
    )

    story.append(para("Repo Orientation", styles["H1"]))
    for line in manifest["repo_front_matter"]["summary"]:
        story.append(bullet(line, styles["Bullet"]))

    story.append(para("Shared patterns to defend across the whole repo", styles["H2"]))
    for line in manifest["repo_front_matter"]["shared_patterns"]:
        story.append(bullet(line, styles["Bullet"]))

    story.append(para("External backbone map", styles["H2"]))
    for line in manifest["external_backbones"]:
        story.append(bullet(line, styles["Bullet"]))

    story.append(para("Root orchestration files worth knowing", styles["H2"]))
    for file_info in manifest["root_files"]:
        story.append(
            para(
                f"`{file_info['path']}` -> {file_info['purpose']} "
                f"({file_info['line_count']} lines).",
                styles["Small"],
            )
        )
        if file_info["type"] == "shell":
            functions = file_info.get("definitions", [])
            if functions:
                story.append(
                    para(
                        "Functions: " + ", ".join(f"`{item['name']}`" for item in functions[:12]),
                        styles["Small"],
                    )
                )

    story.append(para("Likely repo-wide scrutiny points", styles["H2"]))
    for line in manifest["repo_front_matter"]["interview_risks"]:
        story.append(bullet(line, styles["Bullet"]))

    story.append(para("Not found in repo", styles["H2"]))
    for line in manifest["repo_front_matter"]["not_found"]:
        story.append(bullet(line, styles["Bullet"]))

    story.append(para("Question-by-question map", styles["H2"]))
    for question in manifest["questions"]:
        story.append(
            bullet(
                f"{question['short']}: `{question['stats']['file_count']}` files, "
                f"`{question['stats']['total_lines']}` lines, "
                f"`{question['stats']['output_artifact_count']}` output artifacts currently on disk.",
                styles["Bullet"],
            )
        )

    for question in manifest["questions"]:
        append_question_section(story, styles, question)

    story.append(PageBreak())
    story.append(para("Appendix: Question Folder Inventory", styles["H1"]))
    for question in manifest["questions"]:
        story.append(para(question["title"], styles["H2"]))
        for file_info in question["files"]:
            story.append(
                para(
                    f"`{file_info['path']}` | type=`{file_info['type']}` | lines=`{file_info['line_count']}`",
                    styles["Small"],
                )
            )

    return story


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_pdf(output_path: Path, manifest_path: Path) -> None:
    manifest = build_manifest()
    write_manifest(manifest, manifest_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.7 * inch,
        title="Deep Repo Defense Guide",
        author="Codex",
    )
    story = build_story(manifest)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT), help="Destination PDF path.")
    parser.add_argument("--manifest", default=str(MANIFEST_DEFAULT), help="Destination JSON manifest path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    build_pdf(output_path, manifest_path)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote PDF: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
