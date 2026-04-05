#!/usr/bin/env python3
"""End-to-end TLlib VOC->Clipart transfer-learning demo wrapper."""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

QUESTION_DIR = Path(__file__).resolve().parent
REPO_ROOT = QUESTION_DIR.parent
DEFAULT_REPO_DIR = REPO_ROOT / "external" / "Transfer-Learning-Library"
DEFAULT_DATASET_ROOT = QUESTION_DIR / "datasets"
DEFAULT_SMOKE_DATASET_ROOT = QUESTION_DIR / "datasets_smoke"
DEFAULT_SMOKE_SOURCE_OUTPUT_DIR = (
    QUESTION_DIR / "logs" / "source_only_smoke" / "faster_rcnn_R_101_C4" / "voc2clipart"
)
DEFAULT_SMOKE_ADAPT_OUTPUT_ROOT = (
    QUESTION_DIR / "logs" / "d_adapt_smoke" / "faster_rcnn_R_101_C4" / "voc2clipart"
)
DEFAULT_SMOKE_VISUALIZATION_ROOT = QUESTION_DIR / "visualizations" / "voc2clipart_smoke"
DEFAULT_SMOKE_SUMMARY_DIR = QUESTION_DIR / "outputs" / "voc2clipart_smoke"
DEFAULT_BENCHMARK_SOURCE_OUTPUT_DIR = (
    QUESTION_DIR / "logs" / "source_only_benchmark" / "faster_rcnn_R_101_C4" / "voc2clipart"
)
DEFAULT_BENCHMARK_ADAPT_OUTPUT_ROOT = (
    QUESTION_DIR / "logs" / "d_adapt_benchmark" / "faster_rcnn_R_101_C4" / "voc2clipart"
)
DEFAULT_BENCHMARK_VISUALIZATION_ROOT = QUESTION_DIR / "visualizations" / "voc2clipart_benchmark"
DEFAULT_BENCHMARK_SUMMARY_DIR = QUESTION_DIR / "outputs" / "voc2clipart_benchmark"
REQUIRED_MODULES = ("detectron2", "timm")
REQUIRED_DATASET_SUBPATHS = (
    Path("Annotations"),
    Path("JPEGImages"),
    Path("ImageSets") / "Main",
)
DEFAULT_ADAPT_CONFIDENCE_RATIOS = (0.1, 0.2)
# Profile defaults let one wrapper serve two roles: a tiny local smoke test and a
# fuller OSC-style benchmark run with separate output directories.
PROFILE_DEFAULTS = {
    "smoke": {
        "max_iter": 1,
        "checkpoint_period": 1,
        "eval_period": 1,
        "adapt_max_iter": 1,
        "adapt_checkpoint_period": 1,
        "adapt_eval_period": 1,
        "dataloader_workers": 0,
        "category_workers": 0,
        "bbox_workers": 0,
        "max_train_c": 2,
        "max_val_c": 1,
        "max_train_b": 2,
        "max_val_b": 2,
        "batch_size_c": 16,
        "batch_size_b": 8,
        "epochs_c": 1,
        "iters_per_epoch_c": 5,
        "print_freq_c": 1,
        "pretrain_epochs_b": 1,
        "epochs_b": 1,
        "iters_per_epoch_b": 5,
        "print_freq_b": 1,
        "phase_count": 1,
        "adapt_confidence_ratios": "0.1,0.2",
        "source_output_dir": str(DEFAULT_SMOKE_SOURCE_OUTPUT_DIR),
        "adapt_output_root": str(DEFAULT_SMOKE_ADAPT_OUTPUT_ROOT),
        "visualization_root": str(DEFAULT_SMOKE_VISUALIZATION_ROOT),
        "summary_dir": str(DEFAULT_SMOKE_SUMMARY_DIR),
    },
    "benchmark": {
        "phase_count": 3,
        "adapt_confidence_ratios": "0.1,0.2",
        "source_output_dir": str(DEFAULT_BENCHMARK_SOURCE_OUTPUT_DIR),
        "adapt_output_root": str(DEFAULT_BENCHMARK_ADAPT_OUTPUT_ROOT),
        "visualization_root": str(DEFAULT_BENCHMARK_VISUALIZATION_ROOT),
        "summary_dir": str(DEFAULT_BENCHMARK_SUMMARY_DIR),
    },
}

DATASET_SPECS = {
    "VOC2007": {
        "dir_name": "VOC2007",
        "archive_name": "VOC2007.tgz",
        "url": "https://cloud.tsinghua.edu.cn/f/800a9495d3b74612be3f/?dl=1",
    },
    "VOC2012": {
        "dir_name": "VOC2012",
        "archive_name": "VOC2012.tgz",
        "url": "https://cloud.tsinghua.edu.cn/f/a7e7ab88f727408eaf32/?dl=1",
    },
    "Clipart": {
        "dir_name": "clipart",
        "archive_name": "clipart.zip",
        "url": "https://cloud.tsinghua.edu.cn/f/c853a66786e2416a8f18/?dl=1",
    },
}


# Basic path/runtime helpers normalize everything before any TLlib command is built.
def resolve_path(path_str: str | None, base: Path = QUESTION_DIR) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def resolve_python(repo_dir: Path, requested: str | None) -> str:
    if requested:
        return requested
    venv_python = repo_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def resolve_output_dir(output_dir: str) -> Path:
    path = resolve_path(output_dir)
    assert path is not None
    return path


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    # The wrapper exposes many overrides, but most runs should only need
    # --profile smoke or --profile benchmark. This fills in the rest.
    for field, value in PROFILE_DEFAULTS[args.profile].items():
        if getattr(args, field) is None:
            setattr(args, field, value)
    return args


def find_object_detection_dir(repo_dir: Path) -> Path:
    # Prefer the canonical TLlib location, then fall back to a search so the
    # wrapper still works if the upstream repo layout changes slightly.
    candidate = repo_dir / "examples" / "domain_adaptation" / "object_detection"
    if (candidate / "source_only.py").exists():
        return candidate

    for found in repo_dir.rglob("source_only.py"):
        return found.parent

    raise FileNotFoundError(
        "Could not locate TLlib object_detection/source_only.py inside "
        f"{repo_dir}."
    )


def find_dadapt_dir(object_detection_dir: Path) -> Path:
    # Some TLlib checkouts use d_adapt/ while others have d-adapt/ naming, so
    # we probe both instead of assuming one exact folder name.
    for name in ("d_adapt", "d-adapt"):
        candidate = object_detection_dir / name
        if (candidate / "d_adapt.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate d_adapt.py under {object_detection_dir}."
    )


def find_missing_modules(python_bin: str, modules: tuple[str, ...]) -> list[str]:
    probe = subprocess.run(
        [
            python_bin,
            "-c",
            (
                "import importlib.util, sys\n"
                "missing = [name for name in sys.argv[1:] "
                "if importlib.util.find_spec(name) is None]\n"
                "print('\\n'.join(missing))\n"
                "raise SystemExit(1 if missing else 0)\n"
            ),
            *modules,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return []
    return [line.strip() for line in probe.stdout.splitlines() if line.strip()]


def run_python_probe(python_bin: str, code: str) -> str:
    result = subprocess.run(
        [python_bin, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_python_environment(python_bin: str) -> None:
    missing = find_missing_modules(python_bin, REQUIRED_MODULES)
    if not missing:
        return

    # Stop before launching any long training job if the Detectron2 stack is not
    # importable; this is the most common failure mode on fresh machines.
    missing_fmt = ", ".join(missing)
    raise RuntimeError(
        "Question 7 cannot run yet because the TLlib environment is missing "
        f"required Python packages: {missing_fmt}\n\n"
        "This wrapper launches TLlib's Detectron2-based object detection demo, "
        "so detectron2 and timm must be installed before source_only.py or "
        "d_adapt.py can run.\n\n"
        "Suggested next steps:\n"
        "  1) bash question_7_transfer_learning/setup_tllib_osc.sh\n"
        "  2) install detectron2 for your Python / Torch / CUDA platform "
        "(or rerun setup with INSTALL_DETECTRON2=1 if compatible)\n"
        "  3) rerun this wrapper with --mode doctor or --mode full-pipeline\n"
    )


def dataset_layout_ok(path: Path) -> bool:
    return all((path / subpath).exists() for subpath in REQUIRED_DATASET_SUBPATHS)


def validate_dataset_dir(label: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} dataset path does not exist: {path}\n"
            "Run --mode prepare-datasets first or pass --download-datasets."
        )
    if not dataset_layout_ok(path):
        missing = [str(subpath) for subpath in REQUIRED_DATASET_SUBPATHS if not (path / subpath).exists()]
        raise FileNotFoundError(
            f"{label} dataset path is incomplete: {path}\n"
            f"Missing: {', '.join(missing)}"
        )


def validate_source_only_inputs(
    object_detection_dir: Path,
    dataset_paths: dict[str, Path],
    config_file: Path,
) -> None:
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file does not exist: {config_file}\n"
            "Pass --config-file relative to TLlib's object_detection folder or as an absolute path."
        )
    validate_dataset_dir("VOC2007", dataset_paths["VOC2007"])
    validate_dataset_dir("VOC2012", dataset_paths["VOC2012"])
    validate_dataset_dir("Clipart", dataset_paths["Clipart"])
    if not (object_detection_dir / "source_only.py").exists():
        raise FileNotFoundError(f"source_only.py not found in {object_detection_dir}")


def validate_dadapt_inputs(
    dadapt_dir: Path,
    dataset_paths: dict[str, Path],
    config_file: Path,
    weights: Path,
) -> None:
    validate_source_only_inputs(dadapt_dir.parent, dataset_paths, config_file)
    if not (dadapt_dir / "d_adapt.py").exists():
        raise FileNotFoundError(f"d_adapt.py not found in {dadapt_dir}")
    if not weights.exists():
        raise FileNotFoundError(
            f"Required pretrained checkpoint does not exist: {weights}\n"
            "Run source-only training first, or point MODEL.WEIGHTS to an existing checkpoint."
        )


# Dataset helpers download or derive the exact VOC/Clipart layouts that the TLlib
# object-detection examples expect to find.
def extract_archive(archive_path: Path, destination_root: Path) -> None:
    suffixes = archive_path.suffixes
    # Clipart ships as zip while VOC ships as tar; the wrapper hides that
    # packaging difference so dataset prep becomes one command.
    if suffixes[-1] == ".zip":
        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(destination_root)
        return
    with tarfile.open(archive_path) as handle:
        handle.extractall(destination_root)


def download_dataset(target_dir: Path, archive_name: str, url: str) -> None:
    if dataset_layout_ok(target_dir):
        print(f"Dataset already available: {target_dir}", flush=True)
        return

    # Keep the downloaded archives next to the extracted folders so repeated
    # runs can reuse them without re-downloading large files.
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir.parent / archive_name
    if not archive_path.exists():
        print(f"Downloading {archive_name} -> {archive_path}", flush=True)
        urlretrieve(url, archive_path)
    else:
        print(f"Using existing archive: {archive_path}", flush=True)

    print(f"Extracting {archive_path.name} into {target_dir.parent}", flush=True)
    extract_archive(archive_path, target_dir.parent)
    validate_dataset_dir(target_dir.name, target_dir)


def write_split_file(path: Path, image_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(image_ids) + ("\n" if image_ids else ""), encoding="utf-8")


def ensure_trainval_split(dataset_dir: Path) -> None:
    main_dir = dataset_dir / "ImageSets" / "Main"
    trainval_path = main_dir / "trainval.txt"
    if trainval_path.exists():
        return

    # Clipart sometimes lacks trainval.txt, but TLlib expects it. Rebuild it by
    # merging train/test IDs so the dataset matches the script's assumptions.
    train_path = main_dir / "train.txt"
    test_path = main_dir / "test.txt"
    train_ids = []
    test_ids = []
    if train_path.exists():
        train_ids = [line.strip() for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if test_path.exists():
        test_ids = [line.strip() for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    merged_ids: list[str] = []
    seen: set[str] = set()
    for image_id in train_ids + test_ids:
        if image_id in seen:
            continue
        seen.add(image_id)
        merged_ids.append(image_id)

    if not merged_ids:
        raise FileNotFoundError(
            f"Could not synthesize trainval.txt for dataset at {dataset_dir}; "
            "neither train.txt nor test.txt was found."
        )

    write_split_file(trainval_path, merged_ids)


def read_split_file(dataset_dir: Path, split_name: str) -> list[str]:
    split_path = dataset_dir / "ImageSets" / "Main" / f"{split_name}.txt"
    if split_name == "trainval" and not split_path.exists():
        ensure_trainval_split(dataset_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
    return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_subset_files(source_dir: Path, dest_dir: Path, image_ids: set[str]) -> None:
    for subdir, extensions in (
        ("Annotations", [".xml"]),
        ("JPEGImages", [".jpg", ".jpeg", ".png"]),
    ):
        # Copy only the files referenced by the chosen split IDs so the smoke
        # subset preserves the VOC-style directory layout while staying tiny.
        (dest_dir / subdir).mkdir(parents=True, exist_ok=True)
        for image_id in sorted(image_ids):
            source_base = source_dir / subdir / image_id
            matched = None
            for extension in extensions:
                candidate = source_base.with_suffix(extension)
                if candidate.exists():
                    matched = candidate
                    break
            if matched is None:
                raise FileNotFoundError(
                    f"Could not find {subdir} file for image id '{image_id}' in {source_dir}"
                )
            shutil.copy2(matched, dest_dir / subdir / matched.name)


def create_smoke_subset(
    source_dir: Path,
    dest_dir: Path,
    split_counts: dict[str, int],
) -> None:
    if dataset_layout_ok(dest_dir):
        try:
            if all(len(read_split_file(dest_dir, split_name)) == count for split_name, count in split_counts.items()):
                print(f"Smoke subset already available: {dest_dir}", flush=True)
                return
        except FileNotFoundError:
            pass
        print(f"Rebuilding smoke subset: {dest_dir}", flush=True)
    else:
        print(f"Building smoke subset: {dest_dir}", flush=True)

    # Preserve per-split semantics while truncating each split to a tiny prefix.
    # That gives a deterministic local test set without inventing new metadata.
    selected_per_split: dict[str, list[str]] = {}
    for split_name, count in split_counts.items():
        image_ids = read_split_file(source_dir, split_name)
        selected_per_split[split_name] = image_ids[: min(count, len(image_ids))]

    union_ids: set[str] = set()
    for image_ids in selected_per_split.values():
        union_ids.update(image_ids)

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    copy_subset_files(source_dir, dest_dir, union_ids)

    for split_name, image_ids in selected_per_split.items():
        write_split_file(dest_dir / "ImageSets" / "Main" / f"{split_name}.txt", image_ids)

    validate_dataset_dir(dest_dir.name, dest_dir)


def build_smoke_detector_overrides(args: argparse.Namespace) -> list[str]:
    if args.profile != "smoke":
        return []

    # Smoke mode aggressively shrinks the Detectron2 workload so the demo can
    # finish locally without editing TLlib's upstream config files.
    return [
        "SOLVER.IMS_PER_BATCH",
        str(args.smoke_ims_per_batch),
        "SOLVER.WARMUP_ITERS",
        str(args.smoke_warmup_iters),
        "MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE",
        str(args.smoke_roi_batch_size),
        "INPUT.MIN_SIZE_TRAIN",
        f"({args.smoke_min_size_train},)",
        "INPUT.MAX_SIZE_TRAIN",
        str(args.smoke_max_size_train),
        "INPUT.MIN_SIZE_TEST",
        str(args.smoke_min_size_test),
        "INPUT.MAX_SIZE_TEST",
        str(args.smoke_max_size_test),
        "MODEL.RPN.PRE_NMS_TOPK_TRAIN",
        str(args.smoke_pre_nms_topk_train),
        "MODEL.RPN.POST_NMS_TOPK_TRAIN",
        str(args.smoke_post_nms_topk_train),
        "MODEL.RPN.PRE_NMS_TOPK_TEST",
        str(args.smoke_pre_nms_topk_test),
        "MODEL.RPN.POST_NMS_TOPK_TEST",
        str(args.smoke_post_nms_topk_test),
        "TEST.DETECTIONS_PER_IMAGE",
        str(args.smoke_detections_per_image),
    ]


def ensure_full_datasets(dataset_paths: dict[str, Path], allow_download: bool) -> None:
    for dataset_name, target_dir in dataset_paths.items():
        if dataset_layout_ok(target_dir):
            if dataset_name == "Clipart":
                # Normalize Clipart even when the raw files already exist.
                ensure_trainval_split(target_dir)
            continue
        if not allow_download:
            validate_dataset_dir(dataset_name, target_dir)
        spec = DATASET_SPECS[dataset_name]
        download_dataset(target_dir, spec["archive_name"], spec["url"])
        if dataset_name == "Clipart":
            ensure_trainval_split(target_dir)


def ensure_smoke_subsets(
    full_dataset_paths: dict[str, Path],
    smoke_dataset_paths: dict[str, Path],
    smoke_train_count: int,
    smoke_test_count: int,
) -> None:
    # Each dataset gets its own reduced clone so smoke mode can point TLlib at a
    # separate root instead of mutating or trimming the full datasets.
    create_smoke_subset(
        full_dataset_paths["VOC2007"],
        smoke_dataset_paths["VOC2007"],
        {"trainval": smoke_train_count, "test": smoke_test_count},
    )
    create_smoke_subset(
        full_dataset_paths["VOC2012"],
        smoke_dataset_paths["VOC2012"],
        {"trainval": smoke_train_count},
    )
    create_smoke_subset(
        full_dataset_paths["Clipart"],
        smoke_dataset_paths["Clipart"],
        {"trainval": smoke_train_count},
    )


def parse_adapt_confidence_ratios(raw_value: str) -> list[float]:
    # Later D-adapt phases can tighten pseudo-label confidence thresholds.
    # Comma-separated CLI input keeps that schedule easy to explain and tweak.
    if not raw_value.strip():
        return list(DEFAULT_ADAPT_CONFIDENCE_RATIOS)
    return [float(part.strip()) for part in raw_value.split(",") if part.strip()]


def build_solver_steps_value(max_iter: int) -> str:
    # When we shorten training for smoke mode, the LR step schedule also needs
    # to shrink; otherwise Detectron2 would step outside the tiny run window.
    if max_iter <= 1:
        return "()"

    step_1 = max(1, max_iter // 2)
    step_2 = max(step_1 + 1, int(max_iter * 0.8))
    steps = sorted({step for step in (step_1, step_2) if step < max_iter})
    if not steps:
        return "()"
    if len(steps) == 1:
        return f"({steps[0]},)"
    return f"({','.join(str(step) for step in steps)})"


# Command builders translate this local CLI into the three upstream TLlib stages:
# source-only training, D-adapt refinement, and visualization.
def append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def append_cfg_override(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    command.extend([key, str(value)])


def build_source_only_command(
    python_bin: str,
    config_file: Path,
    dataset_paths: dict[str, Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    # This stage trains the baseline detector on VOC and evaluates it on both
    # VOC2007 test and the target-domain Clipart set.
    command = [
        python_bin,
        "source_only.py",
        "--config-file",
        str(config_file),
        "-s",
        "VOC2007",
        str(dataset_paths["VOC2007"]),
        "VOC2012",
        str(dataset_paths["VOC2012"]),
        "-t",
        "Clipart",
        str(dataset_paths["Clipart"]),
        "--test",
        "VOC2007Test",
        str(dataset_paths["VOC2007"]),
        "Clipart",
        str(dataset_paths["Clipart"]),
        "--finetune",
    ]
    if args.max_iter is not None:
        append_cfg_override(command, "SOLVER.MAX_ITER", args.max_iter)
        append_cfg_override(command, "SOLVER.STEPS", build_solver_steps_value(args.max_iter))
    append_cfg_override(command, "SOLVER.CHECKPOINT_PERIOD", args.checkpoint_period)
    append_cfg_override(command, "TEST.EVAL_PERIOD", args.eval_period)
    append_cfg_override(command, "DATALOADER.NUM_WORKERS", args.dataloader_workers)
    append_cfg_override(command, "MODEL.DEVICE", args.resolved_device)
    append_cfg_override(command, "OUTPUT_DIR", output_dir)
    command.extend(build_smoke_detector_overrides(args))
    if args.resume:
        command.append("--resume")
    return command


def build_dadapt_command(
    python_bin: str,
    config_file: Path,
    dataset_paths: dict[str, Path],
    output_dir: Path,
    model_weights: Path,
    phase_index: int,
    phase_confidence_ratio: float | None,
    args: argparse.Namespace,
) -> list[str]:
    # D-adapt keeps the same detector backbone/config but adds TLlib's category
    # and bbox adaptor training around a starting checkpoint.
    command = [
        python_bin,
        "d_adapt.py",
        "--config-file",
        str(config_file),
        "-s",
        "VOC2007",
        str(dataset_paths["VOC2007"]),
        "VOC2012",
        str(dataset_paths["VOC2012"]),
        "-t",
        "Clipart",
        str(dataset_paths["Clipart"]),
        "--test",
        "Clipart",
        str(dataset_paths["Clipart"]),
        "--finetune",
        "--bbox-refine",
    ]
    append_option(command, "--workers-c", args.category_workers)
    append_option(command, "--workers-b", args.bbox_workers)
    append_option(command, "--max-train-c", args.max_train_c)
    append_option(command, "--max-val-c", args.max_val_c)
    append_option(command, "--max-train-b", args.max_train_b)
    append_option(command, "--max-val-b", args.max_val_b)
    append_option(command, "--batch-size-c", args.batch_size_c)
    append_option(command, "--batch-size-b", args.batch_size_b)
    append_option(command, "--epochs-c", args.epochs_c)
    append_option(command, "--iters-per-epoch-c", args.iters_per_epoch_c)
    append_option(command, "--print-freq-c", args.print_freq_c)
    append_option(command, "--pretrain-epochs-b", args.pretrain_epochs_b)
    append_option(command, "--epochs-b", args.epochs_b)
    append_option(command, "--iters-per-epoch-b", args.iters_per_epoch_b)
    append_option(command, "--print-freq-b", args.print_freq_b)
    if args.adapt_max_iter is not None:
        append_cfg_override(command, "SOLVER.MAX_ITER", args.adapt_max_iter)
        append_cfg_override(command, "SOLVER.STEPS", build_solver_steps_value(args.adapt_max_iter))
    append_cfg_override(command, "SOLVER.CHECKPOINT_PERIOD", args.adapt_checkpoint_period)
    append_cfg_override(command, "TEST.EVAL_PERIOD", args.adapt_eval_period)
    append_cfg_override(command, "DATALOADER.NUM_WORKERS", args.dataloader_workers)
    append_cfg_override(command, "MODEL.DEVICE", args.resolved_device)
    append_cfg_override(command, "OUTPUT_DIR", output_dir)
    append_cfg_override(command, "MODEL.WEIGHTS", model_weights)
    append_cfg_override(command, "SEED", args.seed)
    command.extend(build_smoke_detector_overrides(args))
    if phase_index > 1 and phase_confidence_ratio is not None:
        # Phase 1 starts from the baseline model directly; later phases can
        # optionally refine pseudo-label confidence with a stricter ratio.
        command.extend(["--confidence-ratio-c", str(phase_confidence_ratio)])
    if args.resume:
        command.append("--resume")
    return command


def build_visualize_command(
    python_bin: str,
    config_file: Path,
    dataset_paths: dict[str, Path],
    save_path: Path,
    model_weights: Path,
    args: argparse.Namespace,
) -> list[str]:
    # Visualization is the presentation stage: it reuses TLlib's renderer to
    # save qualitative predictions from a chosen checkpoint.
    command = [
        python_bin,
        "visualize.py",
        "--config-file",
        str(config_file),
        "--test",
        "Clipart",
        str(dataset_paths["Clipart"]),
        "--save-path",
        str(save_path),
        "--n-visualizations",
        str(args.n_visualizations),
        "--threshold",
        str(args.threshold),
        "--n-bboxes",
        str(args.n_bboxes),
        "MODEL.DEVICE",
        args.resolved_device,
        "MODEL.WEIGHTS",
        str(model_weights),
    ]
    command.extend(build_smoke_detector_overrides(args))
    return command


def resolve_visualize_config(
    stage_name: str,
    source_config_file: Path,
    dadapt_config_file: Path,
) -> Path:
    # D-adapt checkpoints belong with the D-adapt config; the source-only
    # baseline keeps the original detector config.
    return source_config_file if stage_name == "source-only" else dadapt_config_file


def resolve_expected_stage_path(expected_output: Path) -> Path:
    return expected_output.parent if expected_output.suffix else expected_output


# Stage helpers centralize logging, skip/reuse behavior, and summary extraction.
def print_command(command: list[str], cwd: Path) -> None:
    print(f"Working directory: {cwd}", flush=True)
    print("Running:", " ".join(shlex.quote(part) for part in command), flush=True)


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print_command(command, cwd)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def maybe_run_stage(
    stage_name: str,
    command: list[str],
    cwd: Path,
    expected_output: Path,
    dry_run: bool,
    force: bool,
) -> None:
    # Treat a completed checkpoint directory as the contract for that stage.
    # This makes the wrapper resumable across repeated demo sessions.
    if expected_output.exists() and not force:
        print(
            f"Skipping {stage_name}: expected output already exists at {expected_output}",
            flush=True,
        )
        return
    if force and not dry_run:
        stage_path = resolve_expected_stage_path(expected_output)
        if stage_path.exists():
            # force removes only the specific stage artifact directory/file so
            # the rerun starts cleanly without touching unrelated outputs.
            print(f"Cleaning existing stage output: {stage_path}", flush=True)
            if stage_path.is_dir():
                shutil.rmtree(stage_path)
            else:
                stage_path.unlink()
    print(f"Stage: {stage_name}", flush=True)
    run_command(command, cwd=cwd, dry_run=dry_run)


def parse_eval_metrics_from_log(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None

    # TLlib/Detectron2 logs metrics in more than one textual format depending on
    # the code path, so the parser accepts either the dict payload or the table.
    latest: dict[str, Any] | None = None
    table_metrics: dict[str, Any] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        brace_index = line.find("{'bbox':")
        if brace_index == -1:
            table_match = re.match(r"^\|\s*([^|]+?)\s*\|\s*([0-9.]+|nan)\s*\|$", line)
            if table_match:
                key = table_match.group(1).strip()
                value_str = table_match.group(2).strip()
                if key and key != "class":
                    try:
                        table_metrics[key] = float(value_str)
                    except ValueError:
                        pass
            continue
        payload = line[brace_index:]
        payload = payload.replace("OrderedDict(", "").rstrip(")")
        payload = re.sub(r"np\.float64\(([^)]+)\)", r"\1", payload)
        try:
            parsed = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict) and "bbox" in parsed:
            latest = parsed["bbox"]
    if latest is not None:
        return latest
    return table_metrics or None


def summarize_stage(
    stage_name: str,
    output_dir: Path,
    visualization_dir: Path | None = None,
) -> dict[str, Any]:
    log_path = output_dir / "log.txt"
    model_final = output_dir / "model_final.pth"
    checkpoints = sorted(output_dir.glob("model_*.pth"))
    # Summaries focus on the artifacts a professor cares about: did the stage
    # run, did it produce a final checkpoint, and what metrics were reported.
    summary = {
        "stage": stage_name,
        "output_dir": str(output_dir),
        "log_exists": log_path.exists(),
        "model_final_exists": model_final.exists(),
        "checkpoint_count": len(checkpoints),
        "metrics": parse_eval_metrics_from_log(log_path),
    }
    adaptor_status_path = output_dir / "adaptor_status.json"
    if adaptor_status_path.exists():
        summary["adaptor_status"] = json.loads(adaptor_status_path.read_text(encoding="utf-8"))
    if visualization_dir is not None:
        png_count = len(list(visualization_dir.rglob("*.png")))
        summary["visualization_dir"] = str(visualization_dir)
        summary["visualization_count"] = png_count
    return summary


def write_summary(summary_dir: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "summary.json"
    md_path = summary_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The markdown file is the human-readable handoff, while the JSON preserves
    # the same facts in a machine-friendly form.
    lines = [
        "# TLlib VOC->Clipart Transfer Summary",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Python: `{summary['environment'].get('python', 'unknown')}`",
        f"- Torch: `{summary['environment'].get('torch', 'missing')}`",
        f"- Detectron2: `{summary['environment'].get('detectron2', 'missing')}`",
        "",
    ]
    for stage in summary["stages"]:
        lines.append(f"## {stage['stage']}")
        lines.append(f"- Output dir: `{stage['output_dir']}`")
        lines.append(f"- model_final.pth: `{stage['model_final_exists']}`")
        lines.append(f"- Checkpoints: `{stage['checkpoint_count']}`")
        if stage.get("metrics"):
            metrics = stage["metrics"]
            ap = metrics.get("AP", "n/a")
            ap50 = metrics.get("AP50", "n/a")
            ap75 = metrics.get("AP75", "n/a")
            lines.append(f"- Metrics: `AP={ap}`, `AP50={ap50}`, `AP75={ap75}`")
        else:
            lines.append("- Metrics: `not found in log.txt`")
        if stage.get("visualization_count") is not None:
            lines.append(f"- Visualizations: `{stage['visualization_count']}`")
        if stage.get("adaptor_status"):
            category_status = stage["adaptor_status"].get("category", {}).get("status", "unknown")
            bbox_status = stage["adaptor_status"].get("bbox", {}).get("status", "unknown")
            lines.append(f"- Adaptors: `category={category_status}`, `bbox={bbox_status}`")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def collect_environment_summary(python_bin: str) -> dict[str, str]:
    summary = {
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        summary["python"] = run_python_probe(
            python_bin,
            "import sys; print(sys.version.split()[0])",
        )
    except subprocess.CalledProcessError:
        summary["python"] = "unavailable"

    for module_name in ("torch", "torchvision", "detectron2", "timm"):
        try:
            version = run_python_probe(
                python_bin,
                f"import {module_name}; print(getattr({module_name}, '__version__', 'unknown'))",
            )
        except subprocess.CalledProcessError:
            version = "missing"
        summary[module_name] = version

    if summary["platform"].startswith("macOS") and summary["machine"] == "arm64":
        summary["note"] = (
            "Apple Silicon macOS can be slow for this Detectron2 pipeline. "
            "Use smoke profile locally; prefer OSC/Linux for full runs."
        )
    return summary


def resolve_model_device(
    requested_device: str,
    python_bin: str,
    env_summary: dict[str, str],
) -> str:
    # 'auto' means "ask the actual runtime" rather than guessing from the host.
    # If torch is missing or fails to import, fall back to CPU deterministically.
    if requested_device != "auto":
        return requested_device
    torch_version = env_summary.get("torch", "missing")
    if torch_version == "missing":
        return "cpu"
    try:
        return run_python_probe(
            python_bin,
            "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')",
        )
    except subprocess.CalledProcessError:
        return "cpu"


def build_dataset_paths_from_root(dataset_root: Path) -> dict[str, Path]:
    return {
        "VOC2007": dataset_root / "VOC2007",
        "VOC2012": dataset_root / "VOC2012",
        "Clipart": dataset_root / "clipart",
    }


def resolve_dataset_paths(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    dataset_root = resolve_path(args.dataset_root)
    smoke_dataset_root = resolve_path(args.smoke_dataset_root)
    assert dataset_root is not None and smoke_dataset_root is not None

    full_paths = build_dataset_paths_from_root(dataset_root)
    smoke_paths = build_dataset_paths_from_root(smoke_dataset_root)

    # The active dataset root switches with the profile, but explicit CLI paths
    # still win so the wrapper can target custom dataset locations when needed.
    active_paths = smoke_paths if args.profile == "smoke" else full_paths

    overrides = {
        "VOC2007": resolve_path(args.voc2007_path),
        "VOC2012": resolve_path(args.voc2012_path),
        "Clipart": resolve_path(args.clipart_path),
    }
    for key, value in overrides.items():
        if value is not None:
            active_paths[key] = value
    return full_paths, smoke_paths, active_paths


def resolve_config_file(
    config_file_value: str,
    primary_dir: Path,
    fallback_dir: Path | None = None,
) -> Path:
    # Accept either an absolute path or the same relative path anchored under
    # TLlib's source-only and D-adapt folders.
    raw = Path(config_file_value).expanduser()
    if raw.is_absolute():
        return raw
    primary = (primary_dir / raw).resolve()
    if primary.exists():
        return primary
    if fallback_dir is not None:
        fallback = (fallback_dir / raw).resolve()
        if fallback.exists():
            return fallback
    return primary


def determine_latest_phase(adapt_output_root: Path, phase_count: int) -> int | None:
    latest: int | None = None
    for phase_index in range(1, phase_count + 1):
        if (adapt_output_root / f"phase{phase_index}" / "model_final.pth").exists():
            latest = phase_index
    return latest


def resolve_stage_weights(
    stage: str,
    source_output_dir: Path,
    adapt_output_root: Path,
    phase_count: int,
    allow_missing_latest: bool = False,
) -> tuple[str, Path]:
    # Visualization/reporting can target the baseline or the latest completed
    # adaptation phase, so centralize that checkpoint selection logic here.
    if stage == "source-only":
        return stage, source_output_dir / "model_final.pth"
    if stage == "latest":
        latest_phase = determine_latest_phase(adapt_output_root, phase_count)
        if latest_phase is None and allow_missing_latest:
            latest_phase = phase_count
        if latest_phase is None:
            raise FileNotFoundError(
                f"No completed D-adapt phase found under {adapt_output_root}"
            )
        return f"phase{latest_phase}", adapt_output_root / f"phase{latest_phase}" / "model_final.pth"
    if not stage.startswith("phase"):
        raise ValueError(f"Unsupported stage value: {stage}")
    return stage, adapt_output_root / stage / "model_final.pth"


# The parser exposes both the high-level modes and the most important TLlib knobs,
# while profile defaults fill in the rest.
def print_doctor_summary(
    env_summary: dict[str, str],
    full_dataset_paths: dict[str, Path],
    smoke_dataset_paths: dict[str, Path],
) -> None:
    # doctor is meant to answer "is this machine ready?" at a glance, so it
    # prints dependency and dataset readiness without running any training code.
    print("Environment summary:", flush=True)
    for key in ("platform", "machine", "python", "torch", "torchvision", "detectron2", "timm"):
        print(f"  {key}: {env_summary.get(key, 'unknown')}", flush=True)
    if "note" in env_summary:
        print(f"  note: {env_summary['note']}", flush=True)

    print("Dataset status:", flush=True)
    for label, path in full_dataset_paths.items():
        print(
            f"  full {label}: {'ready' if dataset_layout_ok(path) else 'missing'} -> {path}",
            flush=True,
        )
    for label, path in smoke_dataset_paths.items():
        print(
            f"  smoke {label}: {'ready' if dataset_layout_ok(path) else 'missing'} -> {path}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TLlib VOC->Clipart wrapper for environment checks, dataset prep, "
            "source-only training, D-adapt phases, visualization, and summary generation."
        )
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help="Path to local thuml/Transfer-Learning-Library clone.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter used to run TLlib scripts.",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "doctor",
            "prepare-datasets",
            "help",
            "source-only",
            "d-adapt",
            "visualize",
            "report",
            "full-pipeline",
        ),
        default="doctor",
        help="Pipeline action to perform.",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "benchmark"),
        default="smoke",
        help=(
            "smoke uses tiny subsets and wrapper-controlled short runs; "
            "benchmark uses full datasets, benchmark output folders, and upstream-style defaults."
        ),
    )
    # Core repo/config/data locations.
    parser.add_argument(
        "--config-file",
        default="config/faster_rcnn_R_101_C4_voc.yaml",
        help="Config file relative to TLlib object_detection/ or an absolute path.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Root directory for full VOC2007/VOC2012/Clipart datasets.",
    )
    parser.add_argument(
        "--smoke-dataset-root",
        default=str(DEFAULT_SMOKE_DATASET_ROOT),
        help="Root directory for reduced local-validation subsets.",
    )
    parser.add_argument("--voc2007-path", default=None)
    parser.add_argument("--voc2012-path", default=None)
    parser.add_argument("--clipart-path", default=None)
    parser.add_argument(
        "--download-datasets",
        action="store_true",
        help="Download VOC2007/VOC2012/Clipart if they are missing.",
    )
    parser.add_argument(
        "--smoke-train-count",
        type=int,
        default=4,
        help="Number of images per train split when creating smoke subsets.",
    )
    parser.add_argument(
        "--smoke-test-count",
        type=int,
        default=2,
        help="Number of VOC2007 test images when creating smoke subsets.",
    )
    # Detector schedule overrides.
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Override source-only detector iterations. Omit for benchmark profile to use the config default.",
    )
    parser.add_argument("--checkpoint-period", type=int, default=None)
    parser.add_argument("--eval-period", type=int, default=None)
    parser.add_argument(
        "--adapt-max-iter",
        type=int,
        default=None,
        help="Override detector iterations per D-adapt phase. Omit for benchmark profile to use the config default.",
    )
    parser.add_argument("--adapt-checkpoint-period", type=int, default=None)
    parser.add_argument("--adapt-eval-period", type=int, default=None)
    parser.add_argument("--dataloader-workers", type=int, default=None)
    parser.add_argument("--category-workers", type=int, default=None)
    parser.add_argument("--bbox-workers", type=int, default=None)
    parser.add_argument("--max-train-c", type=int, default=None)
    parser.add_argument("--max-val-c", type=int, default=None)
    parser.add_argument("--max-train-b", type=int, default=None)
    parser.add_argument("--max-val-b", type=int, default=None)
    parser.add_argument("--batch-size-c", type=int, default=None)
    parser.add_argument("--batch-size-b", type=int, default=None)
    parser.add_argument("--epochs-c", type=int, default=None)
    parser.add_argument("--iters-per-epoch-c", type=int, default=None)
    parser.add_argument("--print-freq-c", type=int, default=None)
    parser.add_argument("--pretrain-epochs-b", type=int, default=None)
    parser.add_argument("--epochs-b", type=int, default=None)
    parser.add_argument("--iters-per-epoch-b", type=int, default=None)
    parser.add_argument("--print-freq-b", type=int, default=None)
    parser.add_argument("--phase-count", type=int, default=None)
    parser.add_argument(
        "--adapt-confidence-ratios",
        default=None,
        help="Comma-separated confidence ratios for D-adapt phases after phase1.",
    )
    # Smoke-specific detector reductions so local runs finish quickly.
    parser.add_argument("--smoke-ims-per-batch", type=int, default=1)
    parser.add_argument("--smoke-roi-batch-size", type=int, default=32)
    parser.add_argument("--smoke-warmup-iters", type=int, default=0)
    parser.add_argument("--smoke-min-size-train", type=int, default=128)
    parser.add_argument("--smoke-max-size-train", type=int, default=256)
    parser.add_argument("--smoke-min-size-test", type=int, default=128)
    parser.add_argument("--smoke-max-size-test", type=int, default=256)
    parser.add_argument("--smoke-pre-nms-topk-train", type=int, default=256)
    parser.add_argument("--smoke-post-nms-topk-train", type=int, default=64)
    parser.add_argument("--smoke-pre-nms-topk-test", type=int, default=128)
    parser.add_argument("--smoke-post-nms-topk-test", type=int, default=32)
    parser.add_argument("--smoke-detections-per-image", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Model device override passed through as MODEL.DEVICE.",
    )
    # Output locations for checkpoints, PNGs, and the final summary.
    parser.add_argument(
        "--source-output-dir",
        default=None,
        help="Where source-only logs and checkpoints are written. Defaults depend on profile.",
    )
    parser.add_argument(
        "--adapt-output-root",
        default=None,
        help="Where D-adapt phase outputs are written. Defaults depend on profile.",
    )
    parser.add_argument(
        "--visualization-root",
        default=None,
        help="Where visualization PNGs are written. Defaults depend on profile.",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Where summary.json and summary.md are written. Defaults depend on profile.",
    )
    parser.add_argument(
        "--visualize-stage",
        choices=("source-only", "latest", "phase1", "phase2", "phase3"),
        default="latest",
        help="Checkpoint selection for --mode visualize.",
    )
    parser.add_argument("--n-visualizations", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-bboxes", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume through to TLlib training scripts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even if expected outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and planned outputs without executing training scripts.",
    )
    return parser


def main() -> int:
    try:
        # Resolve all paths, profile defaults, and environment facts once so each
        # mode branch below can focus only on its own stage logic.
        args = apply_profile_defaults(build_parser().parse_args())
        repo_dir = resolve_path(args.repo_dir)
        if repo_dir is None or not repo_dir.exists():
            raise FileNotFoundError(f"Repo directory does not exist: {repo_dir}")

        python_bin = resolve_python(repo_dir, args.python)
        object_detection_dir = find_object_detection_dir(repo_dir)
        dadapt_dir = find_dadapt_dir(object_detection_dir)
        source_config_file = resolve_config_file(args.config_file, object_detection_dir)
        dadapt_config_file = resolve_config_file(
            args.config_file,
            dadapt_dir,
            fallback_dir=object_detection_dir,
        )
        source_output_dir = resolve_output_dir(args.source_output_dir)
        adapt_output_root = resolve_output_dir(args.adapt_output_root)
        visualization_root = resolve_output_dir(args.visualization_root)
        summary_dir = resolve_output_dir(args.summary_dir)

        confidence_ratios = parse_adapt_confidence_ratios(args.adapt_confidence_ratios)
        # If fewer ratios are provided than phases, repeat the last one so every
        # post-phase1 stage still receives an explicit confidence setting.
        while len(confidence_ratios) < max(0, args.phase_count - 1):
            confidence_ratios.append(confidence_ratios[-1] if confidence_ratios else 0.1)

        full_dataset_paths, smoke_dataset_paths, active_dataset_paths = resolve_dataset_paths(args)
        env_summary = collect_environment_summary(python_bin)
        args.resolved_device = resolve_model_device(args.device, python_bin, env_summary)

        # doctor is the lightweight preflight: show environment status without mutating anything.
        if args.mode == "doctor":
            print_doctor_summary(env_summary, full_dataset_paths, smoke_dataset_paths)
            print(f"  resolved_model_device: {args.resolved_device}", flush=True)
            return 0

        # Dataset preparation is split out so downloads/subsets can be reused across runs.
        if args.mode in ("prepare-datasets", "full-pipeline") and not args.dry_run:
            ensure_full_datasets(full_dataset_paths, allow_download=args.download_datasets)
            if args.profile == "smoke":
                # Smoke subsets are derived only after the full datasets are
                # ready, because they are copied from those canonical roots.
                ensure_smoke_subsets(
                    full_dataset_paths,
                    smoke_dataset_paths,
                    smoke_train_count=args.smoke_train_count,
                    smoke_test_count=args.smoke_test_count,
                )

        if args.mode == "prepare-datasets":
            print_doctor_summary(env_summary, full_dataset_paths, smoke_dataset_paths)
            print("Active dataset root:", active_dataset_paths["VOC2007"].parent, flush=True)
            return 0

        if not args.dry_run:
            validate_python_environment(python_bin)

        # help delegates straight to TLlib so the wrapped script's own CLI stays visible.
        if args.mode == "help":
            command = [python_bin, "source_only.py", "--help"]
            run_command(command, object_detection_dir, dry_run=args.dry_run)
            return 0

        # source-only trains the baseline detector that later D-adapt phases refine.
        if args.mode == "source-only":
            if not args.dry_run:
                validate_source_only_inputs(object_detection_dir, active_dataset_paths, source_config_file)
            command = build_source_only_command(
                python_bin,
                source_config_file,
                active_dataset_paths,
                source_output_dir,
                args,
            )
            maybe_run_stage(
                "source-only",
                command,
                object_detection_dir,
                source_output_dir / "model_final.pth",
                dry_run=args.dry_run,
                force=args.force,
            )
            return 0

        # d-adapt runs one or more sequential refinement phases, each consuming the
        # checkpoint from the previous phase.
        if args.mode == "d-adapt":
            if not args.dry_run:
                validate_source_only_inputs(object_detection_dir, active_dataset_paths, source_config_file)
            source_weights = source_output_dir / "model_final.pth"
            for phase_index in range(1, args.phase_count + 1):
                # Each phase bootstraps from the checkpoint produced immediately
                # before it, which is what makes the adaptation sequential.
                weights = source_weights if phase_index == 1 else adapt_output_root / f"phase{phase_index - 1}" / "model_final.pth"
                if not args.dry_run:
                    validate_dadapt_inputs(dadapt_dir, active_dataset_paths, dadapt_config_file, weights)
                output_dir = adapt_output_root / f"phase{phase_index}"
                phase_ratio = None if phase_index == 1 else confidence_ratios[phase_index - 2]
                command = build_dadapt_command(
                    python_bin,
                    dadapt_config_file,
                    active_dataset_paths,
                    output_dir,
                    weights,
                    phase_index,
                    phase_ratio,
                    args,
                )
                maybe_run_stage(
                    f"d-adapt phase{phase_index}",
                    command,
                    dadapt_dir,
                    output_dir / "model_final.pth",
                    dry_run=args.dry_run,
                    force=args.force,
                )
            return 0

        # visualize renders predictions from either the baseline checkpoint or a chosen adapt phase.
        if args.mode == "visualize":
            if not args.dry_run:
                validate_source_only_inputs(object_detection_dir, active_dataset_paths, source_config_file)
            stage_name, weights = resolve_stage_weights(
                args.visualize_stage,
                source_output_dir,
                adapt_output_root,
                args.phase_count,
                allow_missing_latest=args.dry_run,
            )
            if not args.dry_run and not weights.exists():
                raise FileNotFoundError(f"Checkpoint for visualization does not exist: {weights}")
            save_path = visualization_root / stage_name
            visualize_config = resolve_visualize_config(stage_name, source_config_file, dadapt_config_file)
            command = build_visualize_command(
                python_bin,
                visualize_config,
                active_dataset_paths,
                save_path,
                weights,
                args,
            )
            maybe_run_stage(
                f"visualize {stage_name}",
                command,
                object_detection_dir,
                save_path,
                dry_run=args.dry_run,
                force=args.force,
            )
            return 0

        # report is read-only: summarize whatever logs/checkpoints already exist.
        if args.mode == "report":
            if args.dry_run:
                print(f"Would write {summary_dir / 'summary.json'}", flush=True)
                print(f"Would write {summary_dir / 'summary.md'}", flush=True)
                return 0
            # Report mode never retrains anything; it only inspects existing
            # artifacts and turns them into a compact written recap.
            stages = [
                summarize_stage("source-only", source_output_dir, visualization_root / "source-only"),
            ]
            for phase_index in range(1, args.phase_count + 1):
                stage_name = f"phase{phase_index}"
                stages.append(
                    summarize_stage(stage_name, adapt_output_root / stage_name, visualization_root / stage_name)
                )
            summary = {
                "profile": args.profile,
                "dataset_root": str(active_dataset_paths["VOC2007"].parent),
                "environment": env_summary,
                "stages": stages,
            }
            json_path, md_path = write_summary(summary_dir, summary)
            print(f"Wrote {json_path}", flush=True)
            print(f"Wrote {md_path}", flush=True)
            return 0

        # full-pipeline chains the entire demo in the order you would present it:
        # baseline -> adaptation -> visualization -> written summary.
        if args.mode == "full-pipeline":
            if not args.dry_run:
                validate_source_only_inputs(object_detection_dir, active_dataset_paths, source_config_file)
            source_command = build_source_only_command(
                python_bin,
                source_config_file,
                active_dataset_paths,
                source_output_dir,
                args,
            )
            maybe_run_stage(
                "source-only",
                source_command,
                object_detection_dir,
                source_output_dir / "model_final.pth",
                dry_run=args.dry_run,
                force=args.force,
            )

            source_weights = source_output_dir / "model_final.pth"
            for phase_index in range(1, args.phase_count + 1):
                # Run the refinement chain exactly as the demo story is told:
                # baseline first, then phase-by-phase domain adaptation.
                weights = source_weights if phase_index == 1 else adapt_output_root / f"phase{phase_index - 1}" / "model_final.pth"
                if not args.dry_run:
                    validate_dadapt_inputs(dadapt_dir, active_dataset_paths, dadapt_config_file, weights)
                output_dir = adapt_output_root / f"phase{phase_index}"
                phase_ratio = None if phase_index == 1 else confidence_ratios[phase_index - 2]
                command = build_dadapt_command(
                    python_bin,
                    dadapt_config_file,
                    active_dataset_paths,
                    output_dir,
                    weights,
                    phase_index,
                    phase_ratio,
                    args,
                )
                maybe_run_stage(
                    f"d-adapt phase{phase_index}",
                    command,
                    dadapt_dir,
                    output_dir / "model_final.pth",
                    dry_run=args.dry_run,
                    force=args.force,
                )

            visualization_plan = [
                ("source-only", source_output_dir / "model_final.pth"),
            ]
            latest_phase = determine_latest_phase(adapt_output_root, args.phase_count)
            if latest_phase is None and args.dry_run and args.phase_count > 0:
                latest_phase = args.phase_count
            if latest_phase is not None:
                # Show both the starting detector and the latest adapted model so
                # the qualitative before/after comparison is easy to present.
                visualization_plan.append(
                    (f"phase{latest_phase}", adapt_output_root / f"phase{latest_phase}" / "model_final.pth")
                )

            for stage_name, weights in visualization_plan:
                save_path = visualization_root / stage_name
                visualize_config = resolve_visualize_config(stage_name, source_config_file, dadapt_config_file)
                command = build_visualize_command(
                    python_bin,
                    visualize_config,
                    active_dataset_paths,
                    save_path,
                    weights,
                    args,
                )
                maybe_run_stage(
                    f"visualize {stage_name}",
                    command,
                    object_detection_dir,
                    save_path,
                    dry_run=args.dry_run,
                    force=args.force,
                )

            if args.dry_run:
                print(f"Would write {summary_dir / 'summary.json'}", flush=True)
                print(f"Would write {summary_dir / 'summary.md'}", flush=True)
                return 0

            stages = [
                summarize_stage("source-only", source_output_dir, visualization_root / "source-only"),
            ]
            for phase_index in range(1, args.phase_count + 1):
                stage_name = f"phase{phase_index}"
                stages.append(
                    summarize_stage(stage_name, adapt_output_root / stage_name, visualization_root / stage_name)
                )

            summary = {
                "profile": args.profile,
                "dataset_root": str(active_dataset_paths["VOC2007"].parent),
                "environment": env_summary,
                "stages": stages,
            }
            json_path, md_path = write_summary(summary_dir, summary)
            print(f"Wrote {json_path}", flush=True)
            print(f"Wrote {md_path}", flush=True)
            return 0

        raise ValueError(f"Unsupported mode: {args.mode}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(
            f"Error: TLlib command failed with exit code {exc.returncode}. "
            "See the command output above for details.",
            file=sys.stderr,
        )
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
