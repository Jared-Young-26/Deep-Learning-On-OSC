#!/usr/bin/env python3
"""Shared helpers for the Question 5 YOLO11 OBB pipeline."""

from __future__ import annotations

import importlib
import os
import sys
import zipfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

QUESTION_DIR = Path(__file__).resolve().parent
REPO_ROOT = QUESTION_DIR.parent

DEFAULT_REPO_DIR = REPO_ROOT / "external" / "ultralytics"

DEFAULT_INPUT_DIR = QUESTION_DIR / "inputs" / "satellite_images"
DEFAULT_OUTPUT_ROOT = QUESTION_DIR / "outputs" / "satellite_results"
DEFAULT_INDEX_CSV = DEFAULT_OUTPUT_ROOT / "index.csv"
DEFAULT_PREDICT_PROJECT_DIR = QUESTION_DIR / "runs" / "obb" / "predict"
DEFAULT_TRAIN_PROJECT_DIR = QUESTION_DIR / "runs" / "obb" / "train"

DATASETS_DIR = QUESTION_DIR / "datasets"
DOWNLOADS_DIR = DATASETS_DIR / "downloads"
DEFAULT_DOTA_ZIP = DOWNLOADS_DIR / "DOTAv1.zip"
DEFAULT_RAW_DOTA_DIR = DATASETS_DIR / "DOTAv1"
DEFAULT_SPLIT_DOTA_DIR = DATASETS_DIR / "DOTAv1-split"
DEFAULT_DOTA_YAML = DATASETS_DIR / "DOTAv1-split.yaml"

MODELS_DIR = QUESTION_DIR / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
DEFAULT_PRETRAINED_MODEL = PRETRAINED_MODELS_DIR / "yolo11s-obb.pt"
FINETUNED_MODELS_DIR = MODELS_DIR / "dota_obb"
DEFAULT_FINETUNED_MODEL = FINETUNED_MODELS_DIR / "best.pt"

DEFAULT_DOTA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"
DEFAULT_YOLO11_OBB_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-obb.pt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DOTA_CLASS_NAMES = {
    0: "plane",
    1: "ship",
    2: "storage tank",
    3: "baseball diamond",
    4: "tennis court",
    5: "basketball court",
    6: "ground track field",
    7: "harbor",
    8: "bridge",
    9: "large vehicle",
    10: "small vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer ball field",
    14: "swimming pool",
}

_PALETTE = [
    (235, 64, 52),
    (52, 152, 235),
    (52, 235, 140),
    (235, 188, 52),
    (176, 52, 235),
    (52, 235, 225),
    (235, 99, 52),
    (90, 52, 235),
    (121, 235, 52),
    (235, 52, 140),
    (52, 103, 235),
    (235, 232, 52),
    (52, 235, 194),
    (235, 52, 82),
    (82, 235, 52),
]


# Keep colors and class-name resolution centralized so training, bootstrapping,
# and inference all describe DOTA classes the same way.
def class_color(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def normalize_name_map(model_names) -> dict[int, str]:
    # Ultralytics may expose class names as either a dict or a list; normalize
    # both cases so later code can treat the mapping uniformly.
    if isinstance(model_names, dict):
        return {int(k): str(v) for k, v in model_names.items()}
    return {int(i): str(v) for i, v in enumerate(model_names)}


def parse_keep_classes(raw_value: str, name_map: dict[int, str]) -> set[int] | None:
    # Allow the CLI to filter by either numeric DOTA ids or human-readable names.
    if not raw_value:
        return None

    lowered_name_to_id = {name.lower(): idx for idx, name in name_map.items()}
    keep = set()
    for token in [part.strip() for part in raw_value.split(",") if part.strip()]:
        if token.isdigit():
            keep.add(int(token))
            continue

        class_id = lowered_name_to_id.get(token.lower())
        if class_id is None:
            known = ", ".join(sorted(lowered_name_to_id.keys()))
            raise ValueError(
                f"Unknown class token '{token}'. Use class ids or one of: {known}"
            )
        keep.add(class_id)

    return keep if keep else None


# Path and environment helpers make the scripts resilient to being launched from
# either the repo root or the question subdirectory.
def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def resolve_question_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = QUESTION_DIR / path
    return path.resolve()


def resolve_repo_dir(value: str | Path | None) -> Path:
    if value is None:
        return DEFAULT_REPO_DIR.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        # Prefer the caller's cwd when a relative path already exists there; otherwise
        # resolve relative paths from the repository root to match the README examples.
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.exists():
            path = cwd_candidate
        else:
            path = REPO_ROOT / path
    return path.resolve()


def resolve_python(repo_dir: Path, requested_python: str | None) -> str:
    if requested_python:
        return requested_python
    venv_python = repo_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def maybe_reexec_with_repo_python(
    repo_dir: Path,
    requested_python: str | None,
    marker: str,
    argv: list[str] | None = None,
) -> None:
    if os.environ.get(marker) == "1":
        return

    # Re-exec once into the repo-local interpreter so every downstream import and
    # extension module comes from the intended Ultralytics environment.
    target_python = Path(resolve_python(repo_dir, requested_python)).expanduser().resolve()
    current_python = Path(sys.executable).expanduser().resolve()
    if target_python == current_python:
        return

    env = os.environ.copy()
    env[marker] = "1"
    os.execvpe(str(target_python), [str(target_python), *(argv or sys.argv)], env)


def ensure_ultralytics_import(repo_dir: Path, package_name: str = "ultralytics"):
    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name != package_name:
            raise

    repo_dir = ensure_ultralytics_repo(repo_dir)
    # Fall back to importing directly from the clone path if the editable install
    # is missing or the script was launched before activating the venv.
    repo_dir_str = str(repo_dir)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)
    importlib.invalidate_caches()

    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name != package_name:
            raise
        raise ModuleNotFoundError(
            "Unable to import 'ultralytics' from the Question 5 environment. "
            "Rerun setup with 'bash question_5_semantic_segmentation/setup_yolo11_osc.sh' from the repo root "
            "or 'bash setup_yolo11_osc.sh' from inside question_5_semantic_segmentation."
        ) from exc


def ensure_ultralytics_repo(repo_dir: Path) -> Path:
    repo_dir = repo_dir.resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory does not exist: {repo_dir}")
    if not (repo_dir / "ultralytics").exists():
        raise FileNotFoundError(
            f"{repo_dir} does not look like an ultralytics clone (missing ultralytics/)."
        )
    return repo_dir


def resolve_model_argument(raw_model: str | None) -> str:
    # With no explicit --model, prefer the fine-tuned alias if it exists; otherwise
    # fall back to the downloaded pretrained checkpoint so the demo still runs.
    if not raw_model:
        if DEFAULT_FINETUNED_MODEL.exists():
            return str(DEFAULT_FINETUNED_MODEL.resolve())
        return str(DEFAULT_PRETRAINED_MODEL.resolve())

    if is_url(raw_model):
        return raw_model

    candidate = Path(raw_model).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    if any(sep in raw_model for sep in ("/", os.sep)):
        # Treat values containing path separators as question-folder-relative
        # paths so README examples work the same from both supported launch dirs.
        return str(resolve_question_path(raw_model))

    question_candidate = QUESTION_DIR / raw_model
    if question_candidate.exists():
        return str(question_candidate.resolve())

    # Otherwise leave the value untouched so Ultralytics can resolve stock model
    # names such as yolo11s-obb.pt from its own model registry or cache.
    return raw_model


# Input/output helpers are shared so batch and single-image inference write a
# consistent folder layout and asset naming scheme.
def resolve_source(value: str) -> tuple[str, Path | None]:
    if is_url(value):
        return value, None

    source_path = resolve_question_path(value)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if source_path.is_file() and source_path.suffix.lower() not in IMAGE_SUFFIXES:
        suffixes = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(
            f"Unsupported source file type for {source_path.name}. Supported suffixes: {suffixes}"
        )

    return str(source_path), source_path


def list_supported_images(directory: Path) -> list[Path]:
    # Recursive search lets nested image folders map cleanly into nested output folders.
    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: str(path),
    )


def per_image_output_dir(source_dir: Path, image_path: Path, output_root: Path) -> Path:
    # Preserve the source directory structure under the output root so it is easy
    # to trace each summary/overlay folder back to its original input image.
    relative_path = image_path.resolve().relative_to(source_dir.resolve())
    return output_root / relative_path.with_suffix("")


def url_output_suffix(source: str) -> str:
    suffix = Path(urlparse(source).path).suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".jpg"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Data bootstrap helpers keep Question 5 self-contained instead of relying on
# ad-hoc manual download/extract steps.
def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Using existing download: {destination}")
        return destination

    # Stream to a temporary .part file first so interrupted downloads do not leave
    # a half-written archive/checkpoint that looks valid on the next run.
    print(f"Downloading {url}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request) as response, temp_path.open("wb") as handle:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"  downloaded {human_bytes(downloaded)} / {human_bytes(total)}",
                        end="\r",
                        flush=True,
                    )
            if total:
                print(" " * 80, end="\r")
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"Saved download to {destination}")
    return destination


def extract_zip(zip_path: Path, destination_dir: Path) -> Path:
    zip_path = zip_path.resolve()
    destination_dir = destination_dir.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    # DOTAv1 ships as a ZIP archive, so extraction is centralized here for reuse
    # by the bootstrap script.
    print(f"Extracting {zip_path} into {destination_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination_dir)
    return destination_dir


def parse_rates(raw_value: str) -> tuple[float, ...]:
    # split_dota accepts multiple scale factors, so parse the CLI string once here.
    rates = tuple(float(part.strip()) for part in raw_value.split(",") if part.strip())
    if not rates:
        raise ValueError("At least one rate must be provided.")
    return rates


def validate_dota_root(path: Path) -> Path:
    path = path.resolve()
    # The bootstrap script expects the standard DOTA train/val image+label layout
    # before attempting any split generation.
    expected = [path / "images" / "train", path / "images" / "val", path / "labels" / "train", path / "labels" / "val"]
    missing = [item for item in expected if not item.exists()]
    if missing:
        joined = ", ".join(str(item) for item in missing)
        raise FileNotFoundError(f"DOTAv1 dataset layout is incomplete under {path}. Missing: {joined}")
    return path


def write_dota_yaml(yaml_path: Path, split_root: Path) -> Path:
    yaml_path = yaml_path.resolve()
    split_root = split_root.resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics training expects a compact YAML manifest pointing at the tiled split dataset.
    lines = [
        "# Auto-generated for Question 5 DOTA YOLO11 OBB training",
        f"path: {split_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for class_id, class_name in DOTA_CLASS_NAMES.items():
        lines.append(f"  {class_id}: {class_name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def summarize_candidates(paths: Iterable[Path]) -> str:
    return ", ".join(str(path.resolve()) for path in paths)
