#!/usr/bin/env python3
"""Shared helpers for the YOLO11 iSAID segmentation workflow."""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
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
DEFAULT_TRAIN_PROJECT_DIR = QUESTION_DIR / "runs" / "segment" / "train"

DATASETS_DIR = QUESTION_DIR / "datasets"
DOWNLOADS_DIR = DATASETS_DIR / "downloads"
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
DEFAULT_RAW_ROOT = RAW_DATASETS_DIR / "isaid"
DEFAULT_IMAGES_DIR = DEFAULT_RAW_ROOT / "images"
DEFAULT_LABELS_DIR = DEFAULT_RAW_ROOT / "labels"
DEFAULT_ANNOTATIONS_DIR = DEFAULT_RAW_ROOT / "annotations"
DEFAULT_TRAIN_JSON = DEFAULT_ANNOTATIONS_DIR / "instances_train.json"
DEFAULT_VAL_JSON = DEFAULT_ANNOTATIONS_DIR / "instances_val.json"
DEFAULT_DATASET_YAML = DATASETS_DIR / "isaid_seg.yaml"
DEFAULT_CONVERTED_DIR = DATASETS_DIR / "converted" / "isaid_yolo"
DEFAULT_DOTA_ZIP = DOWNLOADS_DIR / "DOTAv1.zip"
DEFAULT_DOTA_EXTRACT_DIR = DOWNLOADS_DIR / "DOTAv1"
DEFAULT_ISAID_GDRIVE_DIR = DOWNLOADS_DIR / "isaid_gdrive"

MODELS_DIR = QUESTION_DIR / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
DEFAULT_PRETRAINED_MODEL = PRETRAINED_MODELS_DIR / "yolo11s-seg.pt"
FINETUNED_MODELS_DIR = MODELS_DIR / "isaid_seg"
DEFAULT_FINETUNED_MODEL = FINETUNED_MODELS_DIR / "best.pt"

DEFAULT_YOLO11_SEG_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-seg.pt"
DEFAULT_DOTA_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"
DEFAULT_ISAID_DATASET_PAGE_URL = "https://captain-whu.github.io/iSAID/dataset.html"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUPPORTED_OSC_PYTHON_VERSION = "3.10"

ISAID_CLASS_NAMES = {
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

CANONICAL_ISAID_NAME_TO_ID = {
    re.sub(r"\s+", " ", name.strip().lower()): class_id + 1
    for class_id, name in ISAID_CLASS_NAMES.items()
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


def class_color(class_id) -> tuple[int, int, int]:
    # Cycle through one fixed palette so class colors stay stable across runs.
    """Return the display color for one class id."""
    return _PALETTE[class_id % len(_PALETTE)]


def normalize_name_map(model_names) -> dict[int, str]:
    # Ultralytics may expose class names as either a dict or a list-like object.
    """Normalize the model name map into one dict."""
    if isinstance(model_names, dict):
        return {int(k): str(v) for k, v in model_names.items()}
    return {int(i): str(v) for i, v in enumerate(model_names)}


def parse_keep_classes(raw_value, name_map) -> set[int] | None:
    # An empty filter means the caller wants every class.
    """Parse the requested class filter."""
    if not raw_value:
        return None

    # Accept either numeric ids or case-insensitive class names.
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


def is_url(value) -> bool:
    """Return True when the value looks like a URL."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def resolve_question_path(value) -> Path:
    # Anchor relative paths to this module folder.
    """Resolve a path under this module folder."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = QUESTION_DIR / path
    return path.resolve()


def resolve_repo_dir(value) -> Path:
    """Resolve the upstream repository directory."""
    # Default to the checked-out Ultralytics clone when no override is given.
    if value is None:
        return DEFAULT_REPO_DIR.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        # Prefer an existing cwd-relative path before falling back to repo-relative.
        cwd_candidate = (Path.cwd() / path).resolve()
        if cwd_candidate.exists():
            path = cwd_candidate
        else:
            path = REPO_ROOT / path
    return path.resolve()


def resolve_python(repo_dir, requested_python) -> str:
    """Resolve the Python executable to use."""
    # Prefer the repository-local virtual environment when the caller does not override it.
    if requested_python:
        return requested_python
    venv_python = repo_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def resolve_python_minor_version(python_executable) -> str:
    """Ask one interpreter for its major.minor version."""
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no stderr output"
        raise RuntimeError(
            f"Unable to determine the Python version for {python_executable}.\n"
            f"stderr: {stderr}"
        )
    return completed.stdout.strip()


def ensure_supported_repo_python_version(python_executable) -> str:
    """Reject unsupported interpreters for the OSC segmentation workflow."""
    version = resolve_python_minor_version(python_executable)
    if version != SUPPORTED_OSC_PYTHON_VERSION:
        raise RuntimeError(
            f"YOLO11 segmentation expects Python {SUPPORTED_OSC_PYTHON_VERSION} in the target runtime, "
            f"but {python_executable} resolved to {version}.\n"
            "Rebuild external/ultralytics/.venv with Python 3.10 or pass --python to a Python 3.10 interpreter."
        )
    return version


def maybe_reexec_with_repo_python(
    repo_dir,
    requested_python,
    marker,
    argv=None,
) -> None:
    """Re-run under the repository interpreter when needed."""
    target_python = Path(resolve_python(repo_dir, requested_python)).expanduser().resolve()
    ensure_supported_repo_python_version(target_python)

    # Re-exec once so later imports run under the same interpreter as the repository environment.
    if os.environ.get(marker) == "1":
        return

    current_python = Path(sys.executable).expanduser().resolve()
    if target_python == current_python:
        return

    env = os.environ.copy()
    env[marker] = "1"
    os.execvpe(str(target_python), [str(target_python), *(argv or sys.argv)], env)


def ensure_ultralytics_import(repo_dir, package_name="ultralytics"):
    """Import Ultralytics from the active environment or source checkout."""
    # First try the current interpreter environment as-is.
    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name != package_name:
            raise

    # If the package is missing, add the source checkout to sys.path and try again.
    repo_dir = ensure_ultralytics_repo(repo_dir)
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
            "Unable to import 'ultralytics' from the local segmentation environment. "
            "Run 'bash question_5_semantic_segmentation/setup_yolo11_osc.sh' from the repo root "
            "or 'bash setup_yolo11_osc.sh' from inside question_5_semantic_segmentation."
        ) from exc


def ensure_ultralytics_repo(repo_dir) -> Path:
    # Validate the checkout before any scripts try to import from it.
    """Check that the Ultralytics clone looks valid."""
    repo_dir = repo_dir.resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory does not exist: {repo_dir}")
    if not (repo_dir / "ultralytics").exists():
        raise FileNotFoundError(
            f"{repo_dir} does not look like an ultralytics clone (missing ultralytics/)."
        )
    return repo_dir


def resolve_model_argument(raw_model) -> str:
    """Resolve the model argument into the value YOLO should load."""
    # With no override, prefer the fine-tuned checkpoint and then the pretrained fallback.
    if not raw_model:
        if DEFAULT_FINETUNED_MODEL.exists():
            return str(DEFAULT_FINETUNED_MODEL.resolve())
        return str(DEFAULT_PRETRAINED_MODEL.resolve())

    # URLs pass straight through to the downstream loader.
    if is_url(raw_model):
        return raw_model

    candidate = Path(raw_model).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    # Path-like strings are anchored to this module folder.
    if any(sep in raw_model for sep in ("/", os.sep)):
        return str(resolve_question_path(raw_model))

    # Plain filenames first check this module folder, then fall through unchanged.
    module_candidate = QUESTION_DIR / raw_model
    if module_candidate.exists():
        return str(module_candidate.resolve())

    return raw_model


def resolve_source(value) -> tuple[str, Path | None]:
    # URLs are left untouched and handled in single-image mode.
    """Resolve the input source into a path or URL."""
    # Return URL sources unchanged and mark them as non-path inputs.
    if is_url(value):
        return value, None

    # Expand "~" before deciding how to anchor the path.
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        source_path = candidate.resolve()
    else:
        # Prefer an existing cwd-relative path before falling back to this module folder.
        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            source_path = cwd_candidate
        else:
            source_path = resolve_question_path(value)
    # Stop early if the resolved path does not exist.
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    # Reject non-image files early so later inference code sees only supported inputs.
    if source_path.is_file() and source_path.suffix.lower() not in IMAGE_SUFFIXES:
        suffixes = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(
            f"Unsupported source file type for {source_path.name}. Supported suffixes: {suffixes}"
        )

    return str(source_path), source_path


def list_supported_images(directory) -> list[Path]:
    # Walk recursively so nested image folders are processed in one batch run.
    """List the supported images under the directory."""
    # Return one sorted list of supported files under the directory tree.
    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: str(path),
    )


def per_image_output_dir(source_dir, image_path, output_root) -> Path:
    # Mirror the source tree under the output root without the file suffix.
    """Build the output folder for one image."""
    # Compute the image path relative to the batch source root.
    relative_path = image_path.resolve().relative_to(source_dir.resolve())
    # Reuse that relative path as the output folder name without the extension.
    return output_root / relative_path.with_suffix("")


def url_output_suffix(source) -> str:
    # Preserve a supported URL suffix when one is present.
    """Pick a usable image suffix for a URL source."""
    # Pull the suffix from the URL path only.
    suffix = Path(urlparse(source).path).suffix.lower()
    # Fall back to .jpg when the URL does not end in a supported image suffix.
    return suffix if suffix in IMAGE_SUFFIXES else ".jpg"


def ensure_parent(path) -> Path:
    # Create the parent directory before writing the file.
    """Create the parent directory for a file path."""
    # Create every missing parent directory in one call.
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def human_bytes(num_bytes) -> str:
    # Convert the raw byte count into the first readable unit.
    """Format a byte count in a readable unit."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        # Stop once the value is small enough for the current unit.
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        # Otherwise move to the next larger unit.
        size /= 1024.0
    return f"{num_bytes}B"


def download_file(url, destination, force=False) -> Path:
    # Reuse an existing download unless the caller forces a refresh.
    """Download one file when it is missing."""
    # Resolve the destination and create its parent directory first.
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the existing file when force is not set.
    if destination.exists() and not force:
        print(f"Using existing download: {destination}")
        return destination

    print(f"Downloading {url}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        # Stream into a temporary file so incomplete downloads never look valid.
        with urlopen(request) as response, temp_path.open("wb") as handle:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            downloaded = 0
            while True:
                # Read the next 1 MB chunk from the response.
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                # Append the chunk to the temporary file.
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    # Overwrite one progress line while the transfer is active.
                    print(
                        f"  downloaded {human_bytes(downloaded)} / {human_bytes(total)}",
                        end="\r",
                            flush=True,
                    )
            if total:
                # Clear the progress line after the transfer finishes.
                print(" " * 80, end="\r")
        # Promote the complete temporary file into place atomically.
        temp_path.replace(destination)
    finally:
        # Clean up the temporary file if the download failed partway through.
        if temp_path.exists():
            temp_path.unlink()

    print(f"Saved download to {destination}")
    return destination


def ensure_clean_dir(path) -> Path:
    # Replace any previous directory contents with a new empty directory.
    """Replace the directory with a clean empty one."""
    path = path.resolve()
    # Remove the old tree when it already exists.
    if path.exists():
        shutil.rmtree(path)
    # Recreate the directory in a clean state.
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_zip(zip_path, destination_dir) -> Path:
    # Expand the archive into the requested destination directory.
    """Extract one zip archive."""
    # Resolve both paths before extraction.
    zip_path = zip_path.resolve()
    destination_dir = destination_dir.resolve()
    # Make sure the destination folder exists first.
    destination_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} into {destination_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        # Unpack every file in the archive into the destination tree.
        archive.extractall(destination_dir)
    return destination_dir


def validate_dota_images_root(path) -> Path:
    # DOTA image bundles must provide both train and val image folders.
    """Check that the DOTA image layout is complete."""
    path = path.resolve()
    expected = [path / "images" / "train", path / "images" / "val"]
    # Collect every missing required directory before raising.
    missing = [item for item in expected if not item.exists()]
    if missing:
        joined = ", ".join(str(item) for item in missing)
        raise FileNotFoundError(f"DOTA image layout is incomplete under {path}. Missing: {joined}")
    return path


def find_dota_root(search_root) -> Path:
    # Check the common extraction path first.
    """Find the extracted DOTA root directory."""
    search_root = search_root.resolve()
    direct = search_root / "DOTAv1"
    if direct.exists():
        try:
            # Return the common extraction root when its layout is complete.
            return validate_dota_images_root(direct)
        except FileNotFoundError:
            pass

    # Fall back to a recursive search if the archive unpacked into a different layout.
    candidates = [search_root, *sorted(path for path in search_root.rglob("*") if path.is_dir())]
    for candidate in candidates:
        try:
            # Stop at the first directory that looks like a valid DOTA root.
            return validate_dota_images_root(candidate)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"Unable to find an extracted DOTA root with images/train and images/val under {search_root}"
    )


def fetch_text(url) -> str:
    # Download one text payload with a browser-like user agent.
    """Fetch a text response from a URL."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:
        # Decode the full response body as UTF-8 text.
        return response.read().decode("utf-8")


def parse_isaid_google_drive_links(dataset_page_url) -> dict[str, str]:
    # Pull the train and validation folder links from the public dataset page.
    """Parse the train and val Google Drive links."""
    html = fetch_text(dataset_page_url)
    match = re.search(
        r"iSAID on Google Drive:\s*<a href=\"([^\"]+)\">Training set</a>,\s*<a href=\"([^\"]+)\">Validation set</a>",
        html,
        flags=re.IGNORECASE,
    )
    if not match:
        raise RuntimeError(
            f"Unable to find Google Drive training/validation links on {dataset_page_url}"
        )
    return {"train": match.group(1), "val": match.group(2)}


def gdown_download_folder(
    url,
    output_dir,
    *,
    force=False,
    python_executable=None,
) -> Path:
    # Reuse an existing folder download unless the caller forces a refresh.
    """Download one Google Drive folder when needed."""
    output_dir = output_dir.resolve()
    # Reuse a non-empty download folder unless force is set.
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        print(f"Using existing Google Drive folder download: {output_dir}")
        return output_dir

    # Remove the old folder when force requested a clean download.
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    # Recreate the destination folder before running gdown.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the gdown command that downloads the full folder tree.
    command = [
        python_executable or sys.executable,
        "-m",
        "gdown",
        "--folder",
        "--remaining-ok",
        url,
        "-O",
        str(output_dir),
    ]
    # Delegate Google Drive folder handling to gdown instead of reimplementing it here.
    print(f"Downloading Google Drive folder into {output_dir}")
    # Let gdown handle the transfer and fail loudly if it returns an error.
    subprocess.run(command, check=True)
    return output_dir


def find_annotation_json(search_root, split) -> Path:
    # Gather every JSON file first so the later selection logic can be explicit.
    """Find the annotation JSON for the requested split."""
    search_root = search_root.resolve()
    candidates = sorted(search_root.rglob("*.json"))
    # Stop immediately when the folder contains no JSON files at all.
    if not candidates:
        raise FileNotFoundError(f"No JSON annotations were found under {search_root}")

    # Prefer filenames that mention the requested split and are not test annotations.
    split_matches = [
        path
        for path in candidates
        if split.lower() in path.name.lower() and "test" not in path.name.lower()
    ]
    if len(split_matches) == 1:
        return split_matches[0]
    if len(split_matches) > 1:
        # Prefer the most instance-like filename when several matches exist.
        preferred = [
            path
            for path in split_matches
            if "instance" in path.name.lower() or "isaid" in path.name.lower()
        ]
        if len(preferred) == 1:
            return preferred[0]
        raise FileNotFoundError(
            f"Multiple candidate {split} annotation JSON files were found under {search_root}: "
            f"{summarize_candidates(split_matches)}"
        )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Unable to determine the {split} annotation JSON under {search_root}. Found: "
        f"{summarize_candidates(candidates)}"
    )


def ensure_symlink_or_copy(source, destination) -> Path:
    # Resolve the source once so later symlink checks compare canonical paths.
    """Create a symlink or copy to the destination."""
    source = source.resolve()
    destination = destination.expanduser()
    # Resolve relative destinations against the current working directory.
    if not destination.is_absolute():
        destination = destination.resolve()

    if destination.is_symlink():
        # Leave an existing correct symlink in place.
        try:
            link_target = destination.readlink()
            if not link_target.is_absolute():
                link_target = (destination.parent / link_target).resolve()
            else:
                link_target = link_target.resolve()
        except OSError:
            link_target = None

        if link_target == source:
            return destination
        # Remove an incorrect symlink before replacing it.
        destination.unlink(missing_ok=True)
    elif destination.exists():
        # Remove an existing real file or directory before replacing it.
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Prefer a symlink so the raw layout does not duplicate large image trees.
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        # Fall back to a real copy on filesystems that do not allow symlinks.
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    return destination


def copy_file(source, destination) -> Path:
    # Copy one file after creating its parent directory.
    """Copy one file to the destination."""
    source = source.resolve()
    destination = destination.resolve()
    # Create the destination folder first.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Copy the file metadata and contents together.
    shutil.copy2(source, destination)
    return destination


def canonicalize_isaid_name(name) -> str:
    # Normalize category names so different punctuation still maps to one class id.
    """Normalize an iSAID class name."""
    value = re.sub(r"[^a-z0-9]+", " ", name.strip().lower())
    return re.sub(r"\s+", " ", value).strip()


def find_matching_image_file(image_dir, raw_name) -> Path:
    # Try the JSON filename first.
    """Find the image file that matches the annotation name."""
    image_dir = image_dir.resolve()
    candidate = image_dir / raw_name
    # Return the exact filename match when it exists.
    if candidate.exists():
        return candidate

    # Then try the same stem with the common image suffixes used by iSAID/DOTA.
    stem = Path(raw_name).stem
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Unable to find an RGB image for {raw_name} under {image_dir}")


def normalize_isaid_annotation_json(source, destination, image_dir=None) -> Path:
    # Load the COCO annotation file before remapping any category ids.
    """Normalize one iSAID annotation JSON file."""
    source = source.resolve()
    destination = destination.resolve()
    # Read the JSON payload into memory so it can be rewritten in place.
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    # Pull out the two sections that need to be normalized.
    categories = data.get("categories")
    annotations = data.get("annotations")
    if not isinstance(categories, list) or not isinstance(annotations, list):
        raise ValueError(f"{source} is missing COCO categories or annotations")

    old_to_new: dict[int, int] = {}
    for category in categories:
        # Map each original category id onto the fixed contiguous iSAID id space.
        old_id = int(category["id"])
        canonical_name = canonicalize_isaid_name(str(category["name"]))
        new_id = CANONICAL_ISAID_NAME_TO_ID.get(canonical_name)
        if new_id is None:
            raise ValueError(
                f"Unknown iSAID category '{category['name']}' in {source}. "
                f"Expected one of: {sorted(CANONICAL_ISAID_NAME_TO_ID)}"
            )
        old_to_new[old_id] = new_id

    for annotation in annotations:
        # Rewrite every annotation to the normalized category id.
        annotation["category_id"] = old_to_new[int(annotation["category_id"])]

    if image_dir is not None:
        from PIL import Image

        image_dir = image_dir.resolve()
        for image in data.get("images", []):
            # Replace the filename with the actual RGB image name used on disk.
            matched_image = find_matching_image_file(image_dir, str(image["file_name"]))
            image["file_name"] = matched_image.name
            if "width" not in image or "height" not in image:
                # Fill in missing image size metadata directly from the file.
                with Image.open(matched_image) as handle:
                    image["width"], image["height"] = handle.size

    # Rewrite the category table to the normalized training ids and names.
    data["categories"] = [
        {"id": class_id + 1, "name": class_name}
        for class_id, class_name in sorted(ISAID_CLASS_NAMES.items())
    ]

    # Create the destination folder before writing the normalized JSON.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write the rewritten JSON payload to the requested destination.
    destination.write_text(json.dumps(data), encoding="utf-8")
    return destination


def prepare_isaid_raw_layout(
    *,
    raw_root,
    dota_root,
    train_json_source,
    val_json_source,
) -> Path:
    # Link or copy the RGB image folders into the normalized raw layout.
    """Build the normalized raw iSAID layout."""
    raw_root = raw_root.resolve()
    # Put the DOTA train images under raw_root/images/train.
    train_image_dir = ensure_symlink_or_copy(dota_root / "images" / "train", raw_root / "images" / "train")
    # Put the DOTA val images under raw_root/images/val.
    val_image_dir = ensure_symlink_or_copy(dota_root / "images" / "val", raw_root / "images" / "val")
    # Normalize both annotation files into the same raw layout.
    normalize_isaid_annotation_json(
        train_json_source,
        raw_root / "annotations" / "instances_train.json",
        image_dir=train_image_dir,
    )
    normalize_isaid_annotation_json(
        val_json_source,
        raw_root / "annotations" / "instances_val.json",
        image_dir=val_image_dir,
    )
    return raw_root


def validate_isaid_root(path) -> Path:
    # The normalized layout must contain both image splits and both annotation files.
    """Check that the normalized iSAID layout is complete."""
    path = path.resolve()
    expected = [
        path / "images" / "train",
        path / "images" / "val",
        path / "annotations" / "instances_train.json",
        path / "annotations" / "instances_val.json",
    ]
    # Collect any missing required paths before raising one combined error.
    missing = [item for item in expected if not item.exists()]
    if missing:
        joined = ", ".join(str(item) for item in missing)
        raise FileNotFoundError(
            "iSAID dataset layout is incomplete. Expected the normalized drop-in layout under "
            f"{path}. Missing: {joined}"
        )
    return path


def load_coco_categories(annotation_path) -> dict[int, str]:
    # Load the category table from one COCO annotation file.
    """Load the COCO categories from one annotation file."""
    annotation_path = annotation_path.resolve()
    # Read the full annotation JSON once.
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    categories = data.get("categories")
    # The category table must exist and contain at least one class.
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"No COCO categories were found in {annotation_path}")

    # Sort categories by id before validating the numeric sequence.
    sorted_categories = sorted(categories, key=lambda item: int(item["id"]))
    expected_ids = list(range(1, len(sorted_categories) + 1))
    actual_ids = [int(item["id"]) for item in sorted_categories]
    if actual_ids != expected_ids:
        raise ValueError(
            f"Category ids in {annotation_path} must be contiguous and start at 1. "
            f"Found {actual_ids}."
        )

    return {int(item["id"]) - 1: str(item["name"]) for item in sorted_categories}


def load_and_validate_category_maps(train_json, val_json) -> dict[int, str]:
    # Require train and validation splits to expose the same class map.
    """Load and compare the train and val category maps."""
    # Load the train split category map.
    train_map = load_coco_categories(train_json)
    # Load the validation split category map.
    val_map = load_coco_categories(val_json)
    # Stop if the two splits disagree on the class id mapping.
    if train_map != val_map:
        raise ValueError(
            f"Train/val category maps differ. Train: {train_map}; Val: {val_map}"
        )
    return train_map


def write_dataset_yaml(yaml_path, dataset_root, names) -> Path:
    # Write the compact dataset description that Ultralytics expects.
    """Write the dataset YAML file."""
    yaml_path = yaml_path.resolve()
    dataset_root = dataset_root.resolve()
    # Create the parent directory before writing the YAML file.
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Auto-generated for YOLO11 iSAID segmentation training",
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        "test:",
        "names:",
    ]
    for class_id, class_name in names.items():
        # Write one numeric class id to class name mapping per line.
        lines.append(f"  {class_id}: {class_name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def replace_label_tree(source_dir, destination_dir) -> None:
    # Replace the destination tree with the newly generated label files.
    """Replace the label tree with the new files."""
    source_dir = source_dir.resolve()
    destination_dir = destination_dir.resolve()
    # Remove any older label tree first.
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    # Recreate the destination root before moving files into it.
    destination_dir.mkdir(parents=True, exist_ok=True)

    for source_file in sorted(source_dir.rglob("*.txt")):
        # Preserve the relative split/subfolder layout while moving the labels.
        destination_file = destination_dir / source_file.relative_to(source_dir)
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_file), str(destination_file))


def summarize_candidates(paths) -> str:
    # Join candidate paths into one readable error string.
    """Join candidate paths into one string."""
    # Resolve each path so the final message is unambiguous.
    return ", ".join(str(path.resolve()) for path in paths)
