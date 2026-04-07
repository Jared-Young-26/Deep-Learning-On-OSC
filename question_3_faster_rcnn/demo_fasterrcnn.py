#!/usr/bin/env python3
"""Run inference with the trzy/FasterRCNN project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


QUESTION_DIR = Path(__file__).resolve().parent
REPO_ROOT = QUESTION_DIR.parent
DEFAULT_REPO_DIR = REPO_ROOT / "external" / "FasterRCNN"
DEFAULT_OUTPUT_DIR = QUESTION_DIR / "outputs"
DEFAULT_INPUTS_DIR = QUESTION_DIR / "inputs"
DEFAULT_DOWNLOADED_INPUT_DIR = DEFAULT_INPUTS_DIR / "downloaded"
DEFAULT_IMAGE = "http://trzy.org/files/fasterrcnn/gary.jpg"
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


DEFAULT_WEIGHTS = {
    "pytorch": "fasterrcnn_pytorch_resnet50.pth",
    "tf2": "fasterrcnn_tf2.h5",
}
SUPPORTED_OSC_PYTHON_VERSION = "3.9.18"
OSC_GPU_BATCH_LAUNCHER_NAME = "osc_gpu_batch.sh"
TF2_ENV_OVERRIDES = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "TF_CPP_MIN_LOG_LEVEL": "2",
}

# The PyTorch batch shim loads the model only once, then walks the entire input
# selection. It also restores a compatible getsize() helper for newer Pillow
# builds before the upstream visualizer imports it.
INLINE_PYTORCH_SCRIPT = r"""
import argparse
import json
from pathlib import Path

import torch as t
from PIL import ImageFont

from pytorch.FasterRCNN import state, visualize
from pytorch.FasterRCNN.datasets import voc
from pytorch.FasterRCNN.datasets import image as image_utils
from pytorch.FasterRCNN.models import resnet, vgg16, vgg16_torch
from pytorch.FasterRCNN.models.faster_rcnn import FasterRCNNModel


def compat_getsize(self, text, *args, **kwargs):
    if hasattr(self, "getbbox"):
        left, top, right, bottom = self.getbbox(text, *args, **kwargs)
        return (max(1, right - left), max(1, bottom - top))
    mask = self.getmask(text, *args, **kwargs)
    return mask.size


for class_name in ("FreeTypeFont", "ImageFont"):
    font_class = getattr(ImageFont, class_name, None)
    if font_class is not None and not hasattr(font_class, "getsize"):
        font_class.getsize = compat_getsize


def build_backbone(name):
    valid_backbones = {"vgg16", "vgg16-torch", "resnet50", "resnet101", "resnet152"}
    if name not in valid_backbones:
        raise ValueError("--backbone must be one of: " + ", ".join(sorted(valid_backbones)))

    if name == "vgg16":
        return vgg16.VGG16Backbone(dropout_probability=0.0)
    if name == "vgg16-torch":
        return vgg16_torch.VGG16Backbone(dropout_probability=0.0)
    if name == "resnet50":
        return resnet.ResNetBackbone(architecture=resnet.Architecture.ResNet50)
    if name == "resnet101":
        return resnet.ResNetBackbone(architecture=resnet.Architecture.ResNet101)
    return resnet.ResNetBackbone(architecture=resnet.Architecture.ResNet152)


def predict_job(model, input_path, output_path, show_image):
    image_data, image_obj, _, _ = image_utils.load_image(
        url=input_path,
        preprocessing=model.backbone.image_preprocessing_params,
        min_dimension_pixels=600,
    )
    with t.no_grad():
        image_tensor = t.from_numpy(image_data).unsqueeze(dim=0).cuda()
        scored_boxes_by_class_index = model.predict(image_data=image_tensor, score_threshold=0.7)
    visualize.show_detections(
        output_path=output_path,
        show_image=show_image,
        image=image_obj,
        scored_boxes_by_class_index=scored_boxes_by_class_index,
        class_index_to_name=voc.Dataset.class_index_to_name,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--backbone", required=True)
parser.add_argument("--mode", choices=("to-file", "viewer"), default="to-file")
parser.add_argument("--jobs-json", required=True)
args = parser.parse_args()

jobs = json.loads(args.jobs_json)
backbone = build_backbone(args.backbone)
model = FasterRCNNModel(
    num_classes=voc.Dataset.num_classes,
    backbone=backbone,
    allow_edge_proposals=True,
).cuda()
state.load(model=model, filepath=args.weights)
model.eval()

show_image = args.mode == "viewer"
total = len(jobs)

for index, job in enumerate(jobs, start=1):
    input_path = job["input"]
    output_path = job["output"]
    if total > 1:
        print(f"[{index}/{total}] {input_path}", flush=True)
        if output_path is not None:
            print(f"      -> {output_path}", flush=True)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    predict_job(
        model=model,
        input_path=input_path,
        output_path=output_path,
        show_image=show_image,
    )

    if output_path is not None:
        if total == 1:
            print(f"Saved demo output to: {output_path}", flush=True)
        else:
            print(f"[{index}/{total}] Saved demo output to: {output_path}", flush=True)
"""

# The upstream TF2 path expects a different entrypoint shape than this file uses.
# This embedded script keeps the TF2 invocation self-contained and preserves H5
# loading on current Python and Pillow environments.
INLINE_TF2_SCRIPT = r"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from tf2.FasterRCNN import visualize
from tf2.FasterRCNN.datasets import voc
from tf2.FasterRCNN.datasets.image import load_image
from tf2.FasterRCNN.models import anchors, faster_rcnn


parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--output", default="")
parser.add_argument("--show-image", action="store_true")
parser.add_argument("--score-threshold", type=float, default=0.7)
args = parser.parse_args()

warnings.filterwarnings(
    "ignore",
    message=r"Do not pass an `input_shape`/`input_dim` argument to a layer.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`build\(\)` was called on layer .*",
    category=UserWarning,
)

# The upstream text-drawing helper is fragile on some Pillow versions, so this
# compatibility layer installs a simpler text renderer before visualization.
def compat_draw_text(image, text, position, color, scale=1.0, offset_lines=0):
    font = ImageFont.load_default()
    left, top, right, bottom = font.getbbox(text)
    width = max(1, right - left)
    height = max(1, bottom - top)
    text_image = Image.new(mode="RGBA", size=(width, height), color=(0, 0, 0, 0))
    ctx = ImageDraw.Draw(text_image)
    ctx.text(xy=(0, 0), text=text, font=font, fill=color)
    scaled = text_image.resize(
        (
            max(1, round(text_image.width * scale)),
            max(1, round(text_image.height * scale)),
        )
    )
    paste_position = (
        round(position[0]),
        round(position[1] + offset_lines * scaled.height),
    )
    image.paste(im=scaled, box=paste_position, mask=scaled)


visualize._draw_text = compat_draw_text


def compat_show_detections(output_path, show_image, image, scored_boxes_by_class_index, class_index_to_name):
    ctx = ImageDraw.Draw(image, mode="RGBA")
    for class_index, scored_boxes in scored_boxes_by_class_index.items():
        for i in range(scored_boxes.shape[0]):
            scored_box = scored_boxes[i, :]
            class_name = class_index_to_name[class_index]
            text = "%s %1.2f" % (class_name, scored_box[4])
            color = visualize._class_to_color(class_index=class_index)
            visualize._draw_rectangle(ctx=ctx, corners=scored_box[0:4], color=color, thickness=2)
            visualize._draw_text(
                image=image,
                text=text,
                position=(scored_box[1], scored_box[0]),
                color=color,
                scale=1.5,
                offset_lines=-1,
            )

    if show_image:
        image.show()
    if output_path is not None:
        image.save(output_path)


visualize.show_detections = compat_show_detections

# Preprocess the input image exactly the way the TF2 Faster R-CNN model expects:
# image tensor, anchor map, and anchor-validity mask.
image_data, image, _, _ = load_image(url=args.image, min_dimension_pixels=600)
anchor_map, anchor_valid_map = anchors.generate_anchor_maps(
    image_shape=image_data.shape,
    feature_pixels=16,
)
image_batch = np.expand_dims(image_data, axis=0)
anchor_batch = np.expand_dims(anchor_map, axis=0)
anchor_valid_batch = np.expand_dims(anchor_valid_map, axis=0)

model = faster_rcnn.FasterRCNNModel(
    num_classes=voc.Dataset.num_classes,
    allow_edge_proposals=True,
    custom_roi_pool=False,
    activate_class_outputs=True,
    l2=0.5 * 5e-4,
    dropout_probability=0.0,
)

# Build variables with a real inference-shaped input before loading H5 weights.
# This is necessary because the weights file references named variables that do
# not exist until the model has seen one concrete input shape.
model(
    [
        tf.convert_to_tensor(image_batch),
        tf.convert_to_tensor(anchor_batch),
        tf.convert_to_tensor(anchor_valid_batch),
    ],
    training=False,
)
model.load_weights(filepath=args.weights, by_name=True)

# Run one forward pass, then convert raw detector outputs into scored boxes
# grouped by class so the visualization helper can draw them.
outputs = model(
    [
        tf.convert_to_tensor(image_batch),
        tf.convert_to_tensor(anchor_batch),
        tf.convert_to_tensor(anchor_valid_batch),
    ],
    training=False,
)
_, _, detector_classes, detector_box_deltas, proposals, _, _, _, _ = outputs
scored_boxes_by_class_index = model._predictions_to_scored_boxes(
    input_image=image_batch,
    classes=detector_classes.numpy(),
    box_deltas=detector_box_deltas.numpy(),
    proposals=proposals.numpy(),
    score_threshold=args.score_threshold,
)

output_path = None
if args.output:
    output_path = str(Path(args.output).resolve())
visualize.show_detections(
    output_path=output_path,
    show_image=args.show_image,
    image=image,
    scored_boxes_by_class_index=scored_boxes_by_class_index,
    class_index_to_name=voc.Dataset.class_index_to_name,
)
"""


@dataclass(frozen=True)
class ImageJob:
    display_input: str
    local_image: Path
    output_relative: Path


# The selection object lets the rest of the file handle one image and a whole
# directory of images through the same execution loop.
@dataclass(frozen=True)
class InputSelection:
    jobs: list[ImageJob]
    source_label: str
    is_directory: bool


# These helpers choose the safest runtime path from the machine and weight names.
def is_url(value) -> bool:
    """Return True when the value looks like a URL."""
    return value.startswith("http://") or value.startswith("https://")


def infer_backbone(weights) -> str | None:
    # Some upstream PyTorch checkpoints require the backbone name separately,
    # so infer it from the filename when the user does not pass --backbone.
    """Infer the backbone name from the weight path."""
    token_map = {
        "vgg16-torch": "vgg16-torch",
        "vgg16": "vgg16",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
        "resnet152": "resnet152",
    }
    lowered = weights.lower()
    for token, backbone in token_map.items():
        if token in lowered:
            return backbone
    return None


def infer_default_framework() -> str:
    # Resolve the practical runtime from the target environment instead of the host
    # platform so one default command works on both OSC GPU and CPU nodes.
    """Pick the default framework mode for this machine."""
    return "auto"


def resolve_python(repo_dir, requested) -> str:
    """Resolve the Python executable to use."""
    # Prefer the repository-local virtual environment when it exists.
    if requested:
        return requested

    venv_python = repo_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_python_probe(
    python_bin,
    probe,
    env_overrides=None,
) -> subprocess.CompletedProcess[str]:
    """Run a short Python probe in the target interpreter."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [python_bin, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def resolve_target_python_version(python_bin) -> str:
    """Ask one target interpreter for its major.minor version."""
    result = run_python_probe(
        python_bin,
        "import sys; print(sys.version.split()[0])",
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "no stderr output"
        raise RuntimeError(
            f"Unable to determine the Python version for {python_bin}.\n"
            f"stderr: {stderr}"
        )
    return result.stdout.strip()


def ensure_supported_runtime_python(python_bin) -> str:
    """Reject unsupported target interpreters before probing frameworks."""
    version = resolve_target_python_version(python_bin)
    if version != SUPPORTED_OSC_PYTHON_VERSION:
        details = [
            f"FasterRCNN expects Python {SUPPORTED_OSC_PYTHON_VERSION} in the target runtime, "
            f"but {python_bin} resolved to {version}.",
        ]
        if version.startswith("3.12"):
            details.append(
                "Python 3.12 remains unsupported here, and this workflow now requires exactly Python 3.9.18."
            )
        details.append(
            "Rebuild external/FasterRCNN/.venv with Python 3.9.18 or pass --python to a Python 3.9.18 interpreter."
        )
        raise RuntimeError("\n".join(details))
    return version


def pytorch_import_available(python_bin) -> bool:
    """Check whether PyTorch imports in the target interpreter."""
    result = run_python_probe(python_bin, "import torch")
    return result.returncode == 0


def pytorch_cuda_available(python_bin) -> bool:
    # The upstream PyTorch path is CUDA-only, so check that before launching a run
    # that would otherwise fail after all path resolution is already done.
    """Check whether PyTorch reports CUDA support."""
    probe = (
        "import torch; "
        "print('1' if getattr(torch.cuda, 'is_available', lambda: False)() else '0')"
    )
    result = run_python_probe(python_bin, probe)
    return result.returncode == 0 and result.stdout.strip() == "1"


def tf2_runtime_available(python_bin) -> bool:
    """Check whether the TF2 fallback runtime imports cleanly."""
    probe = (
        "import matplotlib.pyplot; "
        "import numpy; "
        "import tensorflow"
    )
    result = run_python_probe(
        python_bin,
        probe,
        env_overrides=TF2_ENV_OVERRIDES,
    )
    return result.returncode == 0


def osc_gpu_batch_example() -> str:
    """Build one OSC GPU launcher example command."""
    return (
        f"bash {OSC_GPU_BATCH_LAUNCHER_NAME} --account <OSC_ACCOUNT> --time 01:00:00 -- "
        "python3.9 question_3_faster_rcnn/demo_fasterrcnn.py --framework pytorch --mode to-file"
    )


def resolve_framework(requested_framework, python_bin, repo_dir) -> str:
    """Resolve the implementation to run from the prepared environment."""
    if requested_framework == "pytorch":
        if not pytorch_import_available(python_bin):
            raise RuntimeError(
                f"PyTorch is not available in {python_bin}.\n"
                "Rerun `bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh` "
                "to install the FasterRCNN runtime."
            )
        if not pytorch_cuda_available(python_bin):
            raise RuntimeError(
                "The upstream PyTorch FasterRCNN implementation is CUDA-only and "
                f"cannot run in {python_bin}.\n"
                "On OSC, request a GPU node first instead of running the PyTorch path on a login or CPU-only node.\n"
                f"Example: {osc_gpu_batch_example()}\n"
                "Use `--framework tf2` if TensorFlow is installed, rerun "
                "`INSTALL_TF2=1 bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh`, "
                "or run the PyTorch path on a CUDA-enabled system."
            )
        return "pytorch"

    if requested_framework == "tf2":
        if not tf2_runtime_available(python_bin):
            raise RuntimeError(
                f"The TF2 fallback runtime is incomplete or inconsistent in {python_bin}.\n"
                "It must import numpy, matplotlib.pyplot, and tensorflow together.\n"
                f"Remove {repo_dir / '.venv'} and rerun "
                "`bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh`, or use "
                "`--framework pytorch` on a CUDA-enabled system."
            )
        return "tf2"

    if pytorch_import_available(python_bin) and pytorch_cuda_available(python_bin):
        return "pytorch"
    if tf2_runtime_available(python_bin):
        return "tf2"

    raise RuntimeError(
        "Auto framework selection could not find a usable FasterRCNN runtime in "
        f"{python_bin}.\n"
        "The upstream PyTorch path requires CUDA, and the TF2 fallback must "
        "import numpy, matplotlib.pyplot, and tensorflow together.\n"
        f"If you want the PyTorch path on OSC, request the GPU node first. Example: {osc_gpu_batch_example()}\n"
        f"Remove {repo_dir / '.venv'} and rerun "
        "`bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh`, or run on a "
        "CUDA-enabled system."
    )


# Input resolution supports an explicit local file, a whole input directory,
# or a single remote image URL.
def resolve_weights(repo_dir, weights) -> str:
    """Resolve weights."""
    # Leave remote weight URLs unchanged.
    if is_url(weights):
        return weights

    # Expand "~" and anchor relative paths to the FasterRCNN repo clone.
    candidate = Path(weights).expanduser()
    if not candidate.is_absolute():
        candidate = repo_dir / candidate

    # Stop before launching inference if the checkpoint file is missing.
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find weights file: {candidate}\n"
            "Run `bash download_models_fasterrcnn.sh` from question_3_faster_rcnn "
            "or pass an explicit --weights path/URL."
        )
    return str(candidate.resolve())


def remote_cache_path(image_url) -> Path:
    # Cache remote images under a deterministic hashed filename so repeated runs
    # reuse the same downloaded file instead of downloading it again.
    """Build the cache path for a remote image."""
    # Pull the original filename pieces from the URL path.
    parsed = urlparse(image_url)
    filename = Path(unquote(parsed.path)).name or "input_image.jpg"
    stem = Path(filename).stem or "input_image"
    suffix = Path(filename).suffix or ".jpg"
    # Add a short URL hash so different URLs with the same filename do not collide.
    cache_name = (
        f"{stem}-{hashlib.sha256(image_url.encode('utf-8')).hexdigest()[:12]}{suffix}"
    )
    return DEFAULT_DOWNLOADED_INPUT_DIR / cache_name


def cache_remote_image(image_url, dry_run=False) -> Path:
    """Download and cache one remote image when needed."""
    cache_path = remote_cache_path(image_url)
    # In dry-run mode, only report the cache path that would be used.
    if dry_run:
        return cache_path.resolve()

    # Keep downloaded examples under inputs/downloaded so they are visible in the
    # project tree but still separate from the manually curated input images.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the cached file when it already exists and is non-empty.
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.resolve()

    # Download the remote image with curl so this file can capture any error text.
    curl_cmd = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        image_url,
        "--output",
        str(cache_path),
    ]
    result = subprocess.run(curl_cmd, check=False, capture_output=True, text=True)
    # Raise one script-level error when curl fails.
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to download input image URL: {image_url}\n"
            f"curl error: {result.stderr.strip() or result.stdout.strip() or result.returncode}\n"
            "Pass a local image path with --image to avoid network lookup."
        )
    return cache_path.resolve()


def resolve_local_input(repo_dir, image) -> Path:
    """Resolve a local input path."""
    candidate = Path(image).expanduser()
    search_paths: list[Path] = []
    if candidate.is_absolute():
        # Absolute paths need no extra search roots.
        search_paths.append(candidate)
    else:
        # Search relative to the current shell directory, then this directory,
        # then the upstream repository.
        search_paths.extend(
            [
                Path.cwd() / candidate,
                QUESTION_DIR / candidate,
                repo_dir / candidate,
            ]
        )

    checked: list[Path] = []
    seen: set[Path] = set()
    for path in search_paths:
        # Resolve each candidate path before comparing or returning it.
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        checked.append(resolved)
        if resolved.exists():
            return resolved

    # If nothing matched, show every path that was checked.
    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        f"Could not find input path: {image}\nChecked:\n{checked_text}"
    )


def discover_directory_images(input_root) -> list[Path]:
    """Find supported images under the input directory."""
    skip_root = (
        DEFAULT_DOWNLOADED_INPUT_DIR.resolve()
        if input_root.resolve() == DEFAULT_INPUTS_DIR.resolve()
        else None
    )
    # If the user points at the whole inputs/ directory, skip the downloaded-cache
    # subtree so batch runs only process the examples they intentionally collected.
    images = [
        path
        for path in sorted(input_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and (skip_root is None or skip_root not in path.resolve().parents)
    ]
    # Stop if the directory tree did not yield any supported images.
    if not images:
        raise FileNotFoundError(
            "No supported image files were found under directory:\n"
            f"  {input_root}\n"
            f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}"
        )
    return images


def resolve_input_selection(
    repo_dir,
    image,
    dry_run=False,
) -> InputSelection:
    # Convert the user-facing --image argument into one normalized list of jobs.
    # Everything downstream can then treat single-image and batch mode uniformly.
    """Turn the user input into one normalized job selection."""
    if is_url(image):
        # Download the URL into the local cache and build a one-job selection.
        local_image = cache_remote_image(image, dry_run=dry_run)
        parsed = urlparse(image)
        output_name = Path(unquote(parsed.path)).name or local_image.name
        output_path = Path(output_name)
        # Make sure the output name still has an image suffix.
        if not output_path.suffix:
            output_path = output_path.with_suffix(local_image.suffix or ".jpg")
        return InputSelection(
            jobs=[
                ImageJob(
                    display_input=image,
                    local_image=local_image,
                    output_relative=Path(output_path.name),
                )
            ],
            source_label=image,
            is_directory=False,
        )

    input_path = resolve_local_input(repo_dir, image)
    if input_path.is_dir():
        # Expand a directory input into one ImageJob per discovered image.
        images = discover_directory_images(input_path)
        jobs = [
            ImageJob(
                display_input=str(path),
                local_image=path,
                output_relative=path.relative_to(input_path),
            )
            for path in images
        ]
        return InputSelection(
            jobs=jobs,
            source_label=str(input_path),
            is_directory=True,
        )

    # Otherwise build a one-job selection for the single local file.
    return InputSelection(
        jobs=[
            ImageJob(
                display_input=str(input_path),
                local_image=input_path,
                output_relative=Path(input_path.name),
            )
        ],
        source_label=str(input_path),
        is_directory=False,
    )


def resolve_single_output_path(job, raw_output) -> Path:
    # For single-image runs, accept either an explicit filename or a directory and
    # fill in the actual output filename from the input image name when needed.
    """Resolve the output path for a single-image run."""
    if raw_output is None:
        return (DEFAULT_OUTPUT_DIR / job.output_relative.name).resolve()

    candidate = Path(raw_output).expanduser()
    # Treat an existing directory path as an output folder.
    if candidate.exists() and candidate.is_dir():
        return (candidate / job.output_relative.name).resolve()
    # Treat a suffix-less path as a directory-like destination too.
    if candidate.suffix == "":
        return (candidate / job.output_relative.name).resolve()
    # Otherwise treat the value as an explicit file path.
    return candidate.resolve()


def resolve_directory_output_root(raw_output) -> Path:
    # Directory runs must resolve to one output root because each input image keeps
    # its relative name underneath that root.
    """Resolve the output root for a directory run."""
    if raw_output is None:
        return DEFAULT_OUTPUT_DIR.resolve()

    candidate = Path(raw_output).expanduser()
    # Reject existing files as output roots for batch mode.
    if candidate.exists() and candidate.is_file():
        raise ValueError(
            "When --image is a directory, --output must be a directory path, "
            f"not a file: {candidate}"
        )
    # Reject suffix-looking paths that do not exist yet.
    if candidate.suffix and not candidate.exists():
        raise ValueError(
            "When --image is a directory, --output must be a directory path. "
            f"Received: {candidate}"
        )
    return candidate.resolve()


def resolve_output_paths(
    selection,
    raw_output,
    mode,
) -> tuple[list[Path | None], str | None]:
    """Resolve the output path for each job."""
    # Viewer mode leaves output handling to the upstream GUI/image viewer path.
    # Viewer mode does not write file outputs managed by this script.
    if mode != "to-file":
        return [None] * len(selection.jobs), None

    if selection.is_directory:
        # Batch mode maps each job's relative path under one output root.
        output_root = resolve_directory_output_root(raw_output)
        return (
            [(output_root / job.output_relative).resolve() for job in selection.jobs],
            str(output_root),
        )

    # Single-image mode resolves exactly one output path.
    output_path = resolve_single_output_path(selection.jobs[0], raw_output)
    return ([output_path], str(output_path))


# The public CLI stays small while this file hides the upstream command details.
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Simple inference runner for the trzy/FasterRCNN project."
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help=(
            "Path to your local clone of https://github.com/trzy/FasterRCNN. "
            "Defaults to <repo>/external/FasterRCNN."
        ),
    )
    parser.add_argument(
        "--framework",
        choices=("auto", "pytorch", "tf2"),
        default=infer_default_framework(),
        help=(
            "Choose which implementation to run. Defaults to auto, which picks "
            "PyTorch on CUDA nodes and TF2 otherwise."
        ),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path/URL to trained weights. If omitted, a framework default is used.",
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=(
            "Input image path, image URL, or directory of images. "
            "Directory input is processed recursively."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("to-file", "viewer"),
        default="to-file",
        help="Use to-file on headless OSC nodes; viewer opens an image window.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Single image: output file path or output directory. "
            "Directory input: output root directory. "
            "If omitted, outputs are written under this question folder's outputs/."
        ),
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="Optional override for --backbone (PyTorch only).",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter to use. Defaults to <repo>/.venv/bin/python if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command(s) and exit.",
    )
    parser.add_argument(
        "--verbose-command",
        action="store_true",
        help="Print the full underlying command for each image.",
    )
    return parser


def format_run_summary(
    requested_framework,
    framework,
    weights,
    mode,
    selection,
    output_label,
) -> str:
    # Print a compact summary before execution so the high-level plan is visible
    # without reading the full subprocess command.
    """Build the short run summary text."""
    if selection.is_directory:
        # Describe the batch run in a compact multi-line block.
        lines = [
            "Running Faster R-CNN demo batch",
            f"  framework: {framework}",
            f"  mode: {mode}",
            f"  input root: {selection.source_label}",
            f"  images: {len(selection.jobs)}",
            f"  weights: {weights}",
        ]
        if requested_framework != framework:
            lines.insert(2, f"  requested framework: {requested_framework}")
        if output_label:
            lines.append(f"  output root: {output_label}")
        return "\n".join(lines)

    # Otherwise describe the single-image run.
    lines = [
        "Running Faster R-CNN demo",
        f"  framework: {framework}",
        f"  mode: {mode}",
        f"  image: {selection.jobs[0].display_input}",
        f"  weights: {weights}",
    ]
    if requested_framework != framework:
        lines.insert(2, f"  requested framework: {requested_framework}")
    if output_label:
        lines.append(f"  output: {output_label}")
    return "\n".join(lines)


# Command construction translates one stable local interface into the selected
# upstream Faster R-CNN implementation.
def build_command(
    framework,
    python_bin,
    weights,
    backbone,
    job,
    mode,
    output_path,
) -> list[str]:
    """Build the upstream inference command."""
    # The TF2 path executes the compatibility shim above because it handles model
    # warm-up, H5 loading, and drawing in a way that works reliably on this setup.
    tf2_output = str(output_path) if mode == "to-file" and output_path is not None else ""
    command = [
        python_bin,
        "-c",
        INLINE_TF2_SCRIPT,
        "--weights",
        weights,
        "--image",
        str(job.local_image),
        "--output",
        tf2_output,
    ]
    # Viewer mode adds the flag that opens the rendered image window.
    if mode == "viewer":
        command.append("--show-image")
    return command


def build_pytorch_jobs_payload(
    selection,
    output_paths,
) -> list[dict[str, str | None]]:
    """Build the job list consumed by the inline PyTorch batch runner."""
    return [
        {
            "input": str(job.local_image),
            "output": str(output_path) if output_path is not None else None,
        }
        for job, output_path in zip(selection.jobs, output_paths)
    ]


def build_pytorch_batch_command(
    python_bin,
    weights,
    backbone,
    mode,
    jobs_payload,
) -> list[str]:
    """Build the single PyTorch batch command that handles every image."""
    if not backbone:
        raise ValueError("A PyTorch backbone must be resolved before batch inference.")
    return [
        python_bin,
        "-c",
        INLINE_PYTORCH_SCRIPT,
        "--weights",
        weights,
        "--backbone",
        backbone,
        "--mode",
        mode,
        "--jobs-json",
        json.dumps(jobs_payload),
    ]


def build_subprocess_env(framework, python_bin) -> dict[str, str] | None:
    """Build the environment for the selected upstream command."""
    if framework != "tf2":
        return None

    if pytorch_cuda_available(python_bin):
        return None

    env = os.environ.copy()
    env.update(TF2_ENV_OVERRIDES)
    return env


def print_command(command, index, total) -> None:
    """Print the command before running it."""
    # Prefix batch commands with their position in the run.
    label = "Running:" if total == 1 else f"Running [{index}/{total}]:"
    print(label, " ".join(shlex.quote(part) for part in command))


def print_batch_item(job, output_path, index, total) -> None:
    """Print the batch item being processed."""
    # Print the input image currently being processed.
    print(f"[{index}/{total}] {job.display_input}")
    if output_path is not None:
        # Print the expected output location for this image.
        print(f"      -> {output_path}")


def main() -> int:
    """Run the main CLI flow."""
    args = build_parser().parse_args()
    repo_dir = Path(args.repo_dir).expanduser().resolve()

    # First verify that the requested repo really contains the expected upstream code.
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory does not exist: {repo_dir}")
    required_dirs = ("pytorch", "tf2")
    missing_dirs = [name for name in required_dirs if not (repo_dir / name).exists()]
    if missing_dirs:
        raise FileNotFoundError(
            f"{repo_dir} does not look like a FasterRCNN clone "
            f"(missing {', '.join(f'{name}/' for name in missing_dirs)})."
        )

    python_bin = resolve_python(repo_dir, args.python)
    ensure_supported_runtime_python(python_bin)
    framework = resolve_framework(args.framework, python_bin, repo_dir)

    # Resolve the inputs once so the execution loop can stay simple.
    # At this point the file knows which framework, weights, inputs, and outputs
    # will be used before it launches any upstream inference code.
    weights = args.weights or DEFAULT_WEIGHTS[framework]
    resolved_weights = resolve_weights(repo_dir, weights)
    selection = resolve_input_selection(repo_dir, args.image, dry_run=args.dry_run)

    # Reject viewer mode for directory batches because the upstream tools show one image at a time.
    if selection.is_directory and args.mode == "viewer":
        raise ValueError(
            "--mode viewer only supports a single image input. "
            "Use --mode to-file when --image points to a directory."
        )

    # Output path resolution mirrors the normalized input selection so each job has
    # one concrete destination in file-writing mode.
    output_paths, output_label = resolve_output_paths(selection, args.output, args.mode)
    backbone = args.backbone or infer_backbone(weights)
    # The upstream launcher wants a backbone name in addition to the checkpoint
    # path, so infer it once here instead of asking the user to duplicate it.

    # Print one high-level summary before any subprocesses start.
    if args.dry_run or not args.verbose_command:
        print(
            format_run_summary(
                requested_framework=args.framework,
                framework=framework,
                weights=resolved_weights,
                mode=args.mode,
                selection=selection,
                output_label=output_label,
            )
        )

    total_jobs = len(selection.jobs)
    if framework == "pytorch":
        jobs_payload = build_pytorch_jobs_payload(selection, output_paths)
        command = build_pytorch_batch_command(
            python_bin=python_bin,
            weights=resolved_weights,
            backbone=backbone,
            mode=args.mode,
            jobs_payload=jobs_payload,
        )

        if args.verbose_command or args.dry_run:
            print_command(command, index=1, total=1)

        if args.dry_run:
            return 0

        for output_path in output_paths:
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(command, cwd=repo_dir, check=True)
        return 0

    # Each job becomes one upstream subprocess call so single-image and batch modes
    # both reuse the same execution path.
    for index, (job, output_path) in enumerate(zip(selection.jobs, output_paths), start=1):
        # Each iteration turns one normalized job description into the exact
        # upstream command line needed for TF2 or PyTorch inference.
        command = build_command(
            framework=framework,
            python_bin=python_bin,
            weights=resolved_weights,
            backbone=backbone,
            job=job,
            mode=args.mode,
            output_path=output_path,
        )
        command_env = build_subprocess_env(framework, python_bin)

        if args.verbose_command or args.dry_run:
            print_command(command, index=index, total=total_jobs)
        elif total_jobs > 1:
            # In quiet batch mode, print just the current image and destination.
            print_batch_item(job, output_path, index=index, total=total_jobs)

        if args.dry_run:
            continue

        # Create the destination folder before launching inference so a successful
        # upstream run can immediately copy/save its annotated output.
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # The heavy lifting stays upstream; this file's job is to make sure
        # every run enters with consistent paths, weights, and save behavior.
        subprocess.run(command, cwd=repo_dir, check=True, env=command_env)

        if output_path is None:
            continue

        # Print per-image destinations so the saved artifacts are easy to find later.
        if total_jobs == 1:
            print(f"Saved demo output to: {output_path}")
        else:
            print(f"[{index}/{total_jobs}] Saved demo output to: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as exc:
        command = " ".join(shlex.quote(part) for part in exc.cmd)
        print(
            f"Error: upstream FasterRCNN command failed with exit code "
            f"{exc.returncode}: {command}",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode) from None
