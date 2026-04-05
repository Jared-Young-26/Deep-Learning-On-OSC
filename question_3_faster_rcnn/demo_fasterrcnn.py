#!/usr/bin/env python3
"""Run simple predictions using trzy/FasterRCNN."""

from __future__ import annotations

import argparse
import hashlib
import platform
import shlex
import shutil
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

# The upstream TF2 path expects a different entrypoint shape than this project needs.
# This embedded script keeps the demo self-contained and preserves H5 loading on modern
# local Python/macOS setups.
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

# The upstream text-drawing helper is fragile on some current Pillow versions,
# so the wrapper swaps in a simpler compatible implementation before rendering.
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


# The selection object lets the rest of the wrapper handle one image and a whole
# directory of images through the same execution loop.
@dataclass(frozen=True)
class InputSelection:
    jobs: list[ImageJob]
    source_label: str
    is_directory: bool


# These helpers choose the safest runtime path based on the machine and the weights name.
def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def infer_backbone(weights: str) -> str | None:
    # Some upstream PyTorch checkpoints require the backbone name separately,
    # so infer it from the filename when the user does not pass --backbone.
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
    # On macOS the TF2 path is the practical default because the upstream PyTorch
    # implementation expects CUDA, while on OSC/Linux the PyTorch path is natural.
    return "tf2" if platform.system() == "Darwin" else "pytorch"


def resolve_python(repo_dir: Path, requested: str | None) -> str:
    # Prefer the repo-local virtualenv so the wrapper uses the same dependencies
    # that setup_fasterrcnn_osc.sh installed for this Faster R-CNN clone.
    if requested:
        return requested

    venv_python = repo_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def pytorch_cuda_available(python_bin: str) -> bool:
    # The upstream PyTorch demo is CUDA-only, so check that before launching a run
    # that would otherwise fail after all path resolution is already done.
    probe = (
        "import torch; "
        "print('1' if getattr(torch.cuda, 'is_available', lambda: False)() else '0')"
    )
    result = subprocess.run(
        [python_bin, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "1"


# Input resolution supports the three professor-demo cases: explicit local files,
# a whole input directory, or a single remote example image URL.
def resolve_weights(repo_dir: Path, weights: str) -> str:
    if is_url(weights):
        return weights

    candidate = Path(weights).expanduser()
    if not candidate.is_absolute():
        candidate = repo_dir / candidate

    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find weights file: {candidate}\n"
            "Run `bash download_models_fasterrcnn.sh` from question_3_faster_rcnn "
            "or pass an explicit --weights path/URL."
        )
    return str(candidate.resolve())


def remote_cache_path(image_url: str) -> Path:
    # Cache remote images under a deterministic hashed filename so repeated demos
    # reuse the same downloaded file instead of redownloading every time.
    parsed = urlparse(image_url)
    filename = Path(unquote(parsed.path)).name or "demo_image.jpg"
    stem = Path(filename).stem or "demo_image"
    suffix = Path(filename).suffix or ".jpg"
    cache_name = (
        f"{stem}-{hashlib.sha256(image_url.encode('utf-8')).hexdigest()[:12]}{suffix}"
    )
    return DEFAULT_DOWNLOADED_INPUT_DIR / cache_name


def cache_remote_image(image_url: str, dry_run: bool = False) -> Path:
    cache_path = remote_cache_path(image_url)
    if dry_run:
        return cache_path.resolve()

    # Keep downloaded examples under inputs/downloaded so they are visible in the
    # project tree but still separate from the manually curated input images.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.resolve()

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
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to download input image URL: {image_url}\n"
            f"curl error: {result.stderr.strip() or result.stdout.strip() or result.returncode}\n"
            "Pass a local image path with --image to avoid network lookup."
        )
    return cache_path.resolve()


def resolve_local_input(repo_dir: Path, image: str) -> Path:
    candidate = Path(image).expanduser()
    search_paths: list[Path] = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        # Search relative to the current shell directory, then the question folder,
        # then the upstream repo so the wrapper is forgiving about where it is run.
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
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        checked.append(resolved)
        if resolved.exists():
            return resolved

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        f"Could not find input path: {image}\nChecked:\n{checked_text}"
    )


def discover_directory_images(input_root: Path) -> list[Path]:
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
    if not images:
        raise FileNotFoundError(
            "No supported image files were found under directory:\n"
            f"  {input_root}\n"
            f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}"
        )
    return images


def resolve_input_selection(
    repo_dir: Path,
    image: str,
    dry_run: bool = False,
) -> InputSelection:
    # Convert the user-facing --image argument into one normalized list of jobs.
    # Everything downstream can then treat single-image and batch mode uniformly.
    if is_url(image):
        local_image = cache_remote_image(image, dry_run=dry_run)
        parsed = urlparse(image)
        output_name = Path(unquote(parsed.path)).name or local_image.name
        output_path = Path(output_name)
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


def resolve_single_output_path(job: ImageJob, raw_output: str | None) -> Path:
    # For single-image runs, accept either an explicit filename or a directory and
    # fill in the actual output filename from the input image name when needed.
    if raw_output is None:
        return (DEFAULT_OUTPUT_DIR / job.output_relative.name).resolve()

    candidate = Path(raw_output).expanduser()
    if candidate.exists() and candidate.is_dir():
        return (candidate / job.output_relative.name).resolve()
    if candidate.suffix == "":
        return (candidate / job.output_relative.name).resolve()
    return candidate.resolve()


def resolve_directory_output_root(raw_output: str | None) -> Path:
    # Directory runs must resolve to one output root because each input image keeps
    # its relative name underneath that root.
    if raw_output is None:
        return DEFAULT_OUTPUT_DIR.resolve()

    candidate = Path(raw_output).expanduser()
    if candidate.exists() and candidate.is_file():
        raise ValueError(
            "When --image is a directory, --output must be a directory path, "
            f"not a file: {candidate}"
        )
    if candidate.suffix and not candidate.exists():
        raise ValueError(
            "When --image is a directory, --output must be a directory path. "
            f"Received: {candidate}"
        )
    return candidate.resolve()


def resolve_output_paths(
    selection: InputSelection,
    raw_output: str | None,
    mode: str,
) -> tuple[list[Path | None], str | None]:
    # Viewer mode leaves output handling to the upstream GUI/image viewer path.
    if mode != "to-file":
        return [None] * len(selection.jobs), None

    if selection.is_directory:
        output_root = resolve_directory_output_root(raw_output)
        return (
            [(output_root / job.output_relative).resolve() for job in selection.jobs],
            str(output_root),
        )

    output_path = resolve_single_output_path(selection.jobs[0], raw_output)
    return ([output_path], str(output_path))


# The public CLI stays small while the wrapper hides the upstream command details.
def build_parser() -> argparse.ArgumentParser:
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
        choices=("pytorch", "tf2"),
        default=infer_default_framework(),
        help=(
            "Choose which implementation to run. Defaults to tf2 on macOS "
            "and pytorch elsewhere."
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
    framework: str,
    weights: str,
    mode: str,
    selection: InputSelection,
    output_label: str | None,
) -> str:
    # Print a compact summary before execution so the high-level plan of the run
    # is visible without having to read the full subprocess command.
    if selection.is_directory:
        lines = [
            "Running Faster R-CNN demo batch",
            f"  framework: {framework}",
            f"  mode: {mode}",
            f"  input root: {selection.source_label}",
            f"  images: {len(selection.jobs)}",
            f"  weights: {weights}",
        ]
        if output_label:
            lines.append(f"  output root: {output_label}")
        return "\n".join(lines)

    lines = [
        "Running Faster R-CNN demo",
        f"  framework: {framework}",
        f"  mode: {mode}",
        f"  image: {selection.jobs[0].display_input}",
        f"  weights: {weights}",
    ]
    if output_label:
        lines.append(f"  output: {output_label}")
    return "\n".join(lines)


# Command construction is where this wrapper translates one stable local interface
# into whichever upstream Faster R-CNN implementation is actually being used.
def build_command(
    framework: str,
    python_bin: str,
    weights: str,
    backbone: str | None,
    job: ImageJob,
    mode: str,
    output_path: Path | None,
) -> list[str]:
    if framework == "pytorch":
        # The PyTorch repo exposes a module entrypoint and encodes the prediction
        # mode directly in the CLI flag we pass.
        command = [
            python_bin,
            "-m",
            "pytorch.FasterRCNN",
            f"--load-from={weights}",
        ]
        if backbone:
            command.append(f"--backbone={backbone}")
        if mode == "to-file":
            command.append(f"--predict-to-file={job.local_image}")
        else:
            command.append(f"--predict={job.local_image}")
        return command

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
    if mode == "viewer":
        command.append("--show-image")
    return command


def print_command(command: list[str], index: int, total: int) -> None:
    label = "Running:" if total == 1 else f"Running [{index}/{total}]:"
    print(label, " ".join(shlex.quote(part) for part in command))


def print_batch_item(job: ImageJob, output_path: Path | None, index: int, total: int) -> None:
    print(f"[{index}/{total}] {job.display_input}")
    if output_path is not None:
        print(f"      -> {output_path}")


def main() -> int:
    args = build_parser().parse_args()
    repo_dir = Path(args.repo_dir).expanduser().resolve()

    # First verify that the requested repo really contains the expected upstream code.
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory does not exist: {repo_dir}")
    if not (repo_dir / args.framework).exists():
        raise FileNotFoundError(
            f"{repo_dir} does not look like a FasterRCNN clone "
            f"(missing {args.framework}/ directory)."
        )

    python_bin = resolve_python(repo_dir, args.python)
    if args.framework == "pytorch" and not pytorch_cuda_available(python_bin):
        raise RuntimeError(
            "The upstream PyTorch FasterRCNN implementation is CUDA-only and "
            "cannot run on this machine.\n"
            "Use `--framework tf2` for a local Mac/CPU demo, or run the "
            "PyTorch path on a CUDA-enabled system."
        )

    # Resolve the demo inputs once so the execution loop can stay simple.
    # At this point the wrapper knows which framework, weights, inputs, and outputs
    # will be used before it launches any upstream inference code.
    weights = args.weights or DEFAULT_WEIGHTS[args.framework]
    resolved_weights = resolve_weights(repo_dir, weights)
    selection = resolve_input_selection(repo_dir, args.image, dry_run=args.dry_run)

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

    # Print one high-level summary up front so the run is easy to narrate live.
    if not args.verbose_command and not args.dry_run:
        print(
            format_run_summary(
                framework=args.framework,
                weights=resolved_weights,
                mode=args.mode,
                selection=selection,
                output_label=output_label,
            )
        )

    total_jobs = len(selection.jobs)
    # Each job becomes one upstream subprocess call so single-image and batch modes
    # both reuse the same execution path.
    for index, (job, output_path) in enumerate(zip(selection.jobs, output_paths), start=1):
        # Each iteration turns one normalized job description into the exact
        # upstream command line needed for TF2 or PyTorch inference.
        command = build_command(
            framework=args.framework,
            python_bin=python_bin,
            weights=resolved_weights,
            backbone=backbone,
            job=job,
            mode=args.mode,
            output_path=output_path,
        )

        if args.verbose_command or args.dry_run:
            print_command(command, index=index, total=total_jobs)
        elif total_jobs > 1:
            print_batch_item(job, output_path, index=index, total=total_jobs)

        if args.dry_run:
            continue

        # Create the destination folder before launching inference so a successful
        # upstream run can immediately copy/save its annotated output.
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # The heavy lifting stays upstream; this wrapper's job is to make sure
        # every run enters with consistent paths, weights, and save behavior.
        subprocess.run(command, cwd=repo_dir, check=True)

        # The PyTorch implementation always writes predictions.png in the repo,
        # so the wrapper copies that canonical artifact into this question folder.
        if output_path is None:
            continue

        if args.framework == "pytorch":
            generated = repo_dir / "predictions.png"
            if generated.exists():
                shutil.copy2(generated, output_path)
            else:
                # If the canonical artifact is missing, the subprocess likely ran
                # but the upstream tool did not produce the expected saved image.
                print(
                    "Inference finished, but predictions.png was not found. "
                    "Check command output for errors."
                )
                return 0

        # Print per-image destinations so the saved artifacts are easy to find later.
        if total_jobs == 1:
            print(f"Saved demo output to: {output_path}")
        else:
            print(f"[{index}/{total_jobs}] Saved demo output to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
