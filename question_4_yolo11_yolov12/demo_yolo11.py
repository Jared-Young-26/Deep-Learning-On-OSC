#!/usr/bin/env python3
"""Run YOLO11 inference from a local Ultralytics checkout."""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

QUESTION_DIR = Path(__file__).resolve().parent
REPO_ROOT = QUESTION_DIR.parent
DEFAULT_REPO_DIR = REPO_ROOT / "external" / "ultralytics"
DEFAULT_INPUT_DIR = QUESTION_DIR / "inputs" / "yolo11"
DEFAULT_OUTPUT_DIR = QUESTION_DIR / "outputs" / "yolo11"
DEFAULT_PROJECT_DIR = QUESTION_DIR / "runs" / "detect"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# The inline script keeps the actual model invocation inside the target
# environment while the outer file handles path resolution and artifact copying.
INLINE_SCRIPT = r'''
import argparse

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--project", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--device", default="")
args = parser.parse_args()

# Load the requested checkpoint into one YOLO model object.
model = YOLO(args.model)
# Build the keyword arguments passed into `predict()`.
predict_kwargs = {
    "source": args.source,
    "imgsz": args.imgsz,
    "conf": args.conf,
    "save": True,
    "verbose": False,
    "project": args.project,
    "name": args.name,
    "exist_ok": True,
}
# Pass the device only when the caller provided one.
if args.device:
    predict_kwargs["device"] = args.device

# Run one prediction call with the resolved arguments.
model.predict(**predict_kwargs)
'''


# These helpers normalize local paths so the CLI behaves the same whether it is
# launched from the repository root or from this directory.
def resolve_python(repo_dir, requested) -> str:
    """Resolve the Python executable used for inference."""
    # Prefer the repository-local virtual environment when it exists.
    if requested:
        return requested
    venv_python = repo_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def resolve_project(repo_dir, project) -> Path:
    """Resolve the Ultralytics run-root directory."""
    # YOLO writes raw run artifacts under the project directory first.
    # Expand "~" before anchoring relative paths.
    project_path = Path(project).expanduser()
    if not project_path.is_absolute():
        # Anchor relative project paths to this directory.
        project_path = QUESTION_DIR / project_path
    return project_path.resolve()


def resolve_output_dir(output_dir) -> Path:
    """Resolve the output directory path."""
    output_dir_path = Path(output_dir).expanduser()
    if not output_dir_path.is_absolute():
        # Anchor relative output directories to this directory.
        output_dir_path = QUESTION_DIR / output_dir_path
    return output_dir_path.resolve()


def resolve_output_path(output) -> Path:
    """Resolve one explicit output file path."""
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        # Anchor relative output files to this directory.
        output_path = QUESTION_DIR / output_path
    return output_path.resolve()


def describe_source(source, source_path, image_count) -> str:
    """Build a short description of the resolved source selection."""
    # Keep the printed run summary readable for URLs, files, and directories.
    if source_path is None:
        return f"{image_count} image(s) from URL {source}"

    if source_path.is_dir():
        return f"{image_count} image(s) from {source_path}"

    return f"{image_count} image(s) from {source_path.name}"


def clean_log_lines(text) -> list[str]:
    """Normalize and split the log output."""
    # Replace carriage returns before splitting the captured logs into lines.
    cleaned = ANSI_ESCAPE_RE.sub("", text.replace("\r", "\n"))
    # Drop empty lines and surrounding whitespace.
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def runtime_messages(stdout, stderr) -> list[str]:
    """Keep the short runtime messages worth printing."""
    # Keep only the short runtime notes that explain how the run behaved.
    messages = []
    for line in clean_log_lines(stderr) + clean_log_lines(stdout):
        # Rewrite the FlashAttention warning into a shorter status message.
        if "FlashAttention is not available" in line:
            messages.append("FlashAttention unavailable on this device; using fallback attention.")
        # Skip the raw save-path line because this script prints curated paths later.
        elif line.startswith("Results saved to "):
            continue

    # Drop duplicate messages before printing them.
    deduped = []
    for message in messages:
        # Preserve the first copy of each message only.
        if message not in deduped:
            deduped.append(message)
    return deduped


def is_url(source) -> bool:
    """Return True when the value looks like a URL."""
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def resolve_source(source) -> tuple[str, Path | None]:
    """Resolve the input source into a URL or one absolute path."""
    # Normalize the user input into either a raw URL string or one absolute local path.
    # Return URLs unchanged and mark them as non-path inputs.
    if is_url(source):
        return source, None

    # Expand "~" before anchoring a local source path.
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        # Anchor relative source paths to this directory.
        source_path = QUESTION_DIR / source_path
    source_path = source_path.resolve()

    # Stop before launching inference if the local path is missing.
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    return str(source_path), source_path


def list_supported_images(directory) -> list[Path]:
    """List the supported images under one directory."""
    # Batch mode uses only top-level files so the copied outputs keep one
    # predictable filename per input image.
    # Return one sorted list of top-level supported image files.
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


def expected_output_names(source, source_path) -> list[str]:
    """Predict which annotated filenames Ultralytics will save."""
    # Predict the filenames YOLO will save so the later copy step knows exactly
    # which artifacts to collect.
    if source_path is None:
        url_name = Path(urlparse(source).path).name
        if not url_name:
            raise ValueError("URL source must include a filename in the path or use --output.")
        # Use the URL filename directly for the saved image name.
        return [url_name]

    if source_path.is_dir():
        # In directory mode, expect one saved image per discovered input file.
        images = list_supported_images(source_path)
        if not images:
            suffixes = ", ".join(sorted(IMAGE_SUFFIXES))
            raise FileNotFoundError(
                f"No supported image files found in {source_path}. Add files with one of: {suffixes}"
            )
        return [path.name for path in images]

    # For a single local file, verify the suffix before using its name.
    if source_path.suffix.lower() not in IMAGE_SUFFIXES:
        suffixes = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(
            f"Unsupported source file type for {source_path.name}. Supported suffixes: {suffixes}"
        )

    return [source_path.name]


def copy_saved_images(
    save_dir, names, output_dir, output_path
) -> list[Path]:
    """Copy saved images into the final output location."""
    # Ultralytics saves raw outputs into the run directory first, then this
    # helper copies the final annotated images into the local output location.
    available = sorted(
        path.name for path in save_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if output_path is not None:
        # Single-image mode can write directly to one requested destination path.
        source_image = save_dir / names[0]
        # Stop if the expected saved image does not exist.
        if not source_image.exists():
            raise FileNotFoundError(
                f"Expected annotated image {source_image.name} was not found in {save_dir}. "
                f"Available files: {available}"
            )
        # Create the destination folder before copying the image.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy the one saved image to the requested output path.
        shutil.copy2(source_image, output_path)
        return [output_path]

    # Directory mode writes one copied file per expected saved image.
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    missing_names = []

    for name in names:
        # Directory mode preserves the original filenames so outputs line up with inputs.
        source_image = save_dir / name
        # Record any expected file that YOLO did not save.
        if not source_image.exists():
            missing_names.append(name)
            continue
        destination = output_dir / name
        # Copy the saved image into the managed output directory.
        shutil.copy2(source_image, destination)
        copied_paths.append(destination)

    if missing_names:
        raise FileNotFoundError(
            f"Expected annotated image(s) not found in {save_dir}: {', '.join(missing_names)}. "
            f"Available files: {available}"
        )

    return copied_paths


# The public CLI exposes only the options this file uses directly.
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Simple YOLO11 demo runner (GitHub source install)."
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help=(
            "Path to ultralytics repo clone. "
            "Defaults to <repo>/external/ultralytics."
        ),
    )
    parser.add_argument("--python", default=None, help="Python interpreter to run inference.")
    parser.add_argument("--model", default="yolo11n.pt", help="Model checkpoint name/path.")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_INPUT_DIR),
        help="Input image file, directory, or URL. Defaults to this question folder's inputs directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional single-image output path. Invalid when --source is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where annotated images are copied using matching filenames.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT_DIR),
        help="Where YOLO will write its run directory.",
    )
    parser.add_argument("--name", default="yolo11_demo", help="YOLO run name.")
    parser.add_argument("--device", default="", help="Device (e.g., cpu, 0).")
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    return parser


def main() -> int:
    """Run the YOLO11 inference flow."""
    args = build_parser().parse_args()
    repo_dir = Path(args.repo_dir).expanduser().resolve()

    # Validate that the requested upstream checkout is really the Ultralytics repo.
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory does not exist: {repo_dir}")
    if not (repo_dir / "ultralytics").exists():
        raise FileNotFoundError(
            f"{repo_dir} does not look like an ultralytics clone (missing ultralytics/)."
        )

    # Resolve every path before launching YOLO so the artifact copy step stays deterministic.
    python_bin = resolve_python(repo_dir, args.python)
    project_dir = resolve_project(repo_dir, args.project)
    resolved_source, source_path = resolve_source(args.source)
    output_dir = resolve_output_dir(args.output_dir)
    output_path = resolve_output_path(args.output) if args.output else None

    # Reject a single-file output path when the source expands to multiple images.
    if source_path is not None and source_path.is_dir() and output_path is not None:
        raise ValueError("--output cannot be used when --source is a directory. Use --output-dir instead.")

    # Match the post-run copy step to whatever filenames YOLO is expected to emit.
    output_names = expected_output_names(resolved_source, source_path)

    # Launch one inline Python snippet inside the repository environment so this
    # file controls imports and arguments without depending on a shell command.
    command = [
        python_bin,
        "-c",
        INLINE_SCRIPT,
        "--model",
        args.model,
        "--source",
        resolved_source,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--project",
        str(project_dir),
        "--name",
        args.name,
        "--device",
        args.device,
    ]

    if args.dry_run:
        # Dry-run mode prints the exact command and stops.
        print("Dry run:", " ".join(shlex.quote(part) for part in command))
        return 0

    # Run inference once, then copy the saved images into the local outputs directory.
    print(
        f"Processing YOLO11 on {describe_source(resolved_source, source_path, len(output_names))}",
        flush=True,
    )
    # Capture stdout and stderr so the useful run messages can be replayed on success or failure.
    completed = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    if completed.returncode != 0:
        # Replay captured stdout before surfacing the failure.
        if completed.stdout:
            print(completed.stdout, end="")
        # Replay captured stderr to stderr as well.
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        completed.check_returncode()

    # Ultralytics stores the raw run under project/name before the curated copy step.
    save_dir = project_dir / args.name
    # Filter the upstream logs down to the lines that explain the run outcome.
    for message in runtime_messages(completed.stdout, completed.stderr):
        print(message)
    print(f"Run artifacts saved to: {save_dir}")
    # Copy the annotated images from the raw run folder into the final output location.
    copied_paths = copy_saved_images(save_dir, output_names, output_dir, output_path)
    if output_path is None:
        print(f"Copied {len(copied_paths)} annotated image(s) to: {output_dir}")
    for copied_path in copied_paths:
        # Print every copied output path so the final artifacts are easy to locate.
        print(f"Saved YOLO11 annotated image to: {copied_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
