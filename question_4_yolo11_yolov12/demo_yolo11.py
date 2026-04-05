#!/usr/bin/env python3
"""Run a simple YOLO11 prediction from a local ultralytics/ultralytics clone."""

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

# The wrapper delegates actual inference to the upstream package but keeps all
# path resolution and artifact copying in this question folder.
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

model = YOLO(args.model)
# The upstream YOLO API takes one predict() call with keyword arguments, so the
# local wrapper builds that call explicitly instead of shelling out to yolo CLI.
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
if args.device:
    predict_kwargs["device"] = args.device

model.predict(**predict_kwargs)
'''


# These helpers normalize repo-local paths so the CLI behaves the same whether it
# is launched from the repo root or from inside the question folder.
def resolve_python(repo_dir: Path, requested: str | None) -> str:
    # Prefer the repo-local virtualenv so the wrapper uses the exact packages
    # installed by setup_yolo11_osc.sh for this question.
    if requested:
        return requested
    venv_python = repo_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def resolve_project(repo_dir: Path, project: str) -> Path:
    # YOLO writes a run directory first; this wrapper later copies the saved
    # images out of that run directory into question_4_yolo11_yolov12/outputs.
    project_path = Path(project).expanduser()
    if not project_path.is_absolute():
        project_path = QUESTION_DIR / project_path
    return project_path.resolve()


def resolve_output_dir(output_dir: str) -> Path:
    output_dir_path = Path(output_dir).expanduser()
    if not output_dir_path.is_absolute():
        output_dir_path = QUESTION_DIR / output_dir_path
    return output_dir_path.resolve()


def resolve_output_path(output: str) -> Path:
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = QUESTION_DIR / output_path
    return output_path.resolve()


def describe_source(source: str, source_path: Path | None, image_count: int) -> str:
    # Keep the printed run summary human-readable whether the source is a URL,
    # one local file, or a whole directory.
    if source_path is None:
        return f"{image_count} image(s) from URL {source}"

    if source_path.is_dir():
        return f"{image_count} image(s) from {source_path}"

    return f"{image_count} image(s) from {source_path.name}"


def clean_log_lines(text: str) -> list[str]:
    cleaned = ANSI_ESCAPE_RE.sub("", text.replace("\r", "\n"))
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def runtime_messages(stdout: str, stderr: str) -> list[str]:
    # Filter the verbose Ultralytics output down to a few messages that are
    # actually useful in the assignment walkthrough.
    messages: list[str] = []
    for line in clean_log_lines(stderr) + clean_log_lines(stdout):
        if "FlashAttention is not available" in line:
            messages.append("FlashAttention unavailable on this device; using fallback attention.")
        elif line.startswith("Results saved to "):
            continue

    deduped: list[str] = []
    for message in messages:
        if message not in deduped:
            deduped.append(message)
    return deduped


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"}


def resolve_source(source: str) -> tuple[str, Path | None]:
    # Normalize the user input into either a raw URL string or one absolute local path.
    if is_url(source):
        return source, None

    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = QUESTION_DIR / source_path
    source_path = source_path.resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    return str(source_path), source_path


def list_supported_images(directory: Path) -> list[Path]:
    # Batch mode intentionally uses only top-level files in the inputs folder so
    # the output copying logic can preserve one predictable filename per image.
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


def expected_output_names(source: str, source_path: Path | None) -> list[str]:
    # Predict the filenames YOLO will save so the wrapper knows exactly what to
    # copy back into this question folder after inference finishes.
    if source_path is None:
        url_name = Path(urlparse(source).path).name
        if not url_name:
            raise ValueError("URL source must include a filename in the path or use --output.")
        return [url_name]

    if source_path.is_dir():
        images = list_supported_images(source_path)
        if not images:
            suffixes = ", ".join(sorted(IMAGE_SUFFIXES))
            raise FileNotFoundError(
                f"No supported image files found in {source_path}. Add files with one of: {suffixes}"
            )
        return [path.name for path in images]

    if source_path.suffix.lower() not in IMAGE_SUFFIXES:
        suffixes = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(
            f"Unsupported source file type for {source_path.name}. Supported suffixes: {suffixes}"
        )

    return [source_path.name]


def copy_saved_images(
    save_dir: Path, names: list[str], output_dir: Path, output_path: Path | None
) -> list[Path]:
    # Ultralytics saves into runs/detect/<name>/ first; this helper moves the
    # curated final artifacts into the local outputs/ folder used in the README.
    available = sorted(
        path.name for path in save_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if output_path is not None:
        # Single-image mode can write directly to one requested destination path.
        source_image = save_dir / names[0]
        if not source_image.exists():
            raise FileNotFoundError(
                f"Expected annotated image {source_image.name} was not found in {save_dir}. "
                f"Available files: {available}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, output_path)
        return [output_path]

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []
    missing_names: list[str] = []

    for name in names:
        # Directory mode preserves the original filenames so outputs line up with inputs.
        source_image = save_dir / name
        if not source_image.exists():
            missing_names.append(name)
            continue
        destination = output_dir / name
        shutil.copy2(source_image, destination)
        copied_paths.append(destination)

    if missing_names:
        raise FileNotFoundError(
            f"Expected annotated image(s) not found in {save_dir}: {', '.join(missing_names)}. "
            f"Available files: {available}"
        )

    return copied_paths


# The public CLI intentionally exposes only the parameters that matter for the
# assignment examples, not every Ultralytics option.
def build_parser() -> argparse.ArgumentParser:
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
    args = build_parser().parse_args()
    repo_dir = Path(args.repo_dir).expanduser().resolve()

    # Validate that the requested upstream checkout is really the Ultralytics repo.
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory does not exist: {repo_dir}")
    if not (repo_dir / "ultralytics").exists():
        raise FileNotFoundError(
            f"{repo_dir} does not look like an ultralytics clone (missing ultralytics/)."
        )

    # Resolve source/output layout before launching YOLO so copy-back is deterministic.
    # The wrapper computes every path up front so the actual inference phase is
    # just one subprocess call followed by predictable artifact collection.
    python_bin = resolve_python(repo_dir, args.python)
    project_dir = resolve_project(repo_dir, args.project)
    resolved_source, source_path = resolve_source(args.source)
    output_dir = resolve_output_dir(args.output_dir)
    output_path = resolve_output_path(args.output) if args.output else None

    if source_path is not None and source_path.is_dir() and output_path is not None:
        raise ValueError("--output cannot be used when --source is a directory. Use --output-dir instead.")

    # Match the post-run copy step to whatever filenames YOLO is expected to emit.
    output_names = expected_output_names(resolved_source, source_path)

    # Launch one inline Python snippet inside the repo's own environment so the
    # wrapper controls imports and arguments without depending on a shell-level
    # `yolo` executable being on PATH.
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
        print("Dry run:", " ".join(shlex.quote(part) for part in command))
        return 0

    # Run upstream inference once, then curate the saved images into this folder's outputs/.
    print(
        f"Processing YOLO11 on {describe_source(resolved_source, source_path, len(output_names))}",
        flush=True,
    )
    # Capture stdout/stderr so the wrapper can surface the useful run messages
    # while still failing cleanly if upstream inference raises an error.
    completed = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        completed.check_returncode()

    # Ultralytics stores the raw run under project/name; the wrapper reports that
    # location, then copies the final annotated images to the user-facing outputs dir.
    save_dir = project_dir / args.name
    # Filter the upstream logs down to the lines that help explain where the run
    # artifacts went and what YOLO saved.
    for message in runtime_messages(completed.stdout, completed.stderr):
        print(message)
    print(f"Run artifacts saved to: {save_dir}")
    copied_paths = copy_saved_images(save_dir, output_names, output_dir, output_path)
    if output_path is None:
        print(f"Copied {len(copied_paths)} annotated image(s) to: {output_dir}")
    for copied_path in copied_paths:
        print(f"Saved YOLO11 annotated image to: {copied_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
