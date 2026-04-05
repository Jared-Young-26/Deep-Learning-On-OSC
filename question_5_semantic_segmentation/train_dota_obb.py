#!/usr/bin/env python3
"""Fine-tune YOLO11 OBB on the prepared DOTA dataset and save a reusable checkpoint."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from q5_obb_common import (
    DEFAULT_DOTA_YAML,
    DEFAULT_FINETUNED_MODEL,
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_REPO_DIR,
    DEFAULT_TRAIN_PROJECT_DIR,
    ensure_ultralytics_import,
    ensure_parent,
    ensure_ultralytics_repo,
    is_url,
    maybe_reexec_with_repo_python,
    resolve_model_argument,
    resolve_question_path,
    resolve_repo_dir,
)

REEXEC_MARKER = "Q5_OBB_TRAIN_INNER"


# Training produces the one reusable checkpoint that the demo script will prefer later.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune yolo11s-obb on the prepared DOTA dataset and store "
            "question_5_semantic_segmentation/models/dota_obb/best.pt for later demo reuse."
        )
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help="Path to the local ultralytics clone. Defaults to <repo>/external/ultralytics.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Interpreter to use; defaults to <repo>/.venv/bin/python if present.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_PRETRAINED_MODEL),
        help="Starting checkpoint. Defaults to the repo-local pretrained yolo11s-obb.pt.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DOTA_YAML),
        help="Dataset YAML. Defaults to the repo-local DOTAv1-split.yaml.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--device", default="", help="Device string such as 0, 0,1, or cpu.")
    parser.add_argument(
        "--project",
        default=str(DEFAULT_TRAIN_PROJECT_DIR),
        help="Ultralytics run root. Defaults to question_5_semantic_segmentation/runs/obb/train.",
    )
    parser.add_argument(
        "--name",
        default="dota_yolo11s_obb",
        help="Run name under the training project directory.",
    )
    parser.add_argument(
        "--output-model",
        default=str(DEFAULT_FINETUNED_MODEL),
        help="Alias path for the reusable checkpoint after training completes.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow Ultralytics to reuse an existing run directory with the same name.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    return parser


def resolve_training_model(raw_model: str) -> str:
    # Accept URLs, absolute paths, or question-local model aliases, but fail early
    # if the starting checkpoint the user asked for does not actually exist.
    model_value = resolve_model_argument(raw_model)
    if is_url(model_value):
        return model_value

    model_path = Path(model_value).expanduser()
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            f"Starting checkpoint does not exist: {model_path}. "
            "Run bootstrap_dota_obb.py first or pass a different --model."
        )
    return model_value


def main() -> int:
    args = build_parser().parse_args()

    # Re-enter through the repo-local interpreter so the training run uses the
    # exact Ultralytics environment prepared by setup_yolo11_osc.sh.
    repo_dir = ensure_ultralytics_repo(resolve_repo_dir(args.repo_dir))
    maybe_reexec_with_repo_python(repo_dir, args.python, REEXEC_MARKER)
    ensure_ultralytics_import(repo_dir)

    resolved_model = resolve_model_argument(args.model) if args.dry_run else resolve_training_model(args.model)
    data_yaml = resolve_question_path(args.data)
    project_dir = resolve_question_path(args.project)
    output_model = resolve_question_path(args.output_model)

    if args.dry_run:
        print(f"Repo:          {repo_dir}")
        print(f"Model:         {resolved_model}")
        print(f"Data YAML:     {data_yaml}")
        print(f"Project dir:   {project_dir}")
        print(f"Run name:      {args.name}")
        print(f"Output alias:  {output_model}")
        return 0

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML does not exist: {data_yaml}. Run bootstrap_dota_obb.py first."
        )

    from ultralytics import YOLO

    # Train from the chosen starting checkpoint, then copy the best resulting
    # weights into a stable alias path for later inference reuse.
    print(f"Starting OBB training from {resolved_model}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Runs root:    {project_dir}")

    model = YOLO(resolved_model)
    # The wrapper only sets the training knobs that matter for the assignment;
    # Ultralytics still owns the actual optimization loop and checkpoint writing.
    train_kwargs: dict[str, object] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "task": "obb",
    }
    if args.device:
        train_kwargs["device"] = args.device

    # The actual optimization loop lives inside Ultralytics; this wrapper mainly
    # standardizes the inputs and preserves the best checkpoint under a stable name.
    model.train(**train_kwargs)

    # Preserve the best checkpoint under one predictable alias so the demo script
    # does not need to guess which run directory produced the final weights.
    best_checkpoint = model.trainer.best if model.trainer.best.exists() else model.trainer.last
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Training completed but no checkpoint was found under {model.trainer.save_dir / 'weights'}."
        )

    ensure_parent(output_model)
    shutil.copy2(best_checkpoint, output_model)

    print("")
    print("Training complete.")
    print(f"Run directory:       {model.trainer.save_dir}")
    print(f"Best checkpoint:     {best_checkpoint}")
    print(f"Reusable alias path: {output_model}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
