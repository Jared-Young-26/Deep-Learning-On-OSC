#!/usr/bin/env python3
"""Train a YOLO11 segmentation model on the prepared iSAID dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from q5_seg_common import (
    DEFAULT_DATASET_YAML,
    DEFAULT_FINETUNED_MODEL,
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_REPO_DIR,
    DEFAULT_TRAIN_PROJECT_DIR,
    ensure_gpu_device_ready,
    ensure_parent,
    ensure_ultralytics_import,
    ensure_ultralytics_repo,
    is_url,
    maybe_reexec_with_repo_python,
    resolve_model_argument,
    resolve_question_path,
    resolve_repo_dir,
)

# Re-exec marker used when the wrapper hops into the repo-local Ultralytics
# environment before importing YOLO.
REEXEC_MARKER = "Q5_SEG_TRAIN_INNER"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune yolo11s-seg on the prepared iSAID dataset and store "
            "question_5_semantic_segmentation/models/isaid_seg/best.pt for later demo reuse."
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
        help="Starting checkpoint. Defaults to the repo-local pretrained yolo11s-seg.pt.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATASET_YAML),
        help="Dataset YAML. Defaults to the repo-local iSAID segmentation YAML.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=768, help="Training image size.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers.")
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=None,
        help="Disable mosaic for the final N epochs. Defaults to all epochs on OSC.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write Ultralytics training and validation plot artifacts.",
    )
    parser.add_argument(
        "--validate-each-epoch",
        action="store_true",
        help="Run Ultralytics validation after every epoch instead of only at the end.",
    )
    parser.add_argument("--device", default="", help="Device string such as 0, 0,1, or cpu.")
    parser.add_argument(
        "--project",
        default=str(DEFAULT_TRAIN_PROJECT_DIR),
        help="Ultralytics run root. Defaults to question_5_semantic_segmentation/runs/segment/train.",
    )
    parser.add_argument(
        "--name",
        default="isaid_yolo11s_seg",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <project>/<name>/weights/last.pt if that checkpoint exists.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    return parser


def resolve_training_model(raw_model) -> str:
    """Resolve the model input used to start training."""
    # Resolve the user input into the exact model value YOLO should load.
    model_value = resolve_model_argument(raw_model)
    if is_url(model_value):
        return model_value

    # Fail early if the local checkpoint path does not exist.
    model_path = Path(model_value).expanduser()
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            f"Starting checkpoint does not exist: {model_path}. "
            "Run bootstrap_isaid_seg.py first or pass a different --model."
        )
    return model_value


def main() -> int:
    """Run the segmentation training flow."""
    args = build_parser().parse_args()

    # Resolve the upstream repo path before touching the environment.
    repo_dir = ensure_ultralytics_repo(resolve_repo_dir(args.repo_dir))

    # Resolve the user-facing paths once so later steps reuse the same values.
    resolved_model = resolve_model_argument(args.model) if args.dry_run else resolve_training_model(args.model)
    data_yaml = resolve_question_path(args.data)
    project_dir = resolve_question_path(args.project)
    output_model = resolve_question_path(args.output_model)
    run_dir = project_dir / args.name
    resume_checkpoint = run_dir / "weights" / "last.pt"
    close_mosaic = args.close_mosaic if args.close_mosaic is not None else args.epochs

    # Dry-run mode stops after printing the resolved paths.
    if args.dry_run:
        print(f"Repo:          {repo_dir}")
        print(f"Model:         {resolved_model}")
        print(f"Data YAML:     {data_yaml}")
        print(f"Project dir:   {project_dir}")
        print(f"Run name:      {args.name}")
        print(f"Run dir:       {run_dir}")
        print(f"Resume:        {'yes' if args.resume else 'no'}")
        print(f"Resume ckpt:   {resume_checkpoint}")
        print(f"Epochs:        {args.epochs}")
        print(f"Image size:    {args.imgsz}")
        print(f"Batch:         {args.batch}")
        print(f"Workers:       {args.workers}")
        print(f"Close mosaic:  {close_mosaic}")
        print(f"Val each ep:   {'yes' if args.validate_each_epoch else 'no'}")
        print(f"Plots:         {'yes' if args.plots else 'no'}")
        print(f"Output alias:  {output_model}")
        return 0

    # Re-exec inside the repository-local environment if the current Python differs.
    maybe_reexec_with_repo_python(repo_dir, args.python, REEXEC_MARKER)

    ensure_gpu_device_ready(
        args.device,
        sys.executable,
        "question_5_semantic_segmentation/train_isaid_seg.py",
        "--repo-dir",
        "external/ultralytics",
        "--device",
        "0",
    )

    # Import Ultralytics only after the correct environment is active.
    ensure_ultralytics_import(repo_dir)

    # Stop before training if the prepared dataset YAML is missing.
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML does not exist: {data_yaml}. Run bootstrap_isaid_seg.py first."
        )

    import torch

    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.utils import LOCAL_RANK, LOGGER, RANK
    from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first

    best_resume_checkpoint = run_dir / "weights" / "best.pt"

    def load_trusted_checkpoint(checkpoint_path: Path) -> dict:
        """Load one locally generated training checkpoint across Torch defaults."""
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")

    def completed_epochs(checkpoint_path: Path) -> int | None:
        """Return the 1-based completed epoch count stored in one checkpoint."""
        if not checkpoint_path.exists():
            return None
        checkpoint = load_trusted_checkpoint(checkpoint_path)
        epoch_index = checkpoint.get("epoch")
        if epoch_index is None:
            return None
        return int(epoch_index) + 1

    if args.resume:
        completed = completed_epochs(resume_checkpoint)
        if completed is not None and completed >= args.epochs:
            recovered_checkpoint = best_resume_checkpoint if best_resume_checkpoint.exists() else resume_checkpoint
            print(
                "Detected a completed training checkpoint; recovering the reusable alias "
                "without rerunning the training loop."
            )
            ensure_parent(output_model)
            shutil.copy2(recovered_checkpoint, output_model)
            print(f"Recovered checkpoint: {recovered_checkpoint}")
            print(f"Reusable alias path: {output_model}")
            return 0

    class OSCSafeSegmentationTrainer(SegmentationTrainer):
        """Cap validation batch size on OSC to reduce host-memory pressure."""

        def check_resume(self, overrides):
            """Keep the caller-requested epoch count when resuming training."""
            super().check_resume(overrides)
            if self.resume and "epochs" in overrides:
                self.args.epochs = overrides["epochs"]

        def _build_train_pipeline(self):
            """Rebuild the validation loader with a smaller OSC-safe batch size."""
            super()._build_train_pipeline()
            batch_size = self.batch_size // max(self.world_size, 1)
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size,
                rank=LOCAL_RANK,
                mode="val",
            )
            if RANK in {-1, 0}:
                LOGGER.info(
                    "Using validation batch size %s on OSC to avoid the default doubled val batch.",
                    batch_size,
                )

        def final_eval(self):
            """Skip Ultralytics' redundant post-training best.pt validation on OSC."""
            model = self.best if self.best.exists() else None
            with torch_distributed_zero_first(LOCAL_RANK):
                if RANK in {-1, 0}:
                    ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                    if model:
                        strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
            if model and RANK in {-1, 0}:
                LOGGER.info("Skipping final best.pt validation on OSC because epoch-end validation already ran.")

    if args.resume and not resume_checkpoint.exists():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {resume_checkpoint}. "
            "Start a fresh run first or omit --resume."
        )

    model_source = str(resume_checkpoint) if args.resume else resolved_model

    print(f"Starting segmentation training from {model_source}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Runs root:    {project_dir}")
    print(f"Run dir:      {run_dir}")
    print(f"Resume mode:  {'enabled' if args.resume else 'disabled'}")
    print(f"Close mosaic: {close_mosaic}")
    print(f"Validate each epoch: {'enabled' if args.validate_each_epoch else 'disabled'}")

    # Load the starting checkpoint into a YOLO model object.
    model = YOLO(model_source)

    # Build the keyword arguments passed into the Ultralytics trainer.
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "close_mosaic": close_mosaic,
        "val": args.validate_each_epoch,
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "plots": args.plots,
        "task": "segment",
    }
    # Pass the device only when the caller explicitly set one.
    if args.device:
        train_kwargs["device"] = args.device
    if args.resume:
        train_kwargs["resume"] = str(resume_checkpoint)

    # Launch training with the resolved configuration.
    model.train(trainer=OSCSafeSegmentationTrainer, **train_kwargs)

    # Prefer the best checkpoint, then fall back to the last checkpoint.
    best_checkpoint = model.trainer.best if model.trainer.best.exists() else model.trainer.last
    # Stop if training finished without leaving either checkpoint behind.
    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Training completed but no checkpoint was found under {model.trainer.save_dir / 'weights'}."
        )

    # Copy the selected checkpoint into the stable alias path.
    ensure_parent(output_model)
    # Copy the chosen checkpoint to the stable reusable location.
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
        # Convert Ctrl+C into the standard shell exit code.
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        # Print the final error as a one-line CLI message.
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
