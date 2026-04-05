#!/usr/bin/env python3
"""Download and prepare the DOTAv1 dataset for Question 5 YOLO11 OBB training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from q5_obb_common import (
    DEFAULT_DOTA_URL,
    DEFAULT_DOTA_YAML,
    DEFAULT_DOTA_ZIP,
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_RAW_DOTA_DIR,
    DEFAULT_REPO_DIR,
    DEFAULT_SPLIT_DOTA_DIR,
    DEFAULT_YOLO11_OBB_URL,
    download_file,
    ensure_ultralytics_import,
    ensure_ultralytics_repo,
    extract_zip,
    maybe_reexec_with_repo_python,
    parse_rates,
    resolve_question_path,
    resolve_repo_dir,
    validate_dota_root,
    write_dota_yaml,
)

REEXEC_MARKER = "Q5_OBB_BOOTSTRAP_INNER"


# Bootstrap is responsible only for assets and dataset preparation; training and
# inference stay in separate scripts so each stage is easy to justify independently.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap DOTAv1 and the reusable YOLO11 OBB baseline checkpoint "
            "for Question 5 training and demo reuse."
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
        "--download-url",
        default=DEFAULT_DOTA_URL,
        help="DOTAv1 archive URL. Defaults to the Ultralytics-hosted ZIP.",
    )
    parser.add_argument(
        "--dataset-zip",
        default=str(DEFAULT_DOTA_ZIP),
        help="Where to store the DOTAv1 ZIP archive.",
    )
    parser.add_argument(
        "--raw-root",
        default=str(DEFAULT_RAW_DOTA_DIR),
        help="Raw DOTAv1 root directory after extraction.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_SPLIT_DOTA_DIR),
        help="Output directory for the tiled DOTA split dataset.",
    )
    parser.add_argument(
        "--yaml",
        default=str(DEFAULT_DOTA_YAML),
        help="Path for the generated local dataset YAML file.",
    )
    parser.add_argument(
        "--pretrained-model",
        default=str(DEFAULT_PRETRAINED_MODEL),
        help="Where to save the reusable pretrained yolo11s-obb checkpoint.",
    )
    parser.add_argument(
        "--pretrained-url",
        default=DEFAULT_YOLO11_OBB_URL,
        help="URL for the pretrained yolo11s-obb checkpoint.",
    )
    parser.add_argument("--crop-size", type=int, default=1024, help="Base crop size for split_dota.")
    parser.add_argument("--gap", type=int, default=200, help="Gap between crops for split_dota.")
    parser.add_argument(
        "--rates",
        default="0.5,1.0,1.5",
        help="Comma-separated multi-scale rates for split_dota.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset ZIP and pretrained checkpoint even if they already exist.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract the DOTAv1 archive even if the raw dataset directory already looks complete.",
    )
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Re-run split_trainval and split_test even if the split dataset already exists.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    return parser


def split_ready(split_root: Path) -> bool:
    # Treat the split dataset as usable once the core train image/label folders exist.
    return (split_root / "images" / "train").exists() and (split_root / "labels" / "train").exists()


def main() -> int:
    args = build_parser().parse_args()

    # Re-enter through the repo-local interpreter so Ultralytics imports and helper
    # modules come from the intended Question 5 environment.
    repo_dir = ensure_ultralytics_repo(resolve_repo_dir(args.repo_dir))
    maybe_reexec_with_repo_python(repo_dir, args.python, REEXEC_MARKER)
    ensure_ultralytics_import(repo_dir)

    dataset_zip = resolve_question_path(args.dataset_zip)
    raw_root = resolve_question_path(args.raw_root)
    split_root = resolve_question_path(args.save_dir)
    yaml_path = resolve_question_path(args.yaml)
    pretrained_model = resolve_question_path(args.pretrained_model)
    # Multi-scale rates control how aggressively large aerial scenes are tiled before training.
    rates = parse_rates(args.rates)

    if args.dry_run:
        # Dry-run mode is just a path and parameter sanity check before large
        # downloads or expensive DOTA split generation.
        print(f"Repo:            {repo_dir}")
        print(f"Dataset ZIP:     {dataset_zip}")
        print(f"Raw dataset:     {raw_root}")
        print(f"Split dataset:   {split_root}")
        print(f"Dataset YAML:    {yaml_path}")
        print(f"Pretrained OBB:  {pretrained_model}")
        print(f"Rates:           {rates}")
        return 0

    from ultralytics.data.split_dota import split_test, split_trainval

    # Download the two reusable assets first: the raw dataset archive and the
    # pretrained OBB checkpoint that later training and demos both depend on.
    download_file(args.download_url, dataset_zip, force=args.force_download)
    download_file(args.pretrained_url, pretrained_model, force=args.force_download)

    # Normalize the raw dataset layout, then build the tiled split dataset that
    # Ultralytics expects for large aerial images.
    if args.force_extract or not raw_root.exists():
        extract_zip(dataset_zip, raw_root.parent)
    validate_dota_root(raw_root)

    if args.force_split or not split_ready(split_root):
        split_root.mkdir(parents=True, exist_ok=True)
        print(f"Creating tiled DOTA dataset at {split_root}")
        # split_trainval prepares the train/val crops Ultralytics consumes during
        # OBB training; split_test mirrors that layout for any available test set.
        split_trainval(str(raw_root), str(split_root), crop_size=args.crop_size, gap=args.gap, rates=rates)
        if (raw_root / "images" / "test").exists():
            split_test(str(raw_root), str(split_root), crop_size=args.crop_size, gap=args.gap, rates=rates)
    else:
        print(f"Using existing tiled DOTA dataset at {split_root}")

    # Write a repo-local YAML so later scripts can reference one stable dataset config.
    write_dota_yaml(yaml_path, split_root)

    print("")
    print("Bootstrap complete.")
    print(f"DOTAv1 ZIP:      {dataset_zip}")
    print(f"Raw dataset:     {raw_root}")
    print(f"Split dataset:   {split_root}")
    print(f"Dataset YAML:    {yaml_path}")
    print(f"Pretrained OBB:  {pretrained_model}")
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
