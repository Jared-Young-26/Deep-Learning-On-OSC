#!/usr/bin/env python3
"""Prepare the local iSAID dataset layout for YOLO11 segmentation."""

from __future__ import annotations

import argparse
import sys

from q5_seg_common import (
    DEFAULT_ANNOTATIONS_DIR,
    DEFAULT_CONVERTED_DIR,
    DEFAULT_DATASET_YAML,
    DEFAULT_DOTA_EXTRACT_DIR,
    DEFAULT_DOTA_URL,
    DEFAULT_DOTA_ZIP,
    DEFAULT_ISAID_DATASET_PAGE_URL,
    DEFAULT_ISAID_GDRIVE_DIR,
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_RAW_ROOT,
    DEFAULT_REPO_DIR,
    DEFAULT_TRAIN_JSON,
    DEFAULT_VAL_JSON,
    DEFAULT_YOLO11_SEG_URL,
    DEFAULT_LABELS_DIR,
    download_file,
    ensure_ultralytics_import,
    ensure_ultralytics_repo,
    extract_zip,
    find_annotation_json,
    find_dota_root,
    gdown_download_folder,
    load_and_validate_category_maps,
    maybe_reexec_with_repo_python,
    normalize_isaid_annotation_json,
    parse_isaid_google_drive_links,
    prepare_isaid_raw_layout,
    replace_label_tree,
    resolve_question_path,
    resolve_repo_dir,
    validate_isaid_root,
    write_dataset_yaml,
)

REEXEC_MARKER = "Q5_SEG_BOOTSTRAP_INNER"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate a normalized iSAID dataset drop-in, convert the COCO annotations "
            "to YOLO segmentation labels, and write the local dataset YAML."
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
        "--raw-root",
        default=str(DEFAULT_RAW_ROOT),
        help="Normalized iSAID dataset root containing images/ and annotations/.",
    )
    parser.add_argument(
        "--yaml",
        default=str(DEFAULT_DATASET_YAML),
        help="Path for the generated local dataset YAML.",
    )
    parser.add_argument(
        "--converted-dir",
        default=str(DEFAULT_CONVERTED_DIR),
        help="Temporary directory used while converting COCO annotations to YOLO labels.",
    )
    parser.add_argument(
        "--pretrained-model",
        default=str(DEFAULT_PRETRAINED_MODEL),
        help="Where to save the reusable pretrained yolo11s-seg checkpoint.",
    )
    parser.add_argument(
        "--pretrained-url",
        default=DEFAULT_YOLO11_SEG_URL,
        help="URL for the pretrained yolo11s-seg checkpoint.",
    )
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help=(
            "Automatically download the official DOTA images plus the official iSAID train/val "
            "Google Drive folders and normalize them into the expected raw dataset layout."
        ),
    )
    parser.add_argument(
        "--dataset-page-url",
        default=DEFAULT_ISAID_DATASET_PAGE_URL,
        help="Official iSAID dataset page used to discover the Google Drive train/val links.",
    )
    parser.add_argument(
        "--dota-url",
        default=DEFAULT_DOTA_URL,
        help="URL for the DOTA-v1.0 image archive used by iSAID.",
    )
    parser.add_argument(
        "--dota-zip",
        default=str(DEFAULT_DOTA_ZIP),
        help="Where to store the DOTA ZIP archive.",
    )
    parser.add_argument(
        "--dota-extract-dir",
        default=str(DEFAULT_DOTA_EXTRACT_DIR),
        help="Where to extract the DOTA archive before linking train/val images into the raw iSAID layout.",
    )
    parser.add_argument(
        "--gdrive-dir",
        default=str(DEFAULT_ISAID_GDRIVE_DIR),
        help="Where to store the downloaded iSAID Google Drive train/val folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the checkpoint and rebuild the YOLO labels even if they already exist.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    return parser


def main() -> int:
    """Run the dataset bootstrap flow."""
    args = build_parser().parse_args()

    # Resolve the upstream repository path before touching the environment.
    repo_dir = ensure_ultralytics_repo(resolve_repo_dir(args.repo_dir))

    # Resolve all repository-local paths once so every later step uses the same values.
    raw_root = resolve_question_path(args.raw_root)
    yaml_path = resolve_question_path(args.yaml)
    converted_dir = resolve_question_path(args.converted_dir)
    pretrained_model = resolve_question_path(args.pretrained_model)
    dota_zip = resolve_question_path(args.dota_zip)
    dota_extract_dir = resolve_question_path(args.dota_extract_dir)
    gdrive_dir = resolve_question_path(args.gdrive_dir)
    # Derive the normalized annotation and label directories from the raw root.
    annotations_dir = raw_root / DEFAULT_ANNOTATIONS_DIR.relative_to(DEFAULT_RAW_ROOT)
    labels_dir = raw_root / DEFAULT_LABELS_DIR.relative_to(DEFAULT_RAW_ROOT)
    # Derive the normalized train and validation annotation file paths.
    train_json = raw_root / DEFAULT_TRAIN_JSON.relative_to(DEFAULT_RAW_ROOT)
    val_json = raw_root / DEFAULT_VAL_JSON.relative_to(DEFAULT_RAW_ROOT)

    # Dry-run mode stops after printing the resolved paths.
    if args.dry_run:
        print(f"Repo:             {repo_dir}")
        print(f"Raw dataset:      {raw_root}")
        print(f"Annotations dir:  {annotations_dir}")
        print(f"Labels dir:       {labels_dir}")
        print(f"Dataset YAML:     {yaml_path}")
        print(f"Converted labels: {converted_dir}")
        print(f"Pretrained model: {pretrained_model}")
        print(f"Auto-download:    {args.download_dataset}")
        print(f"DOTA ZIP:         {dota_zip}")
        print(f"DOTA extract dir: {dota_extract_dir}")
        print(f"Google Drive dir: {gdrive_dir}")
        return 0

    # Re-exec inside the repository-local environment if the current Python differs.
    maybe_reexec_with_repo_python(repo_dir, args.python, REEXEC_MARKER)

    # Import Ultralytics only after the correct environment is active.
    ensure_ultralytics_import(repo_dir)

    if args.download_dataset:
        # Download the DOTA image archive first because the iSAID image splits reference it.
        download_file(args.dota_url, dota_zip, force=args.force)
        needs_extract = args.force or not dota_extract_dir.exists()
        if not needs_extract:
            # Reuse the extracted directory only if the expected layout is present.
            try:
                find_dota_root(dota_extract_dir)
            except FileNotFoundError:
                needs_extract = True

        if needs_extract:
            if dota_extract_dir.exists():
                import shutil

                # Remove the stale extraction directory before unpacking again.
                shutil.rmtree(dota_extract_dir)
            # Extract the archive before searching for the DOTA image root.
            extract_zip(dota_zip, dota_extract_dir)

        # Locate the extracted DOTA image root after the archive is ready.
        dota_root = find_dota_root(dota_extract_dir)

        # Parse the official iSAID page to discover the train and validation links.
        drive_links = parse_isaid_google_drive_links(args.dataset_page_url)
        train_drive_dir = gdrive_dir / "train"
        val_drive_dir = gdrive_dir / "val"

        # Download the Google Drive folders into stable local directories.
        gdown_download_folder(
            drive_links["train"],
            train_drive_dir,
            force=args.force,
        )
        gdown_download_folder(
            drive_links["val"],
            val_drive_dir,
            force=args.force,
        )
        train_json_source = find_annotation_json(train_drive_dir, "train")
        val_json_source = find_annotation_json(val_drive_dir, "val")

        # Normalize the downloaded assets into the local raw dataset layout.
        prepare_isaid_raw_layout(
            raw_root=raw_root,
            dota_root=dota_root,
            train_json_source=train_json_source,
            val_json_source=val_json_source,
        )

    # Validate the normalized dataset layout before rewriting annotations.
    validate_isaid_root(raw_root)

    # Rewrite the train and validation JSON files to the canonical class ids.
    normalize_isaid_annotation_json(train_json, train_json, image_dir=raw_root / "images" / "train")
    normalize_isaid_annotation_json(val_json, val_json, image_dir=raw_root / "images" / "val")

    # Load the class map once and download the reusable starting checkpoint.
    name_map = load_and_validate_category_maps(train_json, val_json)
    download_file(args.pretrained_url, pretrained_model, force=args.force)

    train_labels_dir = labels_dir / "train"
    val_labels_dir = labels_dir / "val"
    # Treat the label tree as ready only when both split folders exist.
    labels_ready = train_labels_dir.exists() and val_labels_dir.exists()

    if args.force or not labels_ready:
        from ultralytics.data.converter import convert_coco

        if converted_dir.exists():
            import shutil

            # Remove any previous temporary conversion output first.
            shutil.rmtree(converted_dir)

        # Convert the normalized COCO annotations into YOLO segmentation labels.
        print(f"Converting COCO annotations from {annotations_dir}")
        convert_coco(
            labels_dir=str(annotations_dir),
            save_dir=str(converted_dir),
            use_segments=True,
            cls91to80=False,
        )

        # Move the generated label trees into their final train and validation folders.
        replace_label_tree(converted_dir / "labels" / "train", train_labels_dir)
        replace_label_tree(converted_dir / "labels" / "val", val_labels_dir)
        import shutil

        # Remove the temporary conversion directory after the labels are copied.
        shutil.rmtree(converted_dir, ignore_errors=True)
    else:
        print(f"Using existing YOLO segmentation labels under {labels_dir}")

    # Write the dataset YAML after the label tree and class map are ready.
    write_dataset_yaml(yaml_path, raw_root, name_map)

    print("")
    print("Bootstrap complete.")
    print(f"Raw dataset:      {raw_root}")
    print(f"Labels dir:       {labels_dir}")
    print(f"Dataset YAML:     {yaml_path}")
    print(f"Pretrained model: {pretrained_model}")
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
