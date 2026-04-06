#!/usr/bin/env python3
"""Run YOLO11 segmentation inference on iSAID-style satellite imagery."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from q5_seg_common import (
    DEFAULT_FINETUNED_MODEL,
    DEFAULT_INDEX_CSV,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_REPO_DIR,
    class_color,
    ensure_parent,
    ensure_ultralytics_import,
    ensure_ultralytics_repo,
    is_url,
    list_supported_images,
    maybe_reexec_with_repo_python,
    normalize_name_map,
    parse_keep_classes,
    per_image_output_dir,
    resolve_model_argument,
    resolve_question_path,
    resolve_repo_dir,
    resolve_source,
    url_output_suffix,
)

REEXEC_MARKER = "Q5_SEG_DEMO_INNER"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run reusable YOLO11 segmentation inference for iSAID-style satellite images. "
            "The demo writes an overlay, a one-channel class map, a colorized semantic mask, "
            "and a JSON summary."
        )
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help="Path to the local ultralytics clone. Defaults to <repo>/external/ultralytics.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Checkpoint to use. If omitted, the demo prefers "
            "question_5_semantic_segmentation/models/isaid_seg/best.pt and falls back to "
            "question_5_semantic_segmentation/models/pretrained/yolo11s-seg.pt."
        ),
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_INPUT_DIR),
        help=(
            "Input image file, recursive image directory, or URL. "
            "Defaults to the repo-local satellite image input folder."
        ),
    )
    parser.add_argument(
        "--keep-classes",
        default="",
        help=(
            "Optional comma-separated class ids/names to keep. "
            "Omit this flag to keep all classes. "
            "Example: ship,storage tank,small vehicle"
        ),
    )
    parser.add_argument(
        "--output-overlay",
        default=None,
        help="Single-image only. Output path for the overlay image.",
    )
    parser.add_argument(
        "--output-class-ids",
        default=None,
        help="Single-image only. Output path for the one-channel class_ids PNG.",
    )
    parser.add_argument(
        "--output-mask",
        default=None,
        help="Single-image only. Output path for the colorized semantic mask PNG.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Single-image only. Output path for the per-image JSON summary.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory mode only. Root folder for one subfolder per image "
            "containing overlay, class_ids, mask, and summary files."
        ),
    )
    parser.add_argument(
        "--index-csv",
        default=None,
        help="Directory mode only. Aggregate CSV path for the batch run.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference size passed to YOLO.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Predictor IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per inference call.")
    parser.add_argument("--device", default="", help="Device string such as cpu, 0, or 0,1.")
    parser.add_argument(
        "--python",
        default=None,
        help="Interpreter to use; defaults to <repo>/.venv/bin/python if present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    return parser


def resolve_model_for_demo(raw_model) -> str:
    """Resolve the model input used for inference."""
    # Resolve the requested model before importing YOLO.
    # Resolve the user input into the exact model value YOLO should load.
    model_value = resolve_model_argument(raw_model)
    if is_url(model_value):
        return model_value

    # For local paths, fail early if the checkpoint is missing.
    model_path = Path(model_value).expanduser()
    if model_path.is_absolute() and not model_path.exists():
        if raw_model is None:
            raise FileNotFoundError(
                "No reusable YOLO11 segmentation checkpoint is available yet. "
                f"Expected {DEFAULT_FINETUNED_MODEL.resolve()} or {DEFAULT_PRETRAINED_MODEL.resolve()}. "
                "Run bootstrap_isaid_seg.py to download the pretrained checkpoint or "
                "train_isaid_seg.py to create best.pt."
            )
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    return model_value


def default_single_outputs(source, source_path) -> tuple[Path, Path, Path, Path]:
    # Derive one stable output stem from the local filename or URL.
    """Build the default output paths for one source."""
    if source_path is not None:
        # Reuse the local file stem and suffix for overlay naming.
        overlay_suffix = source_path.suffix.lower() or ".jpg"
        stem = source_path.stem
    else:
        # Use a generic stem for URLs and preserve any usable suffix.
        overlay_suffix = url_output_suffix(source)
        stem = "url_image"

    # Single-image mode writes four sibling artifacts.
    overlay = resolve_question_path(Path("outputs") / f"{stem}_overlay{overlay_suffix}")
    class_ids = resolve_question_path(Path("outputs") / f"{stem}_class_ids.png")
    mask = resolve_question_path(Path("outputs") / f"{stem}_mask.png")
    summary = resolve_question_path(Path("outputs") / f"{stem}_summary.json")
    return overlay, class_ids, mask, summary


def write_json(path, data) -> Path:
    # Create the parent directory and write one formatted JSON file.
    """Write one JSON file."""
    path = path.resolve()
    # Make sure the output folder exists before writing.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write indented JSON so the summary stays easy to read.
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_csv(path, rows) -> Path:
    # Flatten the per-image summaries into one batch index file.
    """Write one CSV file."""
    path = path.resolve()
    # Create the destination folder before opening the CSV file.
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_path",
        "status",
        "error_message",
        "image_height",
        "image_width",
        "requested_keep_classes",
        "resolved_keep_class_ids",
        "detected_instances",
        "classes_present_json",
        "pixel_counts_by_class_json",
        "instance_counts_by_class_json",
        "overlay_path",
        "class_ids_path",
        "mask_path",
        "summary_json_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        # Write the header row first.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        # Then write one row per image summary.
        writer.writerows(rows)
    return path


def row_from_summary(summary, summary_path) -> dict[str, object]:
    # Pull the summary fields into the fixed CSV schema.
    """Convert one summary into a CSV row."""
    # Read the optional image shape from the summary payload.
    image_shape = summary.get("image_shape") or []
    height = image_shape[0] if len(image_shape) == 2 else ""
    width = image_shape[1] if len(image_shape) == 2 else ""
    detected_instances = summary.get("detected_instances", "")
    # Clear the detection count on error rows.
    if summary.get("status") != "ok":
        detected_instances = ""

    # Return one flat row that matches the CSV schema exactly.
    return {
        "input_path": summary.get("source", ""),
        "status": summary.get("status", ""),
        "error_message": summary.get("error_message", ""),
        "image_height": height,
        "image_width": width,
        "requested_keep_classes": summary.get("requested_keep_classes", ""),
        "resolved_keep_class_ids": json.dumps(summary.get("resolved_keep_class_ids", [])),
        "detected_instances": detected_instances,
        "classes_present_json": json.dumps(summary.get("classes_present", []), sort_keys=True),
        "pixel_counts_by_class_json": json.dumps(summary.get("pixel_counts_by_class", {}), sort_keys=True),
        "instance_counts_by_class_json": json.dumps(summary.get("instance_counts_by_class", {}), sort_keys=True),
        "overlay_path": summary.get("overlay_image", ""),
        "class_ids_path": summary.get("class_ids_image", ""),
        "mask_path": summary.get("semantic_mask", ""),
        "summary_json_path": str(summary_path.resolve()),
    }


def build_predict_kwargs(args) -> dict[str, object]:
    # Collect the predictor options once and pass them through to every image.
    """Build the YOLO predict options."""
    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "save": False,
        "retina_masks": True,
        "verbose": False,
    }
    # Pass the device only when the caller explicitly set one.
    if args.device:
        predict_kwargs["device"] = args.device
    return predict_kwargs


def extract_instances(result, name_map, keep_ids, np_module) -> list[dict[str, object]]:
    # A segmentation result needs both boxes and masks to produce semantic outputs.
    """Extract the kept instances from one result."""
    # Return no detections when either boxes or masks are missing.
    if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
        return []

    # Move the model tensors to CPU numpy arrays for filtering and serialization.
    xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    masks = result.masks.data.cpu().numpy()

    detections = []
    for box, confidence, class_id, mask in zip(xyxy, confidences, class_ids, masks):
        # Skip detections outside the requested class filter.
        if keep_ids is not None and class_id not in keep_ids:
            continue

        # Convert the predicted mask into a boolean map.
        binary_mask = mask > 0.5
        pixel_count = int(binary_mask.sum())
        # Skip empty masks after thresholding.
        if pixel_count == 0:
            continue

        # Keep the fields needed for rendering, summaries, and per-pixel fusion.
        detections.append(
            {
                "class_id": int(class_id),
                "output_class_id": int(class_id) + 1,
                "class_name": name_map.get(int(class_id), str(class_id)),
                "confidence": float(confidence),
                "bbox_xyxy": [round(float(value), 3) for value in box.tolist()],
                "mask_pixels": pixel_count,
                "mask": binary_mask.astype(np_module.bool_),
            }
        )

    # Sort by confidence so later stages can process the strongest detections first.
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def build_semantic_map(image_shape, detections, np_module):
    # Start with background everywhere.
    """Build the per-pixel semantic class map."""
    # Allocate the output class map with background id 0.
    class_map = np_module.zeros(image_shape, dtype=np_module.uint8)
    # Track the strongest confidence assigned to each pixel so overlaps can be resolved.
    confidence_map = np_module.full(image_shape, -1.0, dtype=np_module.float32)

    for det in detections:
        # Let the highest-confidence detection own each overlapping pixel.
        mask = det["mask"]
        update = mask & (det["confidence"] >= confidence_map)
        # Write the class id anywhere this detection wins the overlap test.
        class_map[update] = det["output_class_id"]
        # Update the winning confidence map at the same pixels.
        confidence_map[update] = det["confidence"]

    return class_map


def colorize_class_map(class_map, np_module):
    # Expand the one-channel class map into a color image for visualization.
    """Colorize the semantic class map."""
    mask = np_module.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np_module.uint8)
    for pixel_value in sorted(int(value) for value in np_module.unique(class_map) if int(value) > 0):
        # Paint every pixel that belongs to the current class with its display color.
        mask[class_map == pixel_value] = class_color(pixel_value - 1)
    return mask


def render_outputs(
    image_bgr,
    class_map,
    detections,
    overlay_path,
    class_ids_path,
    mask_path,
    cv2_module,
    np_module,
) -> None:
    # Build the color mask first, then blend it with the original image.
    """Write the overlay, class-id map, and color mask."""
    color_mask = colorize_class_map(class_map, np_module)
    # Blend the color mask over the source image to create the overlay.
    overlay = cv2_module.addWeighted(color_mask, 0.35, image_bgr.copy(), 0.65, 0)

    for det in detections:
        # Draw one box and text label for each retained detection.
        x1, y1, x2, y2 = [int(round(value)) for value in det["bbox_xyxy"]]
        color = class_color(int(det["class_id"]))
        # Draw the bounding box first.
        cv2_module.rectangle(overlay, (x1, y1), (x2, y2), color, 2, lineType=cv2_module.LINE_AA)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        # Measure the label text so the background box fits it.
        (text_w, text_h), baseline = cv2_module.getTextSize(
            label, cv2_module.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        box_x2 = min(x1 + text_w + 6, overlay.shape[1] - 1)
        text_y = max(y1 - 6, text_h + baseline + 4)
        # Draw the filled text background box.
        cv2_module.rectangle(
            overlay,
            (x1, text_y - text_h - baseline - 4),
            (box_x2, text_y),
            color,
            thickness=-1,
        )
        # Draw the label text on top of the filled background.
        cv2_module.putText(
            overlay,
            label,
            (x1 + 3, text_y - baseline - 2),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2_module.LINE_AA,
        )

    # Make sure every destination directory exists before writing the images.
    ensure_parent(overlay_path)
    ensure_parent(class_ids_path)
    ensure_parent(mask_path)
    # Write the overlay, raw class-id map, and color mask as separate files.
    if not cv2_module.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to write overlay image: {overlay_path}")
    if not cv2_module.imwrite(str(class_ids_path), class_map):
        raise RuntimeError(f"Failed to write class id image: {class_ids_path}")
    if not cv2_module.imwrite(str(mask_path), color_mask):
        raise RuntimeError(f"Failed to write semantic mask: {mask_path}")


def classes_present(class_map, name_map, np_module) -> list[dict[str, object]]:
    # Report each non-background class id that appears in the semantic map.
    """List the classes that appear in the class map."""
    present = []
    for pixel_value in sorted(int(value) for value in np_module.unique(class_map) if int(value) > 0):
        # Convert the stored output-class id back to the model's zero-based class id.
        class_id = pixel_value - 1
        present.append(
            {
                "class_id": class_id,
                "output_class_id": pixel_value,
                "class_name": name_map.get(class_id, str(class_id)),
            }
        )
    return present


def pixel_counts_by_class(class_map, name_map, np_module) -> dict[str, int]:
    # Count how many pixels belong to each class in the final semantic map.
    """Count the pixels for each class."""
    counts = {}
    for pixel_value in sorted(int(value) for value in np_module.unique(class_map) if int(value) > 0):
        # Convert the stored output-class id back to the model class id.
        class_id = pixel_value - 1
        # Count pixels for this class in the final fused class map.
        counts[name_map.get(class_id, str(class_id))] = int((class_map == pixel_value).sum())
    return counts


def serializable_detections(detections) -> list[dict[str, object]]:
    # Drop the raw boolean mask arrays before writing detections into JSON.
    """Drop raw mask arrays from the detections."""
    # Keep only JSON-friendly scalar fields for each detection.
    return [
        {
            "class_id": det["class_id"],
            "output_class_id": det["output_class_id"],
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 6),
            "bbox_xyxy": det["bbox_xyxy"],
            "mask_pixels": det["mask_pixels"],
        }
        for det in detections
    ]


def success_summary(
    *,
    model_label,
    source_label,
    image_shape,
    requested_keep_classes,
    keep_ids,
    overlay_path,
    class_ids_path,
    mask_path,
    name_map,
    class_map,
    detections,
    np_module,
) -> dict[str, object]:
    # Build one success payload with counts, artifact paths, and per-instance details.
    """Build the success summary payload."""
    # Count how many detections survived for each class name.
    instance_counts = Counter(det["class_name"] for det in detections)
    # Build the output-class-id to class-name map used by the saved class map.
    class_id_map = {str(class_id + 1): name for class_id, name in sorted(name_map.items())}
    return {
        "status": "ok",
        "error_message": "",
        "task": "segment",
        "model": model_label,
        "source": source_label,
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "requested_keep_classes": requested_keep_classes,
        "resolved_keep_class_ids": sorted(list(keep_ids)) if keep_ids else [],
        "detected_instances": len(detections),
        "class_id_map": class_id_map,
        "classes_present": classes_present(class_map, name_map, np_module),
        "pixel_counts_by_class": pixel_counts_by_class(class_map, name_map, np_module),
        "instance_counts_by_class": dict(sorted(instance_counts.items())),
        "overlay_image": str(overlay_path.resolve()),
        "class_ids_image": str(class_ids_path.resolve()),
        "semantic_mask": str(mask_path.resolve()),
        "detections": serializable_detections(detections),
    }


def error_summary(
    *,
    model_label,
    source_label,
    requested_keep_classes,
    keep_ids,
    error_message,
) -> dict[str, object]:
    # Keep the error payload in the same shape as the success payload.
    """Build the error summary payload."""
    # Return the same keys as the success payload with empty artifact fields.
    return {
        "status": "error",
        "error_message": error_message,
        "task": "segment",
        "model": model_label,
        "source": source_label,
        "image_shape": [],
        "requested_keep_classes": requested_keep_classes,
        "resolved_keep_class_ids": sorted(list(keep_ids)) if keep_ids else [],
        "detected_instances": 0,
        "class_id_map": {},
        "classes_present": [],
        "pixel_counts_by_class": {},
        "instance_counts_by_class": {},
        "overlay_image": "",
        "class_ids_image": "",
        "semantic_mask": "",
        "detections": [],
    }


def run_predict(model, source, predict_kwargs):
    """Run one predict call and return the first result."""
    # Ultralytics returns a list of results even for one source.
    # Run predict() once with the prepared keyword arguments.
    results = model.predict(source=source, **predict_kwargs)
    if not results:
        raise RuntimeError("No prediction results were returned.")
    # Use the first result because the caller passes one source at a time.
    return results[0]


def process_one_source(
    model,
    source_label,
    args,
    keep_ids,
    name_map,
    predict_kwargs,
    overlay_path,
    class_ids_path,
    mask_path,
    *,
    cv2_module,
    np_module,
) -> dict[str, object]:
    # Run one prediction and pull the original image back out of the result object.
    """Process one source image from prediction to outputs."""
    result = run_predict(model, source_label, predict_kwargs)
    image = result.orig_img
    # Fail if Ultralytics did not preserve the original image array.
    if image is None:
        raise RuntimeError("The model did not return an original image.")

    # Convert detections into a per-pixel semantic map and write the image artifacts.
    detections = extract_instances(result, name_map, keep_ids, np_module)
    class_map = build_semantic_map(image.shape[:2], detections, np_module)
    render_outputs(
        image.copy(),
        class_map,
        detections,
        overlay_path,
        class_ids_path,
        mask_path,
        cv2_module,
        np_module,
    )

    # Return the success payload after all image artifacts have been written.
    return success_summary(
        model_label=str(args.resolved_model),
        source_label=source_label,
        image_shape=image.shape[:2],
        requested_keep_classes=args.keep_classes,
        keep_ids=keep_ids,
        overlay_path=overlay_path,
        class_ids_path=class_ids_path,
        mask_path=mask_path,
        name_map=name_map,
        class_map=class_map,
        detections=detections,
        np_module=np_module,
    )


def run_single_mode(
    model,
    source,
    args,
    keep_ids,
    name_map,
    predict_kwargs,
    *,
    cv2_module,
    np_module,
) -> int:
    # Start with the default output paths, then apply any explicit overrides.
    """Run the single-image flow."""
    source_path = None if is_url(source) else Path(source)
    overlay_path, class_ids_path, mask_path, summary_path = default_single_outputs(source, source_path)
    # Apply any caller-provided output overrides one by one.
    if args.output_overlay:
        overlay_path = resolve_question_path(args.output_overlay)
    if args.output_class_ids:
        class_ids_path = resolve_question_path(args.output_class_ids)
    if args.output_mask:
        mask_path = resolve_question_path(args.output_mask)
    if args.summary_json:
        summary_path = resolve_question_path(args.summary_json)

    # Process the image, then write the companion JSON summary.
    summary = process_one_source(
        model,
        source,
        args,
        keep_ids,
        name_map,
        predict_kwargs,
        overlay_path,
        class_ids_path,
        mask_path,
        cv2_module=cv2_module,
        np_module=np_module,
    )
    write_json(summary_path, summary)

    # Print the four artifact locations for the caller.
    print(f"Overlay:   {overlay_path.resolve()}")
    print(f"Class ids: {class_ids_path.resolve()}")
    print(f"Mask:      {mask_path.resolve()}")
    print(f"Summary:   {summary_path.resolve()}")
    return 0


def run_directory_mode(
    model,
    source_dir,
    args,
    keep_ids,
    name_map,
    predict_kwargs,
    *,
    cv2_module,
    np_module,
) -> int:
    # Collect every supported image under the source directory.
    """Run the directory batch flow."""
    images = list_supported_images(source_dir)
    if not images:
        raise FileNotFoundError(
            f"No supported image files were found under {source_dir}. "
            "Add .jpg, .jpeg, .png, .bmp, .webp, .tif, or .tiff files."
        )

    # Resolve the batch output locations once before the loop starts.
    output_root = resolve_question_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT.resolve()
    index_csv = resolve_question_path(args.index_csv) if args.index_csv else DEFAULT_INDEX_CSV.resolve()

    rows = []
    succeeded = 0
    failed = 0

    print(f"Processing {len(images)} image(s) from {source_dir}")
    print(f"Writing per-image results under {output_root}")

    for image_path in images:
        # Each image gets its own output subdirectory and summary file.
        image_output_dir = per_image_output_dir(source_dir, image_path, output_root)
        overlay_path = image_output_dir / f"overlay{image_path.suffix.lower()}"
        class_ids_path = image_output_dir / "class_ids.png"
        mask_path = image_output_dir / "mask.png"
        summary_path = image_output_dir / "summary.json"

        try:
            # Successful images produce all artifacts plus a success summary.
            summary = process_one_source(
                model,
                str(image_path),
                args,
                keep_ids,
                name_map,
                predict_kwargs,
                overlay_path,
                class_ids_path,
                mask_path,
                cv2_module=cv2_module,
                np_module=np_module,
            )
            succeeded += 1
        except Exception as exc:
            # Failed images still produce a summary row so the batch index stays complete.
            summary = error_summary(
                model_label=str(args.resolved_model),
                source_label=str(image_path),
                requested_keep_classes=args.keep_classes,
                keep_ids=keep_ids,
                error_message=str(exc),
            )
            failed += 1

        # Write the per-image summary file regardless of success or failure.
        written_summary_path = write_json(summary_path, summary)
        # Convert that summary into one flat CSV row.
        rows.append(row_from_summary(summary, written_summary_path))

    # Write the aggregate CSV after every per-image summary is finished.
    write_csv(index_csv, rows)
    print(f"Succeeded: {succeeded}")
    print(f"Failed:    {failed}")
    print(f"Index CSV: {index_csv.resolve()}")
    return 1 if failed else 0


def main() -> int:
    """Run the segmentation inference flow."""
    parser = build_parser()
    args = parser.parse_args()

    # Resolve the repo, model, and source paths before importing heavy dependencies.
    repo_dir = ensure_ultralytics_repo(resolve_repo_dir(args.repo_dir))

    args.resolved_model = (
        resolve_model_argument(args.model) if args.dry_run else resolve_model_for_demo(args.model)
    )
    resolved_source, source_path = resolve_source(args.source)
    args.resolved_source = resolved_source

    # Directory mode uses one output root, while single-image mode uses explicit files.
    is_directory_mode = source_path is not None and source_path.is_dir()
    # Reject single-image output flags when the source is a directory.
    if is_directory_mode and any(
        value is not None
        for value in (args.output_overlay, args.output_class_ids, args.output_mask, args.summary_json)
    ):
        raise ValueError(
            "When --source is a directory, do not pass --output-overlay, --output-class-ids, "
            "--output-mask, or --summary-json. Use --output-dir and --index-csv instead."
        )

    if source_path is None and (args.output_dir or args.index_csv):
        raise ValueError("URL sources run in single-image mode. Use the single-image output flags instead.")

    if args.dry_run:
        # Print the resolved execution plan without importing YOLO or OpenCV.
        print(f"Repo:   {repo_dir}")
        print(f"Model:  {args.resolved_model}")
        print(f"Source: {resolved_source}")
        if is_directory_mode:
            output_root = resolve_question_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT.resolve()
            index_csv = resolve_question_path(args.index_csv) if args.index_csv else DEFAULT_INDEX_CSV.resolve()
            print("Mode:   dataset")
            print(f"Out:    {output_root}")
            print(f"Index:  {index_csv}")
        else:
            # Recompute the single-image defaults so dry-run can show the final file paths.
            overlay_path, class_ids_path, mask_path, summary_path = default_single_outputs(
                resolved_source, source_path
            )
            if args.output_overlay:
                overlay_path = resolve_question_path(args.output_overlay)
            if args.output_class_ids:
                class_ids_path = resolve_question_path(args.output_class_ids)
            if args.output_mask:
                mask_path = resolve_question_path(args.output_mask)
            if args.summary_json:
                summary_path = resolve_question_path(args.summary_json)
            print("Mode:   single")
            print(f"Overlay:{overlay_path}")
            print(f"ClassId:{class_ids_path}")
            print(f"Mask:   {mask_path}")
            print(f"Summary:{summary_path}")
        return 0

    # Re-enter under the repository interpreter before importing runtime dependencies.
    maybe_reexec_with_repo_python(repo_dir, args.python, REEXEC_MARKER)
    ensure_ultralytics_import(repo_dir)

    # Import the runtime libraries only after the interpreter is finalized.
    import cv2
    import numpy as np
    from ultralytics import YOLO

    print(f"Model checkpoint: {args.resolved_model}")
    # Load the model and resolve any requested class filter.
    model = YOLO(args.resolved_model)
    name_map = normalize_name_map(model.names)
    keep_ids = parse_keep_classes(args.keep_classes, name_map)
    predict_kwargs = build_predict_kwargs(args)
    if keep_ids is None:
        print("Class filter: all classes")
    else:
        print(f"Class filter: {args.keep_classes}")

    # Dispatch to batch or single-image mode with the same shared helpers.
    if is_directory_mode:
        # Directory mode processes every image under the source tree.
        return run_directory_mode(
            model,
            source_path,
            args,
            keep_ids,
            name_map,
            predict_kwargs,
            cv2_module=cv2,
            np_module=np,
        )

    # Single mode processes one path or one URL source.
    print(f"Processing single source: {resolved_source}")
    return run_single_mode(
        model,
        resolved_source,
        args,
        keep_ids,
        name_map,
        predict_kwargs,
        cv2_module=cv2,
        np_module=np,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Return the standard shell exit code for Ctrl+C.
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        # Print one final CLI error line for uncaught failures.
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
