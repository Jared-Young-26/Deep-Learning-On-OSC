#!/usr/bin/env python3
"""Inference-only YOLO11 OBB demo for DOTA-style satellite imagery."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from q5_obb_common import (
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

REEXEC_MARKER = "Q5_OBB_DEMO_INNER"


# The CLI exposes only the controls needed for the assignment demo: model choice,
# class filtering, tiling, and output layout.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run reusable YOLO11 OBB inference for DOTA-style satellite images. "
            "The demo renders oriented detections into both an overlay image and a mask-like PNG."
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
            "question_5_semantic_segmentation/models/dota_obb/best.pt and falls back to "
            "question_5_semantic_segmentation/models/pretrained/yolo11s-obb.pt."
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
            "Optional comma-separated DOTA class ids/names to keep. "
            "Omit this flag to detect all DOTA classes. "
            "Example: ship,plane,small vehicle,large vehicle"
        ),
    )
    parser.add_argument(
        "--output-overlay",
        default=None,
        help="Single-image only. Output path for the rotated-box overlay image.",
    )
    parser.add_argument(
        "--output-mask",
        default=None,
        help="Single-image only. Output path for the class-rendered mask PNG.",
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
            "containing overlay, mask, and summary files."
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
    parser.add_argument(
        "--tile-threshold",
        type=int,
        default=1536,
        help="Tile local images whose width or height exceeds this value.",
    )
    parser.add_argument("--tile-size", type=int, default=1024, help="Sliding-window crop size for large images.")
    parser.add_argument("--tile-gap", type=int, default=200, help="Sliding-window overlap gap for large images.")
    parser.add_argument(
        "--tile-merge-iou",
        type=float,
        default=0.35,
        help="Rotated NMS IoU threshold for merging detections across tiles.",
    )
    parser.add_argument("--device", default="", help="Device string such as cpu, 0, or 0,1.")
    parser.add_argument(
        "--python",
        default=None,
        help="Interpreter to use; defaults to <repo>/.venv/bin/python if present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit.")
    return parser


def resolve_model_for_demo(raw_model: str | None) -> str:
    # Resolve the demo checkpoint with a bias toward the fine-tuned alias, but
    # fail with a helpful message when neither the trained nor pretrained file exists.
    model_value = resolve_model_argument(raw_model)
    if is_url(model_value):
        return model_value

    model_path = Path(model_value).expanduser()
    if model_path.is_absolute() and not model_path.exists():
        if raw_model is None:
            raise FileNotFoundError(
                "No reusable YOLO11 OBB checkpoint is available yet. "
                f"Expected {DEFAULT_FINETUNED_MODEL.resolve()} or {DEFAULT_PRETRAINED_MODEL.resolve()}. "
                "Run bootstrap_dota_obb.py to download the pretrained checkpoint or train_dota_obb.py to create best.pt."
            )
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    return model_value


def default_single_outputs(source: str, source_path: Path | None) -> tuple[Path, Path, Path]:
    # Single-image mode uses one predictable trio of outputs: overlay, mask, and summary.
    if source_path is not None:
        overlay_suffix = source_path.suffix.lower() or ".jpg"
        stem = source_path.stem
    else:
        overlay_suffix = url_output_suffix(source)
        stem = "url_image"

    overlay = resolve_question_path(Path("outputs") / f"{stem}_overlay{overlay_suffix}")
    mask = resolve_question_path(Path("outputs") / f"{stem}_mask.png")
    summary = resolve_question_path(Path("outputs") / f"{stem}_summary.json")
    return overlay, mask, summary


def write_json(path: Path, data: dict[str, object]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    # The batch CSV keeps only the top-level metrics so multiple images can be
    # skimmed quickly without opening every JSON summary.
    path = path.resolve()
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
        "counts_by_class_json",
        "overlay_path",
        "mask_path",
        "summary_json_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def row_from_summary(summary: dict[str, object], summary_path: Path) -> dict[str, object]:
    # Flatten each rich JSON summary into one table row for the aggregate CSV report.
    image_shape = summary.get("image_shape") or []
    height = image_shape[0] if len(image_shape) == 2 else ""
    width = image_shape[1] if len(image_shape) == 2 else ""
    detected_instances = summary.get("detected_instances", "")
    if summary.get("status") != "ok":
        detected_instances = ""

    return {
        "input_path": summary.get("source", ""),
        "status": summary.get("status", ""),
        "error_message": summary.get("error_message", ""),
        "image_height": height,
        "image_width": width,
        "requested_keep_classes": summary.get("requested_keep_classes", ""),
        "resolved_keep_class_ids": json.dumps(summary.get("resolved_keep_class_ids", [])),
        "detected_instances": detected_instances,
        "counts_by_class_json": json.dumps(summary.get("counts_by_class", {}), sort_keys=True),
        "overlay_path": summary.get("overlay_image", ""),
        "mask_path": summary.get("semantic_style_mask", ""),
        "summary_json_path": str(summary_path.resolve()),
    }


# These helpers convert raw OBB results into the simpler overlay/mask/report
# artifacts that are easiest to explain in a professor walkthrough.
def build_predict_kwargs(args: argparse.Namespace) -> dict[str, object]:
    predict_kwargs: dict[str, object] = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "save": False,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device
    return predict_kwargs


def load_local_image(path: Path, cv2_module):
    # OpenCV reads BGR arrays directly, which is what the later rendering helper expects.
    image = cv2_module.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def result_to_detections(result, name_map: dict[int, str], keep_ids: set[int] | None) -> list[dict[str, object]]:
    # Convert the Ultralytics OBB result object into plain Python dicts so the rest
    # of the pipeline can summarize, merge, and serialize detections consistently.
    if result.obb is None or len(result.obb) == 0:
        return []

    xywhr = result.obb.xywhr.cpu().numpy()
    confidences = result.obb.conf.cpu().numpy()
    class_ids = result.obb.cls.cpu().numpy().astype(int)

    detections: list[dict[str, object]] = []
    for box, confidence, class_id in zip(xywhr, confidences, class_ids):
        if keep_ids is not None and class_id not in keep_ids:
            continue
        detections.append(
            {
                "class_id": int(class_id),
                "class_name": name_map.get(int(class_id), str(class_id)),
                "confidence": float(confidence),
                "xywhr": [float(value) for value in box.tolist()],
            }
        )
    return detections


def merge_tile_detections(detections, image_shape, merge_iou, torch_module, batch_probiou, torch_nms):
    if not detections:
        return []

    # Offset boxes by class before NMS so detections from different classes do not
    # suppress each other when tiles overlap.
    boxes = torch_module.tensor([det["xywhr"] for det in detections], dtype=torch_module.float32)
    scores = torch_module.tensor([det["confidence"] for det in detections], dtype=torch_module.float32)
    class_ids = torch_module.tensor([det["class_id"] for det in detections], dtype=torch_module.float32)

    class_offset = float(max(image_shape) + 2048)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, 0] += class_ids * class_offset
    boxes_for_nms[:, 1] += class_ids * class_offset

    keep = torch_nms.fast_nms(boxes_for_nms, scores, merge_iou, iou_func=batch_probiou)
    keep_indices = keep.cpu().tolist()
    return [detections[int(index)] for index in keep_indices]


def attach_polygons(detections, ops_module, torch_module):
    if not detections:
        return detections

    # Convert YOLO's center/width/height/rotation representation into explicit corner
    # polygons because overlays, masks, and JSON reports are easier to inspect that way.
    boxes = torch_module.tensor([det["xywhr"] for det in detections], dtype=torch_module.float32)
    polygons = ops_module.xywhr2xyxyxyxy(boxes).cpu().numpy()
    for det, polygon in zip(detections, polygons):
        det["polygon"] = [[round(float(x), 3), round(float(y), 3)] for x, y in polygon.tolist()]
    return detections


def render_overlay_and_mask(image_bgr, detections, overlay_path: Path, mask_path: Path, cv2_module, np_module) -> None:
    # The overlay is the presentation artifact, while the mask is a simpler class-colored
    # region map that looks more like a segmentation output.
    overlay = image_bgr.copy()
    fill_layer = image_bgr.copy()
    mask = np_module.zeros_like(image_bgr)

    for det in detections:
        polygon = np_module.round(np_module.asarray(det["polygon"], dtype=np_module.float32)).astype(np_module.int32)
        color = class_color(int(det["class_id"]))
        cv2_module.fillPoly(fill_layer, [polygon], color)
        cv2_module.fillPoly(mask, [polygon], color)

    overlay = cv2_module.addWeighted(fill_layer, 0.22, overlay, 0.78, 0)

    for det in detections:
        polygon = np_module.round(np_module.asarray(det["polygon"], dtype=np_module.float32)).astype(np_module.int32)
        color = class_color(int(det["class_id"]))
        cv2_module.polylines(overlay, [polygon], True, color, 2, lineType=cv2_module.LINE_AA)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        anchor = polygon[0]
        (text_w, text_h), baseline = cv2_module.getTextSize(
            label, cv2_module.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        x = max(int(anchor[0]), 0)
        y = max(int(anchor[1]) - 8, text_h + baseline + 4)
        x2 = min(x + text_w + 6, overlay.shape[1] - 1)
        y2 = y + baseline + 4
        cv2_module.rectangle(overlay, (x, y - text_h - baseline - 4), (x2, y2), color, thickness=-1)
        cv2_module.putText(
            overlay,
            label,
            (x + 3, y - baseline - 2),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2_module.LINE_AA,
        )

    ensure_parent(overlay_path)
    ensure_parent(mask_path)
    if not cv2_module.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to write overlay image: {overlay_path}")
    if not cv2_module.imwrite(str(mask_path), mask):
        raise RuntimeError(f"Failed to write mask image: {mask_path}")


def success_summary(
    *,
    model_label: str,
    source_label: str,
    image_shape: tuple[int, int],
    requested_keep_classes: str,
    keep_ids: set[int] | None,
    overlay_path: Path,
    mask_path: Path,
    detections: list[dict[str, object]],
    tiled_inference: bool,
    tiles_evaluated: int,
) -> dict[str, object]:
    # The JSON summary preserves both high-level counts and the raw per-box data
    # so you can explain either the overview or a specific detection.
    counts = Counter(det["class_name"] for det in detections)
    return {
        "status": "ok",
        "error_message": "",
        "task": "obb",
        "model": model_label,
        "source": source_label,
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "requested_keep_classes": requested_keep_classes,
        "resolved_keep_class_ids": sorted(list(keep_ids)) if keep_ids else [],
        "detected_instances": len(detections),
        "counts_by_class": dict(sorted(counts.items())),
        "overlay_image": str(overlay_path.resolve()),
        "semantic_style_mask": str(mask_path.resolve()),
        "tiled_inference": tiled_inference,
        "tiles_evaluated": int(tiles_evaluated),
        "detections": detections,
    }


def error_summary(
    *,
    model_label: str,
    source_label: str,
    requested_keep_classes: str,
    keep_ids: set[int] | None,
    error_message: str,
) -> dict[str, object]:
    # Batch mode records failures as structured summaries too, so one bad image does
    # not erase the rest of the run history.
    return {
        "status": "error",
        "error_message": error_message,
        "task": "obb",
        "model": model_label,
        "source": source_label,
        "image_shape": [],
        "requested_keep_classes": requested_keep_classes,
        "resolved_keep_class_ids": sorted(list(keep_ids)) if keep_ids else [],
        "detected_instances": 0,
        "counts_by_class": {},
        "overlay_image": "",
        "semantic_style_mask": "",
        "tiled_inference": False,
        "tiles_evaluated": 0,
        "detections": [],
    }


# Split local-image inference away from URL inference because only local files can
# be tiled and merged when the aerial image is too large for one pass.
def run_predict(model, source, predict_kwargs):
    # Wrap model.predict so the rest of the script can assume one result object exists.
    results = model.predict(source=source, **predict_kwargs)
    if not results:
        raise RuntimeError("No prediction results were returned.")
    return results[0]


def infer_local_path(
    model,
    image_path: Path,
    args: argparse.Namespace,
    keep_ids: set[int] | None,
    name_map: dict[int, str],
    predict_kwargs: dict[str, object],
    *,
    cv2_module,
    np_module,
    torch_module,
    get_windows,
    ops_module,
    batch_probiou,
    torch_nms,
):
    image = load_local_image(image_path, cv2_module)
    image_height, image_width = image.shape[:2]

    # Small images run in one pass; very large aerial images are tiled and merged
    # so fine objects are not missed at one global resize.
    if max(image_height, image_width) <= args.tile_threshold:
        result = run_predict(model, str(image_path), predict_kwargs)
        detections = result_to_detections(result, name_map, keep_ids)
        detections = attach_polygons(detections, ops_module, torch_module)
        return image, detections, False, 1

    windows = get_windows((image_height, image_width), crop_sizes=(args.tile_size,), gaps=(args.tile_gap,))
    detections = []
    for x_start, y_start, x_stop, y_stop in windows.tolist():
        # Shift each tile's detections back into full-image coordinates before merging.
        tile = image[y_start:y_stop, x_start:x_stop]
        result = run_predict(model, tile, predict_kwargs)
        tile_detections = result_to_detections(result, name_map, keep_ids)
        for det in tile_detections:
            det["xywhr"][0] += float(x_start)
            det["xywhr"][1] += float(y_start)
        detections.extend(tile_detections)

    detections = merge_tile_detections(
        detections,
        (image_height, image_width),
        args.tile_merge_iou,
        torch_module,
        batch_probiou,
        torch_nms,
    )
    detections = attach_polygons(detections, ops_module, torch_module)
    return image, detections, True, len(windows)


def infer_url_source(
    model,
    source: str,
    keep_ids: set[int] | None,
    name_map: dict[int, str],
    predict_kwargs: dict[str, object],
    *,
    cv2_module,
    torch_module,
    ops_module,
):
    # URL sources rely on Ultralytics to fetch and decode the image, then the wrapper
    # converts the result into the same detection structure used for local files.
    result = run_predict(model, source, predict_kwargs)
    image = result.orig_img
    if image is None:
        raise RuntimeError("The model did not return an original image for the URL source.")
    detections = result_to_detections(result, name_map, keep_ids)
    detections = attach_polygons(detections, ops_module, torch_module)
    return image.copy(), detections, False, 1


def process_one_source(
    model,
    source_label: str,
    source_path: Path | None,
    args: argparse.Namespace,
    keep_ids: set[int] | None,
    name_map: dict[int, str],
    predict_kwargs: dict[str, object],
    overlay_path: Path,
    mask_path: Path,
    *,
    cv2_module,
    np_module,
    torch_module,
    get_windows,
    ops_module,
    batch_probiou,
    torch_nms,
) -> dict[str, object]:
    # This is the one-image pipeline shared by both single-image mode and batch mode:
    # infer -> render overlay/mask -> build a JSON summary.
    if source_path is None:
        image, detections, tiled_inference, tiles_evaluated = infer_url_source(
            model,
            source_label,
            keep_ids,
            name_map,
            predict_kwargs,
            cv2_module=cv2_module,
            torch_module=torch_module,
            ops_module=ops_module,
        )
    else:
        image, detections, tiled_inference, tiles_evaluated = infer_local_path(
            model,
            source_path,
            args,
            keep_ids,
            name_map,
            predict_kwargs,
            cv2_module=cv2_module,
            np_module=np_module,
            torch_module=torch_module,
            get_windows=get_windows,
            ops_module=ops_module,
            batch_probiou=batch_probiou,
            torch_nms=torch_nms,
        )

    render_overlay_and_mask(image, detections, overlay_path, mask_path, cv2_module, np_module)
    return success_summary(
        model_label=str(args.resolved_model),
        source_label=source_label,
        image_shape=image.shape[:2],
        requested_keep_classes=args.keep_classes,
        keep_ids=keep_ids,
        overlay_path=overlay_path,
        mask_path=mask_path,
        detections=detections,
        tiled_inference=tiled_inference,
        tiles_evaluated=tiles_evaluated,
    )


# Single-image mode writes exactly one overlay, one mask, and one JSON summary.
def run_single_mode(
    model,
    source: str,
    source_path: Path | None,
    args: argparse.Namespace,
    keep_ids: set[int] | None,
    name_map: dict[int, str],
    predict_kwargs: dict[str, object],
    *,
    cv2_module,
    np_module,
    torch_module,
    get_windows,
    ops_module,
    batch_probiou,
    torch_nms,
) -> int:
    overlay_path, mask_path, summary_path = default_single_outputs(source, source_path)
    if args.output_overlay:
        overlay_path = resolve_question_path(args.output_overlay)
    if args.output_mask:
        mask_path = resolve_question_path(args.output_mask)
    if args.summary_json:
        summary_path = resolve_question_path(args.summary_json)

    summary = process_one_source(
        model,
        source,
        source_path,
        args,
        keep_ids,
        name_map,
        predict_kwargs,
        overlay_path,
        mask_path,
        cv2_module=cv2_module,
        np_module=np_module,
        torch_module=torch_module,
        get_windows=get_windows,
        ops_module=ops_module,
        batch_probiou=batch_probiou,
        torch_nms=torch_nms,
    )
    write_json(summary_path, summary)

    print(f"Overlay: {overlay_path.resolve()}")
    print(f"Mask:    {mask_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    return 0


# Directory mode repeats the same per-image pipeline and then adds one aggregate CSV
# so the whole batch can be summarized quickly.
def run_directory_mode(
    model,
    source_dir: Path,
    args: argparse.Namespace,
    keep_ids: set[int] | None,
    name_map: dict[int, str],
    predict_kwargs: dict[str, object],
    *,
    cv2_module,
    np_module,
    torch_module,
    get_windows,
    ops_module,
    batch_probiou,
    torch_nms,
) -> int:
    images = list_supported_images(source_dir)
    if not images:
        raise FileNotFoundError(
            f"No supported image files were found under {source_dir}. "
            "Add .jpg, .jpeg, .png, .bmp, .webp, .tif, or .tiff files."
        )

    output_root = resolve_question_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT.resolve()
    index_csv = resolve_question_path(args.index_csv) if args.index_csv else DEFAULT_INDEX_CSV.resolve()

    rows = []
    succeeded = 0
    failed = 0

    print(f"Processing {len(images)} image(s) from {source_dir}")
    print(f"Writing per-image results under {output_root}")

    for image_path in images:
        # Give each image its own output folder so overlays, masks, and summaries
        # stay grouped together even for nested input directories.
        image_output_dir = per_image_output_dir(source_dir, image_path, output_root)
        overlay_path = image_output_dir / f"overlay{image_path.suffix.lower()}"
        mask_path = image_output_dir / "mask.png"
        summary_path = image_output_dir / "summary.json"

        try:
            summary = process_one_source(
                model,
                str(image_path),
                image_path,
                args,
                keep_ids,
                name_map,
                predict_kwargs,
                overlay_path,
                mask_path,
                cv2_module=cv2_module,
                np_module=np_module,
                torch_module=torch_module,
                get_windows=get_windows,
                ops_module=ops_module,
                batch_probiou=batch_probiou,
                torch_nms=torch_nms,
            )
            succeeded += 1
        except Exception as exc:
            summary = error_summary(
                model_label=str(args.resolved_model),
                source_label=str(image_path),
                requested_keep_classes=args.keep_classes,
                keep_ids=keep_ids,
                error_message=str(exc),
            )
            failed += 1

        written_summary_path = write_json(summary_path, summary)
        rows.append(row_from_summary(summary, written_summary_path))

    write_csv(index_csv, rows)
    print(f"Succeeded: {succeeded}")
    print(f"Failed:    {failed}")
    print(f"Index CSV: {index_csv.resolve()}")
    return 1 if failed else 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Re-enter through the repo-local interpreter so inference always uses the
    # prepared Ultralytics environment and checkpoint resolution logic.
    repo_dir = ensure_ultralytics_repo(resolve_repo_dir(args.repo_dir))
    maybe_reexec_with_repo_python(repo_dir, args.python, REEXEC_MARKER)
    ensure_ultralytics_import(repo_dir)

    args.resolved_model = (
        resolve_model_argument(args.model) if args.dry_run else resolve_model_for_demo(args.model)
    )
    resolved_source, source_path = resolve_source(args.source)
    args.resolved_source = resolved_source

    # The wrapper has two modes only: single-image or directory/batch. Validate the
    # output arguments here so downstream code can assume the mode is consistent.
    is_directory_mode = source_path is not None and source_path.is_dir()
    if is_directory_mode and any(
        value is not None for value in (args.output_overlay, args.output_mask, args.summary_json)
    ):
        raise ValueError(
            "When --source is a directory, do not pass --output-overlay, --output-mask, or --summary-json. "
            "Use --output-dir and --index-csv instead."
        )

    if source_path is None and (args.output_dir or args.index_csv):
        raise ValueError("URL sources run in single-image mode. Use --output-overlay, --output-mask, and --summary-json.")

    if args.dry_run:
        # Dry-run prints the fully resolved paths without importing the heavy ML stack.
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
            overlay_path, mask_path, summary_path = default_single_outputs(resolved_source, source_path)
            if args.output_overlay:
                overlay_path = resolve_question_path(args.output_overlay)
            if args.output_mask:
                mask_path = resolve_question_path(args.output_mask)
            if args.summary_json:
                summary_path = resolve_question_path(args.summary_json)
            print("Mode:   single")
            print(f"Overlay:{overlay_path}")
            print(f"Mask:   {mask_path}")
            print(f"Summary:{summary_path}")
        return 0

    # Import the heavy runtime modules only after CLI validation so --help and
    # --dry-run stay lightweight.
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.split_dota import get_windows
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import batch_probiou
    from ultralytics.utils.nms import TorchNMS

    # Resolve class names from the loaded checkpoint once so every downstream
    # detection summary and rendered label uses the same mapping.
    print(f"Model checkpoint: {args.resolved_model}")
    model = YOLO(args.resolved_model)
    name_map = normalize_name_map(model.names)
    keep_ids = parse_keep_classes(args.keep_classes, name_map)
    predict_kwargs = build_predict_kwargs(args)
    if keep_ids is None:
        print("Class filter: all DOTA classes")
    else:
        print(f"Class filter: {args.keep_classes}")

    if is_directory_mode:
        return run_directory_mode(
            model,
            source_path,
            args,
            keep_ids,
            name_map,
            predict_kwargs,
            cv2_module=cv2,
            np_module=np,
            torch_module=torch,
            get_windows=get_windows,
            ops_module=ops,
            batch_probiou=batch_probiou,
            torch_nms=TorchNMS,
        )

    print(f"Processing single source: {resolved_source}")
    return run_single_mode(
        model,
        resolved_source,
        source_path,
        args,
        keep_ids,
        name_map,
        predict_kwargs,
        cv2_module=cv2,
        np_module=np,
        torch_module=torch,
        get_windows=get_windows,
        ops_module=ops,
        batch_probiou=batch_probiou,
        torch_nms=TorchNMS,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
