#!/usr/bin/env python3
"""Apply directional convolution filters to one image or a directory of images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageOps

from convolution_filter_library import (
    apply_all_directional_filters,
    matrix_stats,
    normalize_matrix_abs_to_uint8,
)

QUESTION_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = QUESTION_DIR / "inputs"
DEFAULT_OUTPUT_DIR = QUESTION_DIR / "outputs"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
THRESHOLD = 128

FILTER_ORDER = [
    "horizontal",
    "vertical",
    "upper_left",
    "lower_right",
    "upper_right",
    "lower_left",
]

GRID_ORDER = [
    "original",
    "preprocessed_bw",
    "horizontal",
    "vertical",
    "upper_left",
    "lower_right",
    "upper_right",
    "lower_left",
]

Matrix = List[List[float]]
UInt8Matrix = List[List[int]]


# The image demo is intentionally simple: resolve input files, run preprocessing,
# save per-filter feature maps, and summarize the run in JSON.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run directional convolution filters on a real image or a directory "
            "of images and save visualizable feature maps."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Path to one image or a directory of images. Supported extensions: "
            ".png, .jpg, .jpeg, .bmp."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory where per-image feature-map folders will be written. "
            f"Defaults to {DEFAULT_OUTPUT_DIR}."
        ),
    )
    return parser


def collect_image_paths(source: Path) -> List[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if source.is_file():
        # Single-image mode still goes through the same list-based pipeline so
        # the rest of the script does not need separate branches.
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported image file extension for {source.name}. "
                f"Supported extensions: {supported}"
            )
        return [source]

    if not source.is_dir():
        raise ValueError(f"Source must be a file or directory: {source}")

    image_paths = sorted(
        [
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files were found in directory: {source}"
        )
    return image_paths


def resolve_output_stems(image_paths: List[Path]) -> Dict[Path, str]:
    # Make output folder names unique even if two files share the same stem but
    # differ only by extension.
    assigned: set[str] = set()
    resolved: Dict[Path, str] = {}

    for image_path in image_paths:
        base_stem = image_path.stem
        candidate = base_stem
        if candidate in assigned:
            suffix = image_path.suffix.lower().lstrip(".") or "image"
            candidate = f"{base_stem}_{suffix}"

        unique_candidate = candidate
        counter = 2
        while unique_candidate in assigned:
            unique_candidate = f"{candidate}_{counter}"
            counter += 1

        assigned.add(unique_candidate)
        resolved[image_path] = unique_candidate

    return resolved


def load_rgb_image(source_path: Path) -> Image.Image:
    try:
        # Convert everything to RGB up front so later visualization behaves consistently.
        with Image.open(source_path) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise RuntimeError(f"Failed to read image: {source_path}") from exc


def preprocess_image(original_rgb: Image.Image) -> Image.Image:
    # Collapse the image into a clean black/white structure map so the filter
    # outputs emphasize edges and directions instead of color variation.
    grayscale = original_rgb.convert("L")
    contrast_stretched = ImageOps.autocontrast(grayscale)
    # Thresholding exaggerates edge structure so the directional responses are easier to read.
    return contrast_stretched.point(
        lambda value: 255 if value >= THRESHOLD else 0
    ).convert("L")


def image_to_matrix(image: Image.Image) -> Matrix:
    # Convert the PIL grayscale image into the plain 2D matrix format expected by
    # the convolution helper library.
    width, height = image.size
    pixels = image.tobytes()
    return [
        [float(pixels[row * width + col]) for col in range(width)]
        for row in range(height)
    ]


def uint8_matrix_to_image(matrix: UInt8Matrix) -> Image.Image:
    if not matrix or not matrix[0]:
        raise ValueError("Matrix must be a non-empty 2D matrix.")

    width = len(matrix[0])
    for row in matrix:
        if len(row) != width:
            raise ValueError("Matrix rows must have equal length.")

    image = Image.new("L", (width, len(matrix)))
    image.putdata([value for row in matrix for value in row])
    return image


def create_feature_map_grid(tile_images: Dict[str, Image.Image], tile_size: tuple[int, int]) -> Image.Image:
    # Arrange the original, preprocessed, and six filter outputs into one image
    # so the qualitative comparison is visible at a glance.
    tile_width, tile_height = tile_size
    grid = Image.new("RGB", (tile_width * 4, tile_height * 2), color=(255, 255, 255))

    for index, key in enumerate(GRID_ORDER):
        row = index // 4
        col = index % 4
        grid.paste(tile_images[key].convert("RGB"), (col * tile_width, row * tile_height))

    return grid


# Each processed image produces the full artifact bundle used in the README:
# preprocessed input, six feature maps, a comparison grid, and a JSON summary.
def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_cli_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def process_image(source_path: Path, output_dir: Path, output_stem: str) -> Dict[str, object]:
    original_rgb = load_rgb_image(source_path)
    preprocessed_bw = preprocess_image(original_rgb)
    image_matrix = image_to_matrix(preprocessed_bw)

    # Run the same directional filters used in the synthetic demo, then convert
    # the signed responses into viewable grayscale images.
    responses = apply_all_directional_filters(image_matrix, normalize_kernels=False)
    response_stats = {name: matrix_stats(response) for name, response in responses.items()}

    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_path = output_dir / f"{output_stem}_preprocessed_bw.png"
    preprocessed_bw.save(preprocessed_path)

    feature_map_paths: Dict[str, str] = {}
    tile_images: Dict[str, Image.Image] = {
        "original": original_rgb,
        "preprocessed_bw": preprocessed_bw,
    }

    for name in FILTER_ORDER:
        # Save each directional response as its own grayscale image so the effect
        # of each kernel can be inspected independently.
        display_matrix = normalize_matrix_abs_to_uint8(responses[name])
        feature_map_image = uint8_matrix_to_image(display_matrix)
        feature_map_path = output_dir / f"{output_stem}_{name}.png"
        feature_map_image.save(feature_map_path)
        feature_map_paths[name] = str(feature_map_path.resolve())
        tile_images[name] = feature_map_image

    feature_map_grid_path = output_dir / f"{output_stem}_feature_maps_grid.png"
    # The grid is the easiest qualitative artifact to show because it puts the
    # original image and every derived feature map side by side.
    grid = create_feature_map_grid(tile_images, preprocessed_bw.size)
    grid.save(feature_map_grid_path)

    # The JSON summary records the preprocessing choices, saved artifact paths,
    # and per-filter statistics in one machine-readable report.
    summary_path = output_dir / f"{output_stem}_summary.json"
    summary = {
        "source_image": str(source_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "output_stem": output_stem,
        "image_size": {
            "width": original_rgb.width,
            "height": original_rgb.height,
        },
        "preprocessing": {
            "pipeline": ["rgb", "grayscale", "autocontrast", "threshold"],
            "threshold": THRESHOLD,
            "binary_values": [0, 255],
        },
        "artifacts": {
            "preprocessed_bw": str(preprocessed_path.resolve()),
            "feature_maps": feature_map_paths,
            "feature_maps_grid": str(feature_map_grid_path.resolve()),
        },
        "feature_maps_grid_order": GRID_ORDER,
        "response_stats": response_stats,
    }
    write_json(summary_path, summary)

    return {
        "source_image": str(source_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "summary_json": str(summary_path.resolve()),
    }


def main() -> int:
    args = build_parser().parse_args()

    try:
        source = Path(args.source).expanduser().resolve()
        output_root = Path(args.output_dir).expanduser().resolve()

        # Resolve the batch first so naming collisions and output folders are planned once.
        image_paths = collect_image_paths(source)
        output_stems = resolve_output_stems(image_paths)

        print(
            f"Processing {len(image_paths)} image(s) from "
            f"{format_cli_path(source)}"
        )

        results = []
        for image_path in image_paths:
            output_stem = output_stems[image_path]
            image_output_dir = output_root / output_stem
            # Give each input image its own output folder so all derived artifacts stay grouped.
            print(
                f"- {image_path.name} -> "
                f"{format_cli_path(image_output_dir)}"
            )
            results.append(process_image(image_path, image_output_dir, output_stem))

        if len(results) == 1:
            # Single-image mode points straight to the one JSON artifact you are
            # most likely to open while explaining the result.
            summary_path = Path(results[0]["summary_json"])
            print(
                f"Done. Summary: {format_cli_path(summary_path)}"
            )
        else:
            print(
                f"Done. Wrote {len(results)} output folder(s) under "
                f"{format_cli_path(output_root)}"
            )

        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
