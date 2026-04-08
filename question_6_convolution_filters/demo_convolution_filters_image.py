#!/usr/bin/env python3
"""Apply the directional filters to one image or a directory of images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

from convolution_filter_library import (
    apply_all_directional_filters,
    matrix_stats,
    normalize_matrix_abs_to_uint8,
)

# Repository-local path defaults and rendering order used by the image demo.
QUESTION_DIR = Path(__file__).resolve().parent
REPO_ROOT = QUESTION_DIR.parent
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

Matrix = list[list[float]]
UInt8Matrix = list[list[int]]


# CLI and source-resolution helpers.
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
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


def collect_image_paths(source) -> list[Path]:
    """Resolve the set of input images to process."""
    # Stop immediately if the requested source path does not exist.
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    if source.is_file():
        # Reuse the directory pipeline by returning a one-item list.
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported image file extension for {source.name}. "
                f"Supported extensions: {supported}"
            )
        return [source]

    # Stop if the source is neither a file nor a directory.
    if not source.is_dir():
        raise ValueError(f"Source must be a file or directory: {source}")

    # Collect only supported image files from the directory root.
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


def resolve_output_stems(image_paths) -> dict[Path, str]:
    """Build one unique output stem for each image path."""
    # Keep output directory names unique when two files share the same stem.
    assigned = set()
    resolved = {}

    for image_path in image_paths:
        # Start from the source file stem.
        base_stem = image_path.stem
        candidate = base_stem
        if candidate in assigned:
            # Add the suffix text when another file already uses this stem.
            suffix = image_path.suffix.lower().lstrip(".") or "image"
            candidate = f"{base_stem}_{suffix}"

        unique_candidate = candidate
        counter = 2
        while unique_candidate in assigned:
            # Keep incrementing until the stem becomes unique.
            unique_candidate = f"{candidate}_{counter}"
            counter += 1

        # Reserve the chosen stem and store it for this image path.
        assigned.add(unique_candidate)
        resolved[image_path] = unique_candidate

    return resolved


def load_rgb_image(source_path) -> Image.Image:
    """Load one image as RGB."""
    try:
        # Convert every input to RGB so the later steps see one image format.
        with Image.open(source_path) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise RuntimeError(f"Failed to read image: {source_path}") from exc


def preprocess_image(original_rgb) -> Image.Image:
    """Convert one RGB image into a thresholded grayscale map."""
    # Convert to grayscale before stretching contrast or thresholding.
    grayscale = original_rgb.convert("L")
    contrast_stretched = ImageOps.autocontrast(grayscale)
    # Convert the grayscale image to a binary edge map.
    return contrast_stretched.point(
        lambda value: 255 if value >= THRESHOLD else 0
    ).convert("L")


def image_to_matrix(image) -> Matrix:
    """Convert one grayscale image into the matrix representation."""
    # Convert the grayscale image into the 2D matrix shape used by the filters.
    width, height = image.size
    # Read the image pixels as one flat byte buffer.
    pixels = image.tobytes()
    # Rebuild the flat pixel buffer into a 2D float matrix.
    return [
        [float(pixels[row * width + col]) for col in range(width)]
        for row in range(height)
    ]


def uint8_matrix_to_image(matrix) -> Image.Image:
    """Convert one uint8 matrix back into a grayscale image."""
    # Reject empty matrices before allocating the output image.
    if not matrix or not matrix[0]:
        raise ValueError("Matrix must be a non-empty 2D matrix.")

    # Verify that every row has the same width.
    width = len(matrix[0])
    for row in matrix:
        if len(row) != width:
            raise ValueError("Matrix rows must have equal length.")

    # Flatten the matrix back into one grayscale image buffer.
    image = Image.new("L", (width, len(matrix)))
    # Copy every matrix value into the image buffer in row-major order.
    image.putdata([value for row in matrix for value in row])
    return image


def create_feature_map_grid(tile_images, tile_size) -> Image.Image:
    """Arrange the original image and response maps into one grid."""
    # Place the original image and derived maps into one contact sheet.
    tile_width, tile_height = tile_size
    grid = Image.new("RGB", (tile_width * 4, tile_height * 2), color=(255, 255, 255))

    for index, key in enumerate(GRID_ORDER):
        # Convert the linear position into one row and one column.
        row = index // 4
        col = index % 4
        # Paste the current tile image into its grid cell.
        grid.paste(tile_images[key].convert("RGB"), (col * tile_width, row * tile_height))

    return grid


def write_json(path, payload) -> None:
    """Write one JSON file to disk."""
    # Create the destination folder before writing the JSON file.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write the JSON payload with indentation for readability.
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_cli_path(path) -> str:
    """Format a path for CLI output."""
    try:
        # Prefer a path relative to the current shell directory.
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        # Fall back to the absolute path when no relative form exists.
        return str(path)


def portable_repo_path(path) -> str:
    """Return one repo-relative path string."""
    # Store repo-relative paths so saved summaries remain portable.
    return os.path.relpath(path.resolve(), start=REPO_ROOT)


# Per-image processing and CLI entrypoint helpers.
def process_image(source_path, output_dir, output_stem) -> dict[str, object]:
    """Process one image and write all derived artifacts."""
    # Load the current input image and build the thresholded filter input.
    original_rgb = load_rgb_image(source_path)
    preprocessed_bw = preprocess_image(original_rgb)
    image_matrix = image_to_matrix(preprocessed_bw)

    # Apply every directional filter to the preprocessed image matrix.
    responses = apply_all_directional_filters(image_matrix, normalize_kernels=False)
    response_stats = {name: matrix_stats(response) for name, response in responses.items()}

    # Create the per-image output folder before writing any artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_path = output_dir / f"{output_stem}_preprocessed_bw.png"
    # Save the thresholded grayscale image before the derived feature maps.
    preprocessed_bw.save(preprocessed_path)

    # Track the saved file paths and in-memory tiles for the grid image.
    feature_map_paths = {}
    tile_images = {
        "original": original_rgb,
        "preprocessed_bw": preprocessed_bw,
    }

    for name in FILTER_ORDER:
        # Convert the signed response map into a viewable 8-bit image.
        display_matrix = normalize_matrix_abs_to_uint8(responses[name])
        feature_map_image = uint8_matrix_to_image(display_matrix)
        feature_map_path = output_dir / f"{output_stem}_{name}.png"
        # Save the current filter response image.
        feature_map_image.save(feature_map_path)
        feature_map_paths[name] = portable_repo_path(feature_map_path)
        tile_images[name] = feature_map_image

    feature_map_grid_path = output_dir / f"{output_stem}_feature_maps_grid.png"
    # Combine the original image and all derived maps into one grid image.
    grid = create_feature_map_grid(tile_images, preprocessed_bw.size)
    # Save the combined contact-sheet image.
    grid.save(feature_map_grid_path)

    # Record the preprocessing steps, artifact paths, and response stats.
    summary_path = output_dir / f"{output_stem}_summary.json"
    summary = {
        "source_image": portable_repo_path(source_path),
        "output_directory": portable_repo_path(output_dir),
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
            "preprocessed_bw": portable_repo_path(preprocessed_path),
            "feature_maps": feature_map_paths,
            "feature_maps_grid": portable_repo_path(feature_map_grid_path),
        },
        "feature_maps_grid_order": GRID_ORDER,
        "response_stats": response_stats,
    }
    write_json(summary_path, summary)

    # Return the key output paths used by the CLI summary.
    return {
        "source_image": str(source_path.resolve()),
        "output_directory": str(output_dir.resolve()),
        "summary_json": str(summary_path.resolve()),
    }


def main() -> int:
    """Run the image-processing CLI flow."""
    args = build_parser().parse_args()

    try:
        # Resolve the source path and output root before processing any images.
        source = Path(args.source).expanduser().resolve()
        output_root = Path(args.output_dir).expanduser().resolve()

        # Collect the input images and reserve one output stem for each one.
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
            # Group every artifact for this image under one output directory.
            print(
                f"- {image_path.name} -> "
                f"{format_cli_path(image_output_dir)}"
            )
            # Process the current image and record its summary paths.
            results.append(process_image(image_path, image_output_dir, output_stem))

        if len(results) == 1:
            # Report the summary path directly in single-image mode.
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
        # Convert common validation failures into one CLI error line.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
