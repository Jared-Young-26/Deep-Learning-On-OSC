#!/usr/bin/env python3
"""Run the directional filters on a synthetic test image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convolution_filter_library import (
    apply_all_directional_filters,
    get_directional_kernels,
    matrix_stats,
)

# Repository-local output defaults used by the synthetic demo.
QUESTION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = QUESTION_DIR / "outputs" / "convolution_filter_demo.json"

Matrix = list[list[float]]


def build_synthetic_image(size, amplitude=255.0) -> Matrix:
    """Build a synthetic image containing six oriented structures."""
    # Reject sizes that cannot place the center-aligned patterns cleanly.
    if size < 7 or size % 2 == 0:
        raise ValueError("Size must be odd and >= 7.")

    # Start from an all-zero image.
    image = [[0.0 for _ in range(size)] for _ in range(size)]
    center = size // 2

    # Draw the horizontal stroke through the center row.
    for c in range(1, size - 1):
        image[center][c] = amplitude

    # Draw the vertical stroke through the center column.
    for r in range(1, size - 1):
        image[r][center] = amplitude

    # Draw both diagonals so every kernel sees one matching structure.
    for i in range(1, size - 1):
        image[i][i] = amplitude
        image[i][size - 1 - i] = amplitude

    # Return the completed synthetic image matrix.
    return image


def format_matrix(matrix, width=6) -> str:
    """Format one matrix as aligned text."""
    # Format each row to a fixed width so the grid stays readable.
    lines = []
    for row in matrix:
        # Format one row of numeric values into aligned columns.
        lines.append(" ".join(f"{int(round(value)):>{width}d}" for value in row))
    return "\n".join(lines)


def summarize_responses(responses) -> dict[str, dict[str, float]]:
    """Reduce each full response map to summary metrics."""
    # Run the shared stats helper on each named response matrix.
    return {name: matrix_stats(matrix) for name, matrix in responses.items()}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate directional convolution filters on a synthetic image "
            "containing horizontal, vertical, and diagonal structures."
        )
    )
    parser.add_argument("--size", type=int, default=11, help="Synthetic image size (odd).")
    parser.add_argument(
        "--amplitude", type=float, default=255.0, help="Foreground intensity."
    )
    parser.add_argument(
        "--save-json",
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Path to save kernels and response statistics as JSON. "
            "Defaults to this question folder's outputs directory."
        ),
    )
    return parser


def main() -> int:
    """Run the synthetic-image filter comparison."""
    # Parse the CLI configuration once at startup.
    args = build_parser().parse_args()

    # Build the input image before deriving kernels or responses.
    image = build_synthetic_image(size=args.size, amplitude=args.amplitude)

    # Derive the kernels and evaluate them against the same input matrix.
    kernels = get_directional_kernels(normalize=False)
    responses = apply_all_directional_filters(image, normalize_kernels=False)
    stats = summarize_responses(responses)

    print("Input image:")
    print(format_matrix(image))
    print()

    print("Directional kernels:")
    for name, kernel in kernels.items():
        # Showing the kernel values directly makes it easier to justify why a
        # filter responds strongly to one orientation and weakly to the others.
        print(f"\n[{name}]")
        print(format_matrix(kernel, width=3))

    print("\nResponse summary (higher sum_abs => stronger response):")
    for name, values in stats.items():
        print(
            f"{name:>12}: "
            f"sum_abs={values['sum_abs']:.1f}, "
            f"mean_abs={values['mean_abs']:.2f}, "
            f"min={values['min']:.1f}, "
            f"max={values['max']:.1f}"
        )

    # Record the same kernel and response information in a structured report.
    output = {
        "input_size": args.size,
        "input_amplitude": args.amplitude,
        "kernels": kernels,
        "response_stats": stats,
    }
    # Save the same information that was printed to the terminal.
    output_path = Path(args.save_json).expanduser().resolve()
    # Create the output folder before writing the JSON file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the kernels and summary stats as formatted JSON.
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
