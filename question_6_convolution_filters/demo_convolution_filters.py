#!/usr/bin/env python3
"""Demonstrate horizontal, vertical, and diagonal convolution filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from convolution_filter_library import (
    apply_all_directional_filters,
    get_directional_kernels,
    matrix_stats,
)

QUESTION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = QUESTION_DIR / "outputs" / "convolution_filter_demo.json"

Matrix = List[List[float]]


# The synthetic image deliberately contains one strong structure in each supported
# direction so the filter responses are easy to justify by inspection.
def build_synthetic_image(size: int, amplitude: float = 255.0) -> Matrix:
    if size < 7 or size % 2 == 0:
        raise ValueError("Size must be odd and >= 7.")

    image: Matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    center = size // 2

    # Horizontal and vertical strokes.
    for c in range(1, size - 1):
        image[center][c] = amplitude
    for r in range(1, size - 1):
        image[r][center] = amplitude

    # Diagonal strokes.
    for i in range(1, size - 1):
        image[i][i] = amplitude
        image[i][size - 1 - i] = amplitude

    return image


def format_matrix(matrix: Matrix, width: int = 6) -> str:
    # Print the matrices in aligned columns so the directional structure is easy to read.
    lines = []
    for row in matrix:
        lines.append(" ".join(f"{int(round(value)):>{width}d}" for value in row))
    return "\n".join(lines)


def summarize_responses(responses: Dict[str, Matrix]) -> Dict[str, Dict[str, float]]:
    # Collapse each full response map into a compact set of comparison metrics.
    return {name: matrix_stats(matrix) for name, matrix in responses.items()}


# The CLI keeps the demo lightweight: generate a pattern, run filters, and save one report.
def build_parser() -> argparse.ArgumentParser:
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
    args = build_parser().parse_args()
    image = build_synthetic_image(size=args.size, amplitude=args.amplitude)

    # Build the directional kernels once, then compare how strongly each one fires
    # on the same toy image.
    kernels = get_directional_kernels(normalize=False)
    responses = apply_all_directional_filters(image, normalize_kernels=False)
    stats = summarize_responses(responses)

    print("Input image:")
    print(format_matrix(image))
    print()

    print("Directional kernels:")
    for name, kernel in kernels.items():
        # Showing the kernel values directly makes it easier to justify why a
        # filter should respond most strongly to one orientation and not another.
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

    output = {
        "input_size": args.size,
        "input_amplitude": args.amplitude,
        "kernels": kernels,
        "response_stats": stats,
    }
    # Save the structured report so the printed explanation and the JSON artifact agree.
    output_path = Path(args.save_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
