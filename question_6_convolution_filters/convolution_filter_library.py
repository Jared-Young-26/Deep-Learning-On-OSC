#!/usr/bin/env python3
"""Directional convolution filters and a basic 2D convolution implementation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

Matrix = List[List[float]]
Direction = Tuple[int, int]


DIRECTION_VECTORS: Dict[str, Direction] = {
    "horizontal": (1, 0),
    "vertical": (0, 1),
    "upper_left": (-1, -1),
    "lower_right": (1, 1),
    "upper_right": (1, -1),
    "lower_left": (-1, 1),
}


# Kernel construction starts from an intuitive direction vector, then turns that
# into a 3x3 derivative-style filter with signed responses.
def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _normalize_direction(direction: Direction) -> Direction:
    # Reduce every direction component to {-1, 0, 1} so the resulting kernels stay
    # in the simple derivative-style family used throughout the assignment.
    dx, dy = direction
    dx = _sign(dx)
    dy = _sign(dy)
    if dx == 0 and dy == 0:
        raise ValueError("Direction vector cannot be (0, 0).")
    return dx, dy


def derive_directional_kernel(direction: Direction, normalize: bool = False) -> Matrix:
    """Derive a 3x3 directional derivative kernel from a direction vector."""
    dx, dy = _normalize_direction(direction)
    kernel: Matrix = []
    for row in (-1, 0, 1):
        kernel_row: List[float] = []
        for col in (-1, 0, 1):
            # Project each kernel position onto the requested direction:
            # negative side -> -1, center/orthogonal -> 0, positive side -> +1.
            projection = dx * col + dy * row
            kernel_row.append(float(_sign(projection)))
        kernel.append(kernel_row)

    if not normalize:
        return kernel

    # Optional normalization keeps the total positive weight at 1.0, which can be
    # useful when comparing kernels on a more equal overall scale.
    positive_weight = sum(value for row in kernel for value in row if value > 0)
    if positive_weight == 0:
        return kernel
    scale = float(positive_weight)
    return [[value / scale for value in row] for row in kernel]


def get_directional_kernels(normalize: bool = False) -> Dict[str, Matrix]:
    """Return horizontal, vertical, and 4 diagonal kernels."""
    return {
        name: derive_directional_kernel(vector, normalize=normalize)
        for name, vector in DIRECTION_VECTORS.items()
    }


# This is the core convolution routine: zero-pad conceptually at the borders and
# keep the output matrix the same size as the input image.
def convolve2d(image: Matrix, kernel: Matrix) -> Matrix:
    """Convolve an image with a kernel using zero-padding and same-size output."""
    if not image or not image[0]:
        raise ValueError("Image must be a non-empty 2D matrix.")
    if not kernel or not kernel[0]:
        raise ValueError("Kernel must be a non-empty 2D matrix.")

    image_h = len(image)
    image_w = len(image[0])
    for row in image:
        if len(row) != image_w:
            raise ValueError("Image rows must have equal length.")

    kernel_h = len(kernel)
    kernel_w = len(kernel[0])
    for row in kernel:
        if len(row) != kernel_w:
            raise ValueError("Kernel rows must have equal length.")

    if kernel_h % 2 == 0 or kernel_w % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    row_radius = kernel_h // 2
    col_radius = kernel_w // 2

    # Allocate a same-size output image so every input pixel gets one response value.
    output: Matrix = [[0.0 for _ in range(image_w)] for _ in range(image_h)]

    for out_r in range(image_h):
        for out_c in range(image_w):
            value = 0.0
            for k_r in range(kernel_h):
                for k_c in range(kernel_w):
                    # Shift the kernel window over the current output location.
                    in_r = out_r + k_r - row_radius
                    in_c = out_c + k_c - col_radius
                    # Values outside the image bounds are skipped, which is equivalent
                    # to convolving against zeros around the border.
                    if 0 <= in_r < image_h and 0 <= in_c < image_w:
                        value += image[in_r][in_c] * kernel[k_r][k_c]
            output[out_r][out_c] = value

    return output


def apply_all_directional_filters(
    image: Matrix, normalize_kernels: bool = False
) -> Dict[str, Matrix]:
    """Apply all 6 directional filters to an image."""
    kernels = get_directional_kernels(normalize=normalize_kernels)
    return {name: convolve2d(image, kernel) for name, kernel in kernels.items()}


# Post-processing helpers make the raw signed responses easier to compare and display.
def normalize_matrix_abs_to_uint8(matrix: Matrix) -> List[List[int]]:
    """Scale absolute matrix responses independently into the 0-255 range."""
    if not matrix or not matrix[0]:
        raise ValueError("Matrix must be a non-empty 2D matrix.")

    matrix_w = len(matrix[0])
    for row in matrix:
        if len(row) != matrix_w:
            raise ValueError("Matrix rows must have equal length.")

    max_abs = max(abs(value) for row in matrix for value in row)
    if max_abs == 0:
        return [[0 for _ in row] for row in matrix]

    # Use absolute value here because we want to visualize response strength,
    # regardless of whether the edge transition produced a positive or negative sign.
    scale = 255.0 / max_abs
    return [
        [int(round(abs(value) * scale)) for value in row]
        for row in matrix
    ]


def matrix_stats(matrix: Matrix) -> Dict[str, float]:
    """Compute simple response stats for demonstration/reporting."""
    # These metrics are enough to compare filters without storing every response matrix.
    values: Iterable[float] = [value for row in matrix for value in row]
    values = list(values)
    abs_values = [abs(value) for value in values]
    return {
        "min": min(values),
        "max": max(values),
        "mean_abs": sum(abs_values) / len(abs_values),
        "sum_abs": sum(abs_values),
    }
