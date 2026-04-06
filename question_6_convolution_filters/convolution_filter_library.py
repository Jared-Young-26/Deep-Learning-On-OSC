#!/usr/bin/env python3
"""Directional kernels and basic 2D convolution helpers."""

from __future__ import annotations

Matrix = list[list[float]]
Direction = tuple[int, int]


# Map each filter label to the direction vector it measures.
DIRECTION_VECTORS: dict[str, Direction] = {
    "horizontal": (1, 0),
    "vertical": (0, 1),
    "upper_left": (-1, -1),
    "lower_right": (1, 1),
    "upper_right": (1, -1),
    "lower_left": (-1, 1),
}


def _sign(value) -> int:
    """Return the sign of one numeric value."""
    # Reduce every input to the {-1, 0, 1} range used by the kernels.
    # Positive values map to 1.
    if value > 0:
        return 1
    # Negative values map to -1.
    if value < 0:
        return -1
    # Zero stays zero.
    return 0


def _normalize_direction(direction) -> Direction:
    """Clamp a direction vector to single-step components."""
    # Normalize each component before the kernel builder uses it.
    # Unpack the input direction tuple.
    dx, dy = direction
    # Reduce each component to -1, 0, or 1.
    dx = _sign(dx)
    dy = _sign(dy)
    # Reject the zero vector because it has no direction.
    if dx == 0 and dy == 0:
        raise ValueError("Direction vector cannot be (0, 0).")
    return dx, dy


def derive_directional_kernel(direction, normalize=False) -> Matrix:
    """Derive a 3x3 directional derivative kernel from a direction vector."""
    # Normalize the requested direction before building the kernel.
    dx, dy = _normalize_direction(direction)
    kernel = []
    for row in (-1, 0, 1):
        # Walk the 3x3 window row by row.
        kernel_row = []
        for col in (-1, 0, 1):
            # Project the current cell onto the requested direction.
            projection = dx * col + dy * row
            # Convert the projection back to -1, 0, or 1.
            kernel_row.append(float(_sign(projection)))
        # Finish the current kernel row.
        kernel.append(kernel_row)

    # Return the raw signed kernel unless normalized weights were requested.
    if not normalize:
        return kernel

    # Scale the positive side so different kernels share the same total weight.
    positive_weight = sum(value for row in kernel for value in row if value > 0)
    # Leave the kernel unchanged if it has no positive weights.
    if positive_weight == 0:
        return kernel
    scale = float(positive_weight)
    # Divide every entry by the positive weight total.
    return [[value / scale for value in row] for row in kernel]


def get_directional_kernels(normalize=False) -> dict[str, Matrix]:
    """Return horizontal, vertical, and 4 diagonal kernels."""
    # Build one named kernel for each predefined direction.
    return {
        name: derive_directional_kernel(vector, normalize=normalize)
        for name, vector in DIRECTION_VECTORS.items()
    }


def convolve2d(image, kernel) -> Matrix:
    """Convolve an image with a kernel using zero-padding and same-size output."""
    # Reject empty inputs before any shape math runs.
    if not image or not image[0]:
        raise ValueError("Image must be a non-empty 2D matrix.")
    if not kernel or not kernel[0]:
        raise ValueError("Kernel must be a non-empty 2D matrix.")

    # Capture the image shape once and verify the rows are consistent.
    image_h = len(image)
    image_w = len(image[0])
    for row in image:
        # Every image row must match the first row width.
        if len(row) != image_w:
            raise ValueError("Image rows must have equal length.")

    # Capture the kernel shape once and verify the rows are consistent.
    kernel_h = len(kernel)
    kernel_w = len(kernel[0])
    for row in kernel:
        # Every kernel row must match the first row width.
        if len(row) != kernel_w:
            raise ValueError("Kernel rows must have equal length.")

    # Same-size convolution needs odd kernel dimensions so there is a true center.
    if kernel_h % 2 == 0 or kernel_w % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    # Use the half-width of the kernel to center it over each output pixel.
    row_radius = kernel_h // 2
    col_radius = kernel_w // 2

    # Allocate one output value per input pixel.
    output = [[0.0 for _ in range(image_w)] for _ in range(image_h)]

    for out_r in range(image_h):
        for out_c in range(image_w):
            # Reset the accumulator for the current output location.
            value = 0.0
            for k_r in range(kernel_h):
                for k_c in range(kernel_w):
                    # Translate the kernel cell back to the matching input cell.
                    in_r = out_r + k_r - row_radius
                    in_c = out_c + k_c - col_radius
                    # Skip cells that fall outside the image bounds.
                    if 0 <= in_r < image_h and 0 <= in_c < image_w:
                        # Accumulate this input pixel times this kernel weight.
                        value += image[in_r][in_c] * kernel[k_r][k_c]
            # Store the finished response for this pixel.
            output[out_r][out_c] = value

    return output


def apply_all_directional_filters(
    image, normalize_kernels=False
) -> dict[str, Matrix]:
    """Apply all 6 directional filters to an image."""
    # Build the kernel set once, then run one convolution per direction.
    kernels = get_directional_kernels(normalize=normalize_kernels)
    # Convolve the image with every named kernel.
    return {name: convolve2d(image, kernel) for name, kernel in kernels.items()}


def normalize_matrix_abs_to_uint8(matrix) -> list[list[int]]:
    """Scale absolute matrix responses independently into the 0-255 range."""
    if not matrix or not matrix[0]:
        raise ValueError("Matrix must be a non-empty 2D matrix.")

    matrix_w = len(matrix[0])
    for row in matrix:
        # Every row must match the first row width.
        if len(row) != matrix_w:
            raise ValueError("Matrix rows must have equal length.")

    # Keep an all-zero matrix all-zero instead of dividing by zero.
    max_abs = max(abs(value) for row in matrix for value in row)
    if max_abs == 0:
        return [[0 for _ in row] for row in matrix]

    # Scale by the strongest absolute response so the image uses the full range.
    scale = 255.0 / max_abs
    # Convert every absolute response into an 8-bit intensity.
    return [
        [int(round(abs(value) * scale)) for value in row]
        for row in matrix
    ]


def matrix_stats(matrix) -> dict[str, float]:
    """Compute summary statistics for one response matrix."""
    # Flatten the matrix once so the summary metrics reuse the same values.
    values = [value for row in matrix for value in row]
    # Build the absolute-value view used by the magnitude stats.
    abs_values = [abs(value) for value in values]
    return {
        "min": min(values),
        "max": max(values),
        "mean_abs": sum(abs_values) / len(abs_values),
        "sum_abs": sum(abs_values),
    }
