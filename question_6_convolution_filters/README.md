# Question 6: Directional Convolution Filter Library

This folder demonstrates a small library of directional convolution filters:

- `horizontal`
- `vertical`
- `upper_left`
- `lower_right`
- `upper_right`
- `lower_left`

## Canonical Files

- `convolution_filter_library.py`
- `demo_convolution_filters.py`
- `demo_convolution_filters_image.py`

## What Each Script Does

- `convolution_filter_library.py`: derives the six directional 3x3 kernels, applies convolution, and computes the response summaries used by both demos
- `demo_convolution_filters.py`: builds one synthetic image with horizontal, vertical, and diagonal structure so each directional filter has something obvious to detect
- `demo_convolution_filters_image.py`: applies the same filters to a real image or directory of images and saves per-filter visualizations plus a JSON report

## How To Use This Question

The key idea is that each kernel looks for intensity change along one
direction. Strong responses indicate that the input contains an edge or line
aligned with that filter.

## Synthetic Demo

Run from this folder:

```bash
python3 demo_convolution_filters.py --size 11
```

The synthetic demo prints:

- the generated input matrix
- each derived directional kernel
- response statistics such as `sum_abs`, `mean_abs`, `min`, and `max`

It also writes:

- `outputs/convolution_filter_demo.json`

## Real Image Demo

Run on one image:

```bash
python3 demo_convolution_filters_image.py \
  --source inputs/example.jpg \
  --output-dir outputs
```

Run on a directory:

```bash
python3 demo_convolution_filters_image.py \
  --source inputs \
  --output-dir outputs
```

## Output Layout

For each image, the real-image demo writes a dedicated output folder:

- `outputs/<stem>/<stem>_preprocessed_bw.png`
- `outputs/<stem>/<stem>_horizontal.png`
- `outputs/<stem>/<stem>_vertical.png`
- `outputs/<stem>/<stem>_upper_left.png`
- `outputs/<stem>/<stem>_lower_right.png`
- `outputs/<stem>/<stem>_upper_right.png`
- `outputs/<stem>/<stem>_lower_left.png`
- `outputs/<stem>/<stem>_feature_maps_grid.png`
- `outputs/<stem>/<stem>_summary.json`

## Preprocessing Pipeline

The real-image demo applies:

1. RGB
2. grayscale
3. autocontrast
4. threshold at 128 to create a black/white image

This makes the directional responses easier to read because the filters are
responding to strong edge structure instead of raw color variation.

## Feature-Map Grid

The contact sheet is a fixed 2x4 grid in this order:

- `original`
- `preprocessed_bw`
- `horizontal`
- `vertical`
- `upper_left`
- `lower_right`
- `upper_right`
- `lower_left`

That grid is the easiest artifact to show in a walkthrough because it puts the
original image, the thresholded input, and all six feature maps in one place.
