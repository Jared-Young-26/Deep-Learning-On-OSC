# Directional Convolution Filters

This module derives six 3x3 directional convolution kernels and applies them to
synthetic patterns or real images. The demos produce JSON summaries and viewable
feature maps that highlight horizontal, vertical, and diagonal responses.

## Overview

- `question_6_convolution_filters/convolution_filter_library.py` derives the
  six directional kernels, applies convolution, and computes response
  statistics.
- `question_6_convolution_filters/demo_convolution_filters.py` generates a
  synthetic image with horizontal, vertical, and diagonal structure and reports
  the filter responses.
- `question_6_convolution_filters/demo_convolution_filters_image.py` applies the
  same filters to a real image or directory of images and writes visualization
  artifacts for each input.

## Setup

No module-specific setup script is included. Run the Python scripts from an
environment that has the required Python dependencies available. On OSC, the
supported baseline is Python 3.9.18.

Minimal dependencies:

- Synthetic demo: Python 3.9.18 only.
- Real-image demo: Python 3.9.18 plus `Pillow`.

## Run

Synthetic demo:

```bash
python3.9 question_6_convolution_filters/demo_convolution_filters.py --size 11
```

Real-image demo on one image:

```bash
python3.9 question_6_convolution_filters/demo_convolution_filters_image.py \
  --source question_6_convolution_filters/inputs/000000001000.jpg \
  --output-dir question_6_convolution_filters/outputs
```

Real-image demo on a directory:

```bash
python3.9 question_6_convolution_filters/demo_convolution_filters_image.py \
  --source question_6_convolution_filters/inputs \
  --output-dir question_6_convolution_filters/outputs
```

## Inputs and Outputs

- The image demo accepts `.png`, `.jpg`, `.jpeg`, and `.bmp` inputs.
- The synthetic demo writes
  `question_6_convolution_filters/outputs/convolution_filter_demo.json`.
- For each real-image input, the image demo writes a dedicated directory under
  `question_6_convolution_filters/outputs/<stem>/`.
- Each real-image output directory contains `<stem>_preprocessed_bw.png`,
  `<stem>_horizontal.png`, `<stem>_vertical.png`, `<stem>_upper_left.png`,
  `<stem>_lower_right.png`, `<stem>_upper_right.png`,
  `<stem>_lower_left.png`, `<stem>_feature_maps_grid.png`, and
  `<stem>_summary.json`.
- The real-image preprocessing pipeline is: RGB -> grayscale -> autocontrast ->
  threshold at 128.
- The feature-map grid combines the original image, the thresholded black/white
  image, and the six directional responses in one 2x4 contact sheet.
