# OSC Python Compatibility

This repository standardizes OSC on Python 3.9.18.

## Decision

- `q3` Faster R-CNN: Python 3.9.18 only.
- `q4` YOLO11 and YOLOv12: Python 3.9.18 baseline on OSC.
- `q5` YOLO11 segmentation: Python 3.9.18 baseline on OSC.
- `q6` Convolution filters: Python 3.9.18 baseline on OSC.
- `q7` TLlib/Detectron2: Python 3.9.18 only.

## Why

- Repo-owned entrypoints were audited for Python 3.9 syntax compatibility.
- The fragile stacks are `q3` and `q7`, where upstream package support matters
  more than local script syntax.
- `q4` and `q5` could likely tolerate newer minors in isolation, but the repo
  uses one OSC baseline to avoid per-module environment drift.
- `q6` is lightweight, but keeping the same Python minor across the repo avoids
  special cases in OSC instructions.

## Evidence

- [question_3_faster_rcnn/setup_fasterrcnn_osc.sh](/Users/jaredyoung/Documents/Programs/GitHub/Deep-Learning-On-OSC/question_3_faster_rcnn/setup_fasterrcnn_osc.sh)
  installs the FasterRCNN PyTorch and TF2 fallback stack.
- [question_7_transfer_learning/setup_tllib_osc.sh](/Users/jaredyoung/Documents/Programs/GitHub/Deep-Learning-On-OSC/question_7_transfer_learning/setup_tllib_osc.sh)
  installs the TLlib and Detectron2-backed object-detection stack.
- [PyTorch 2.0.0 on PyPI](https://pypi.org/project/torch/2.0.0/)
  is the relevant upstream signal for the FasterRCNN PyTorch path.
- [Detectron2 install docs](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
  are the relevant upstream signal for the TLlib object-detection path.
- [external/ultralytics/pyproject.toml](/Users/jaredyoung/Documents/Programs/GitHub/Deep-Learning-On-OSC/external/ultralytics/pyproject.toml)
  and [PyTorch 2.2.2 on PyPI](https://pypi.org/project/torch/2.2.2/)
  support keeping the YOLO workflows on the shared 3.9.18 baseline.
