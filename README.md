# Deep Learning on OSC

This repository collects several computer vision workflows for object detection,
semantic segmentation, convolution filtering, and transfer learning. Each
module includes its own README, runnable scripts, sample inputs, and output
folders. Third-party source dependencies are cloned under `external/` so the
modules can share repo-local environments.

Commands below assume you are running from the repository root.

## Modules

### [Faster R-CNN Object Detection](question_3_faster_rcnn/README.md)

Runs image object detection with a local clone of `trzy/FasterRCNN`.

- Setup: `bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh` installs both runtimes by default
- Models: `bash question_3_faster_rcnn/download_models_fasterrcnn.sh`
- Run: `python3 question_3_faster_rcnn/demo_fasterrcnn.py --mode to-file` auto-selects CUDA PyTorch or the TF2 fallback

### [YOLO11 and YOLOv12 Object Detection](question_4_yolo11_yolov12/README.md)

Provides separate wrappers for YOLO11 and YOLOv12 with dedicated inputs and
outputs for each model.

- YOLO11 setup: `bash question_4_yolo11_yolov12/setup_yolo11_osc.sh`
- YOLO11 run: `python3 question_4_yolo11_yolov12/demo_yolo11.py`
- YOLOv12 setup: `bash question_4_yolo11_yolov12/setup_yolov12_osc.sh` prefers Python 3.11 and falls back to `python3`
- YOLOv12 run: `python3 question_4_yolo11_yolov12/demo_yolov12.py`

### [Semantic Segmentation with YOLO11 and iSAID](question_5_semantic_segmentation/README.md)

Bootstraps iSAID data, fine-tunes YOLO11 segmentation, and exports semantic
masks for aerial imagery.

- Setup: `bash question_5_semantic_segmentation/setup_yolo11_osc.sh`
- Train: `external/ultralytics/.venv/bin/python question_5_semantic_segmentation/train_isaid_seg.py --repo-dir external/ultralytics --device 0`
- Demo: `external/ultralytics/.venv/bin/python question_5_semantic_segmentation/demo_yolo_segmentation.py --repo-dir external/ultralytics --device cpu`

### [Directional Convolution Filters](question_6_convolution_filters/README.md)

Derives six directional convolution kernels and applies them to synthetic
patterns or real images.

- Synthetic demo: `python3 question_6_convolution_filters/demo_convolution_filters.py --size 11`
- Image demo: `python3 question_6_convolution_filters/demo_convolution_filters_image.py --source question_6_convolution_filters/inputs --output-dir question_6_convolution_filters/outputs`

### [Transfer Learning Object Detection with TLlib](question_7_transfer_learning/README.md)

Wraps TLlib's VOC-to-Clipart domain adaptation example with environment checks,
dataset preparation, training, and reporting.

- Setup: `bash question_7_transfer_learning/setup_tllib_osc.sh`
- Doctor: `python3 question_7_transfer_learning/demo_tllib_object_detection.py --mode doctor`
- Full run: `bash question_7_transfer_learning/run_tllib_osc.sh`
