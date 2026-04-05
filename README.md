# Deep Learning on OSC Assignment Index

This repository is organized by assignment question. Each question folder
contains its own README, setup script, and demo wrapper.

Third-party upstream repositories are cloned into the shared `external/`
directory so they can be reused across questions.

## Question Folders

### Question 3: Faster R-CNN

- Folder: [`question_3_faster_rcnn`](question_3_faster_rcnn/README.md)
- Setup: `cd question_3_faster_rcnn && bash setup_fasterrcnn_osc.sh`
- Run: `python3 demo_fasterrcnn.py --mode to-file`

### Question 4: YOLO11 and YOLOv12

- Folder: [`question_4_yolo11_yolov12`](question_4_yolo11_yolov12/README.md)
- Setup: `cd question_4_yolo11_yolov12 && bash setup_yolov12_osc.sh`
- Run: `python3 demo_yolov12.py --source https://ultralytics.com/images/bus.jpg`

### Question 5: Semantic Segmentation

- Folder: [`question_5_semantic_segmentation`](question_5_semantic_segmentation/README.md)
- Setup: `cd question_5_semantic_segmentation && bash setup_yolov12_osc.sh`
- Run: `python3 demo_yolo_segmentation.py --source /path/to/image.jpg`

### Question 6: Convolution Filters

- Folder: [`question_6_convolution_filters`](question_6_convolution_filters/README.md)
- Setup: `cd question_6_convolution_filters`
- Run: `python3 demo_convolution_filters.py --size 11`

### Question 7: Transfer Learning Library

- Folder: [`question_7_transfer_learning`](question_7_transfer_learning/README.md)
- Setup: `cd question_7_transfer_learning && bash setup_tllib_osc.sh`
- Run: `bash run_tllib_osc.sh`
