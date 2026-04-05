# Question 4: YOLO11 and YOLOv12 on OSC

This folder packages two separate object-detection demos:

- YOLO11 from [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- YOLOv12 from [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)

Each model gets its own setup script, demo wrapper, input folder, and output
folder so the two workflows stay separate and easy to compare.

## Canonical Files

- `setup_yolo11_osc.sh`
- `setup_yolov12_osc.sh`
- `demo_yolo11.py`
- `demo_yolov12.py`
- `inputs/yolo11/`
- `inputs/yolov12/`
- `outputs/yolo11/`
- `outputs/yolov12/`

## What Each Script Does

- `setup_yolo11_osc.sh`: clones the Ultralytics repo and builds the YOLO11 environment in `external/ultralytics/.venv`
- `demo_yolo11.py`: resolves the source image(s), runs one upstream `YOLO.predict(...)` call, and copies the annotated images from `runs/detect/` back into `outputs/yolo11/`
- `setup_yolov12_osc.sh`: clones the YOLOv12 repo and installs the smaller dependency subset needed for this assignment demo
- `demo_yolov12.py`: follows the same wrapper pattern, but targets the YOLOv12 checkout and writes to `outputs/yolov12/`

## Canonical Workflow

For each model:

1. run the matching setup script once
2. place images into the matching `inputs/` folder, or use a local file / URL
3. run the matching demo wrapper
4. show the saved annotated image(s) from the matching `outputs/` folder

## YOLO11 Setup and Demo

Run from this folder:

```bash
bash setup_yolo11_osc.sh
python3 demo_yolo11.py
```

The default upstream clone location is `../external/ultralytics`.

Examples:

```bash
# Process every supported image in inputs/yolo11/
python3 demo_yolo11.py

# Process one local file
python3 demo_yolo11.py --source inputs/yolo11/my_test.jpg

# Process a URL
python3 demo_yolo11.py --source https://ultralytics.com/images/bus.jpg
```

## YOLOv12 Setup and Demo

Run from this folder:

```bash
bash setup_yolov12_osc.sh
python3 demo_yolov12.py
```

Optional on supported Linux `x86_64` CUDA systems only:

```bash
INSTALL_FLASH_ATTN=1 bash setup_yolov12_osc.sh
```

The default upstream clone location is `../external/yolov12`.

Examples:

```bash
# Process every supported image in inputs/yolov12/
python3 demo_yolov12.py

# Process one local file
python3 demo_yolov12.py --source inputs/yolov12/my_test.jpg --device cpu

# Process a URL
python3 demo_yolov12.py --source https://ultralytics.com/images/bus.jpg --device cpu
```

## Input and Output Behavior

- YOLO11 reads from `inputs/yolo11/` by default and writes to `outputs/yolo11/`
- YOLOv12 reads from `inputs/yolov12/` by default and writes to `outputs/yolov12/`
- If `--source` is a directory, the wrapper writes one annotated file per image using matching filenames
- If `--source` is one file, the wrapper writes one annotated file with the same filename unless `--output` is used
- If `--source` is a URL, the wrapper derives the saved filename from the URL path unless `--output` is passed
- `--output` is for single-image runs only
- `--output-dir` changes the destination directory for mirrored filenames

## Platform Notes

- On headless nodes, both demos save files directly and do not need a GUI
- If needed, load OSC Python or CUDA modules before setup
- You can force CPU inference with `--device cpu`
- Supported image suffixes are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`
- `INSTALL_FLASH_ATTN=1` only makes sense on supported Linux `x86_64` CUDA nodes
- On macOS, YOLOv12 setup intentionally skips FlashAttention instead of failing


