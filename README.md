# Deep Learning on OSC

This repository collects several computer vision workflows for object detection,
semantic segmentation, convolution filtering, and transfer learning. Each
module includes its own README, runnable scripts, sample inputs, and output
folders. Third-party source dependencies are cloned under `external/` so the
modules can share repo-local environments.

Commands below assume you are running from the repository root with Python 3.9.18
loaded on OSC.

## OSC Python Baseline

Use Python 3.9.18 for every question directory on OSC.

- `q3` Faster R-CNN: Python 3.9.18 only.
- `q4` YOLO11 and YOLOv12: use Python 3.9.18 on OSC.
- `q5` YOLO11 segmentation: use Python 3.9.18 on OSC.
- `q6` Convolution filters: use Python 3.9.18 on OSC.
- `q7` TLlib/Detectron2: Python 3.9.18 only.

See `Compatibility.md` for the audit summary and upstream support
notes behind this baseline.

## OSC GPU Launchers

GPU resources on OSC come from Slurm allocations, not from the Python scripts
themselves. Request the GPU node first, then run the repo command inside that
allocation.

Batch example:

```bash
bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- \
  bash question_7_transfer_learning/run_tllib_osc.sh
```

Interactive example:

```bash
bash osc_gpu_interactive.sh --account <OSC_ACCOUNT> --time 01:00:00
```

The batch launcher submits an `sbatch` job with `--gpus-per-node=1` by default,
loads `cuda`, and runs the requested command after the GPU allocation starts.
The interactive launcher starts an `salloc ... srun --pty /bin/bash` shell with
the same default GPU request.

## One-Command OSC Run

Use the repo-root orchestrator when you want questions 3 through 7 to run in
order from one command:

```bash
bash run_assignment_osc.sh --account <OSC_ACCOUNT> --time 16:00:00
```

By default, this script:

- self-submits one GPU batch job through `osc_gpu_batch.sh` when you launch it from an OSC login node
- runs the staged pipeline directly when you are already inside a Slurm allocation
- reuses existing environments, model downloads, datasets, and heavy output sentinels when they already exist
- defaults question 7 to the `benchmark` profile

Useful flags:

- `--q7-profile smoke` switches question 7 to the smoke profile
- `--force` reruns heavy setup, training, and Q7 pipeline stages even when sentinels already exist
- `--cluster`, `--nodes`, `--gpus-per-node`, and `--job-name` are forwarded to `osc_gpu_batch.sh`
- `--dry-run` prints either the self-submit command or the ordered inner stage plan without executing it

The script uses `--inside-allocation` internally after self-submit. You
normally do not need to pass that flag yourself.

## Modules

### [Faster R-CNN Object Detection](question_3_faster_rcnn/README.md)

Runs image object detection with a local clone of `trzy/FasterRCNN`.

- Setup: `bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh` installs both runtimes by default
- Models: `bash question_3_faster_rcnn/download_models_fasterrcnn.sh`
- OSC GPU run: `bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- python3.9 question_3_faster_rcnn/demo_fasterrcnn.py --framework pytorch --mode to-file`
- CPU-capable run: `python3.9 question_3_faster_rcnn/demo_fasterrcnn.py --mode to-file` auto-selects CUDA PyTorch or the TF2 fallback
- OSC CPU-style sessions should normally resolve to the TF2 fallback; remove `external/FasterRCNN/.venv` and rerun setup if an older environment shows NumPy/Matplotlib import errors

### [YOLO11 and YOLOv12 Object Detection](question_4_yolo11_yolov12/README.md)

Provides separate wrappers for YOLO11 and YOLOv12 with dedicated inputs and
outputs for each model.

- YOLO11 setup: `bash question_4_yolo11_yolov12/setup_yolo11_osc.sh`
- YOLO11 OSC GPU run: `bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- python3.9 question_4_yolo11_yolov12/demo_yolo11.py --device 0`
- YOLO11 CPU run: `python3.9 question_4_yolo11_yolov12/demo_yolo11.py --device cpu`
- YOLOv12 setup: `bash question_4_yolo11_yolov12/setup_yolov12_osc.sh`
- YOLOv12 OSC GPU run: `bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- python3.9 question_4_yolo11_yolov12/demo_yolov12.py --device 0`
- YOLOv12 CPU run: `python3.9 question_4_yolo11_yolov12/demo_yolov12.py --device cpu`

### [Semantic Segmentation with YOLO11 and iSAID](question_5_semantic_segmentation/README.md)

Bootstraps iSAID data, fine-tunes YOLO11 segmentation, and exports semantic
masks for aerial imagery.

- Setup: `bash question_5_semantic_segmentation/setup_yolo11_osc.sh`
- OSC GPU train: `bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 04:00:00 -- external/ultralytics/.venv/bin/python question_5_semantic_segmentation/train_isaid_seg.py --repo-dir external/ultralytics --device 0`
- Demo: `external/ultralytics/.venv/bin/python question_5_semantic_segmentation/demo_yolo_segmentation.py --repo-dir external/ultralytics --device cpu`

### [Directional Convolution Filters](question_6_convolution_filters/README.md)

Derives six directional convolution kernels and applies them to synthetic
patterns or real images.

- Synthetic demo: `python3.9 question_6_convolution_filters/demo_convolution_filters.py --size 11`
- Image demo: `python3.9 question_6_convolution_filters/demo_convolution_filters_image.py --source question_6_convolution_filters/inputs --output-dir question_6_convolution_filters/outputs`

### [Transfer Learning Object Detection with TLlib](question_7_transfer_learning/README.md)

Wraps TLlib's VOC-to-Clipart domain adaptation example with environment checks,
dataset preparation, training, and reporting.

- Setup: `bash question_7_transfer_learning/setup_tllib_osc.sh`
- Doctor: `python3.9 question_7_transfer_learning/demo_tllib_object_detection.py --mode doctor`
- OSC GPU full run: `bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 04:00:00 -- bash question_7_transfer_learning/run_tllib_osc.sh`
- CPU smoke validation: `ALLOW_CPU=1 PROFILE=smoke bash question_7_transfer_learning/run_tllib_osc.sh`
