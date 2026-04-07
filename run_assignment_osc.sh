#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
OSC_GPU_BATCH_LAUNCHER="${REPO_ROOT}/osc_gpu_batch.sh"
OSC_GPU_PREFLIGHT="${REPO_ROOT}/osc_gpu_preflight.sh"

SUPPORTED_PYTHON_VERSION="3.9.18"

Q3_REPO_PYTHON="${REPO_ROOT}/external/FasterRCNN/.venv/bin/python"
Q4_ULTRALYTICS_PYTHON="${REPO_ROOT}/external/ultralytics/.venv/bin/python"
Q4_YOLOV12_PYTHON="${REPO_ROOT}/external/yolov12/.venv/bin/python"
Q5_MODEL_ALIAS="${REPO_ROOT}/question_5_semantic_segmentation/models/isaid_seg/best.pt"
Q5_PRETRAINED_MODEL="${REPO_ROOT}/question_5_semantic_segmentation/models/pretrained/yolo11s-seg.pt"
Q5_DATASET_YAML="${REPO_ROOT}/question_5_semantic_segmentation/datasets/isaid_seg.yaml"
Q5_TRAIN_PROJECT_DIR="${REPO_ROOT}/question_5_semantic_segmentation/runs/segment/train"
Q5_TRAIN_RUN_NAME="isaid_yolo11s_seg"
Q5_LAST_CHECKPOINT="${Q5_TRAIN_PROJECT_DIR}/${Q5_TRAIN_RUN_NAME}/weights/last.pt"
Q7_REPO_PYTHON="${REPO_ROOT}/external/Transfer-Learning-Library/.venv/bin/python"

ACCOUNT=""
TIME_LIMIT=""
CLUSTER=""
NODES="1"
GPUS_PER_NODE="1"
JOB_NAME=""
Q7_PROFILE="benchmark"
FORCE="0"
DRY_RUN="0"
INSIDE_ALLOCATION="0"

CURRENT_STAGE=""
STAGE_COUNT=0

usage() {
  cat <<'EOF'
Usage:
  bash run_assignment_osc.sh --account <OSC_ACCOUNT> --time <HH:MM:SS> [options]

Options:
  --account <value>          Required when self-submitting from an OSC login node.
  --time <HH:MM:SS>          Required when self-submitting from an OSC login node.
  --cluster <name>           Optional Slurm cluster forwarded to osc_gpu_batch.sh.
  --nodes <count>            Slurm node count. Default: 1.
  --gpus-per-node <count>    GPUs per node. Default: 1.
  --job-name <name>          Optional Slurm job name forwarded to osc_gpu_batch.sh.
  --q7-profile <profile>     TLlib profile: benchmark or smoke. Default: benchmark.
  --force                    Rerun heavy setup/train/pipeline stages even when sentinels exist.
  --dry-run                  Print the submit command or inner stage plan without executing it.
  --inside-allocation        Internal flag used after self-submit; runs the stage pipeline directly.
  --help, -h                 Show this help text.
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  local level="$1"
  shift
  printf '[%s] [%s] %s\n' "$(timestamp)" "${level}" "$*"
}

quote_command() {
  local quoted=()
  local part
  for part in "$@"; do
    quoted+=("$(printf '%q' "${part}")")
  done
  printf '%s' "${quoted[*]}"
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  if [[ -n "${CURRENT_STAGE}" ]]; then
    echo "Error: stage failed: ${CURRENT_STAGE}" >&2
  fi
  exit "${exit_code}"
}
trap on_error ERR

start_stage() {
  CURRENT_STAGE="$1"
  STAGE_COUNT=$((STAGE_COUNT + 1))
  printf '\n'
  log "STAGE" "${CURRENT_STAGE}"
}

finish_stage() {
  log "DONE" "${CURRENT_STAGE}"
}

run_cmd() {
  log "RUN" "$(quote_command "$@")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    fail "required file is missing after ${CURRENT_STAGE}: ${path}"
  fi
}

python_version_matches() {
  local python_bin="$1"
  [[ -x "${python_bin}" ]] || return 1
  [[ "$("${python_bin}" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || true)" == "${SUPPORTED_PYTHON_VERSION}" ]]
}

python_imports_available() {
  local python_bin="$1"
  shift
  [[ -x "${python_bin}" ]] || return 1
  local imports=()
  local module
  for module in "$@"; do
    imports+=("import ${module}")
  done
  "${python_bin}" -c "$(printf '%s; ' "${imports[@]}")" >/dev/null 2>&1
}

in_slurm_allocation() {
  [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_STEP_ID:-}" ]]
}

q7_summary_path() {
  printf '%s/question_7_transfer_learning/outputs/voc2clipart_%s/summary.json' "${REPO_ROOT}" "${Q7_PROFILE}"
}

resolve_q6_python() {
  if [[ -x "${Q4_ULTRALYTICS_PYTHON}" ]]; then
    printf '%s\n' "${Q4_ULTRALYTICS_PYTHON}"
  else
    printf '%s\n' "python3.9"
  fi
}

submit_self() {
  local submit_cmd=(
    bash "${OSC_GPU_BATCH_LAUNCHER}"
    --account "${ACCOUNT}"
    --time "${TIME_LIMIT}"
    --nodes "${NODES}"
    --gpus-per-node "${GPUS_PER_NODE}"
    --workdir "${REPO_ROOT}"
  )

  if [[ -n "${CLUSTER}" ]]; then
    submit_cmd+=(--cluster "${CLUSTER}")
  fi
  if [[ -n "${JOB_NAME}" ]]; then
    submit_cmd+=(--job-name "${JOB_NAME}")
  fi

  local inner_cmd=(
    bash "${REPO_ROOT}/run_assignment_osc.sh"
    --inside-allocation
    --q7-profile "${Q7_PROFILE}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    inner_cmd+=(--force)
  fi

  submit_cmd+=(-- "${inner_cmd[@]}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    quote_command "${submit_cmd[@]}"
    printf '\n'
    return 0
  fi

  log "SUBMIT" "$(quote_command "${submit_cmd[@]}")"
  "${submit_cmd[@]}"
}

stage_q3_readiness() {
  start_stage "Q3 readiness"

  local setup_needed="0"
  if ! python_version_matches "${Q3_REPO_PYTHON}"; then
    setup_needed="1"
    log "INFO" "FasterRCNN runtime is missing or not Python ${SUPPORTED_PYTHON_VERSION}; running setup."
    run_cmd bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh
    if [[ "${DRY_RUN}" != "1" ]]; then
      python_version_matches "${Q3_REPO_PYTHON}" || fail "Q3 setup did not produce a Python ${SUPPORTED_PYTHON_VERSION} runtime at ${Q3_REPO_PYTHON}"
    fi
  else
    log "SKIP" "Q3 setup already satisfied."
  fi

  local q3_weights=(
    "${REPO_ROOT}/external/FasterRCNN/vgg16_caffe.pth"
    "${REPO_ROOT}/external/FasterRCNN/fasterrcnn_pytorch_vgg16.pth"
    "${REPO_ROOT}/external/FasterRCNN/fasterrcnn_tf2.h5"
    "${REPO_ROOT}/external/FasterRCNN/fasterrcnn_pytorch_resnet50.pth"
  )
  local missing_weights=()
  local weight
  for weight in "${q3_weights[@]}"; do
    if [[ ! -f "${weight}" ]]; then
      missing_weights+=("${weight}")
    fi
  done

  if (( ${#missing_weights[@]} > 0 )); then
    if [[ "${setup_needed}" == "1" ]]; then
      log "INFO" "Q3 setup completed; downloading missing model weights."
    else
      log "INFO" "Missing Q3 model weights detected; downloading them."
    fi
    run_cmd bash question_3_faster_rcnn/download_models_fasterrcnn.sh
    if [[ "${DRY_RUN}" != "1" ]]; then
      for weight in "${q3_weights[@]}"; do
        require_file "${weight}"
      done
    fi
  else
    log "SKIP" "Q3 model weights already present."
  fi

  finish_stage
}

stage_q3_execution() {
  start_stage "Q3 execution"
  run_cmd python3.9 question_3_faster_rcnn/demo_fasterrcnn.py \
    --image question_3_faster_rcnn/inputs \
    --framework pytorch \
    --mode to-file
  finish_stage
}

stage_q4_yolo11_readiness() {
  start_stage "Q4 YOLO11 readiness"
  if ! python_version_matches "${Q4_ULTRALYTICS_PYTHON}"; then
    log "INFO" "Ultralytics runtime is missing or not Python ${SUPPORTED_PYTHON_VERSION}; running YOLO11 setup."
    run_cmd bash question_4_yolo11_yolov12/setup_yolo11_osc.sh
    if [[ "${DRY_RUN}" != "1" ]]; then
      python_version_matches "${Q4_ULTRALYTICS_PYTHON}" || fail "Q4 YOLO11 setup did not produce a Python ${SUPPORTED_PYTHON_VERSION} runtime at ${Q4_ULTRALYTICS_PYTHON}"
    fi
  else
    log "SKIP" "Q4 YOLO11 setup already satisfied."
  fi
  finish_stage
}

stage_q4_yolo11_execution() {
  start_stage "Q4 YOLO11 execution"
  run_cmd python3.9 question_4_yolo11_yolov12/demo_yolo11.py --device 0
  finish_stage
}

stage_q4_yolov12_readiness() {
  start_stage "Q4 YOLOv12 readiness"
  if ! python_version_matches "${Q4_YOLOV12_PYTHON}"; then
    log "INFO" "YOLOv12 runtime is missing or not Python ${SUPPORTED_PYTHON_VERSION}; running YOLOv12 setup."
    run_cmd bash question_4_yolo11_yolov12/setup_yolov12_osc.sh
    if [[ "${DRY_RUN}" != "1" ]]; then
      python_version_matches "${Q4_YOLOV12_PYTHON}" || fail "Q4 YOLOv12 setup did not produce a Python ${SUPPORTED_PYTHON_VERSION} runtime at ${Q4_YOLOV12_PYTHON}"
    fi
  else
    log "SKIP" "Q4 YOLOv12 setup already satisfied."
  fi
  finish_stage
}

stage_q4_yolov12_execution() {
  start_stage "Q4 YOLOv12 execution"
  run_cmd python3.9 question_4_yolo11_yolov12/demo_yolov12.py --device 0
  finish_stage
}

stage_q5_readiness() {
  start_stage "Q5 readiness"
  if ! python_version_matches "${Q4_ULTRALYTICS_PYTHON}" || [[ ! -f "${Q5_DATASET_YAML}" ]] || [[ ! -f "${Q5_PRETRAINED_MODEL}" ]]; then
    log "INFO" "Q5 prerequisites are incomplete; running YOLO11 segmentation setup."
    run_cmd bash question_5_semantic_segmentation/setup_yolo11_osc.sh
    if [[ "${DRY_RUN}" != "1" ]]; then
      python_version_matches "${Q4_ULTRALYTICS_PYTHON}" || fail "Q5 setup did not leave a Python ${SUPPORTED_PYTHON_VERSION} Ultralytics runtime at ${Q4_ULTRALYTICS_PYTHON}"
      require_file "${Q5_DATASET_YAML}"
      require_file "${Q5_PRETRAINED_MODEL}"
    fi
  else
    log "SKIP" "Q5 setup already satisfied."
  fi
  finish_stage
}

stage_q5_training() {
  start_stage "Q5 training"
  local q5_train_cmd=(
    "${Q4_ULTRALYTICS_PYTHON}" question_5_semantic_segmentation/train_isaid_seg.py
    --repo-dir external/ultralytics
    --device 0
    --imgsz 1024
    --batch 1
    --workers 0
    --exist-ok
  )

  if [[ -f "${Q5_MODEL_ALIAS}" && "${FORCE}" != "1" ]]; then
    log "SKIP" "Q5 training skipped because best.pt exists at ${Q5_MODEL_ALIAS}."
  elif [[ "${FORCE}" == "1" ]]; then
    log "INFO" "Q5 training starting fresh because --force was set."
    run_cmd "${q5_train_cmd[@]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      require_file "${Q5_MODEL_ALIAS}"
    fi
  elif [[ -f "${Q5_LAST_CHECKPOINT}" ]]; then
    log "INFO" "Q5 training resuming from last.pt at ${Q5_LAST_CHECKPOINT}."
    run_cmd "${q5_train_cmd[@]}" --resume
    if [[ "${DRY_RUN}" != "1" ]]; then
      require_file "${Q5_MODEL_ALIAS}"
    fi
  else
    log "INFO" "Q5 training starting fresh because no reusable checkpoint exists yet."
    run_cmd "${q5_train_cmd[@]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      require_file "${Q5_MODEL_ALIAS}"
    fi
  fi
  finish_stage
}

stage_q5_demo() {
  start_stage "Q5 demo"
  run_cmd "${Q4_ULTRALYTICS_PYTHON}" question_5_semantic_segmentation/demo_yolo_segmentation.py \
    --repo-dir external/ultralytics \
    --device 0
  finish_stage
}

stage_q6_execution() {
  start_stage "Q6 execution"
  local q6_python
  q6_python="$(resolve_q6_python)"
  log "INFO" "Using Q6 interpreter: ${q6_python}"
  run_cmd "${q6_python}" question_6_convolution_filters/demo_convolution_filters.py --size 11
  run_cmd "${q6_python}" question_6_convolution_filters/demo_convolution_filters_image.py \
    --source question_6_convolution_filters/inputs \
    --output-dir question_6_convolution_filters/outputs
  finish_stage
}

stage_q7_readiness() {
  start_stage "Q7 readiness"
  if ! python_imports_available "${Q7_REPO_PYTHON}" torch timm detectron2; then
    log "INFO" "Q7 runtime is missing or incomplete; running TLlib setup with torch and detectron2."
    run_cmd env INSTALL_TORCH=1 INSTALL_DETECTRON2=1 bash question_7_transfer_learning/setup_tllib_osc.sh
    if [[ "${DRY_RUN}" != "1" ]]; then
      python_imports_available "${Q7_REPO_PYTHON}" torch timm detectron2 || fail "Q7 setup did not produce a working torch+timm+detectron2 runtime at ${Q7_REPO_PYTHON}"
    fi
  else
    log "SKIP" "Q7 setup already satisfied."
  fi
  finish_stage
}

stage_q7_execution() {
  start_stage "Q7 execution"
  local summary_path
  summary_path="$(q7_summary_path)"

  if [[ "${FORCE}" == "1" || ! -f "${summary_path}" ]]; then
    if [[ "${FORCE}" == "1" && -f "${summary_path}" ]]; then
      log "INFO" "Force enabled; rerunning Q7 ${Q7_PROFILE} pipeline despite existing summary."
    else
      log "INFO" "Missing Q7 ${Q7_PROFILE} summary; running pipeline."
    fi
    run_cmd env PROFILE="${Q7_PROFILE}" bash question_7_transfer_learning/run_tllib_osc.sh
    if [[ "${DRY_RUN}" != "1" ]]; then
      require_file "${summary_path}"
    fi
  else
    log "SKIP" "Q7 ${Q7_PROFILE} summary already present at ${summary_path}."
  fi
  finish_stage
}

run_pipeline() {
  stage_q3_readiness
  stage_q3_execution
  stage_q4_yolo11_readiness
  stage_q4_yolo11_execution
  stage_q4_yolov12_readiness
  stage_q4_yolov12_execution
  stage_q5_readiness
  stage_q5_training
  stage_q5_demo
  stage_q6_execution
  stage_q7_readiness
  stage_q7_execution
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --account)
        ACCOUNT="${2:-}"
        shift 2
        ;;
      --time)
        TIME_LIMIT="${2:-}"
        shift 2
        ;;
      --cluster)
        CLUSTER="${2:-}"
        shift 2
        ;;
      --nodes)
        NODES="${2:-}"
        shift 2
        ;;
      --gpus-per-node)
        GPUS_PER_NODE="${2:-}"
        shift 2
        ;;
      --job-name)
        JOB_NAME="${2:-}"
        shift 2
        ;;
      --q7-profile)
        Q7_PROFILE="${2:-}"
        shift 2
        ;;
      --force)
        FORCE="1"
        shift
        ;;
      --dry-run)
        DRY_RUN="1"
        shift
        ;;
      --inside-allocation)
        INSIDE_ALLOCATION="1"
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        fail "unrecognized option: $1"
        ;;
    esac
  done

  if [[ "${Q7_PROFILE}" != "benchmark" && "${Q7_PROFILE}" != "smoke" ]]; then
    fail "--q7-profile must be either benchmark or smoke"
  fi
}

main() {
  parse_args "$@"
  cd "${REPO_ROOT}"

  # shellcheck disable=SC1090
  source "${OSC_GPU_PREFLIGHT}"

  if [[ "${INSIDE_ALLOCATION}" != "1" ]]; then
    if in_slurm_allocation && osc_slurm_gpu_signals_present && osc_visible_gpu_present; then
      :
    else
      [[ -n "${ACCOUNT}" && -n "${TIME_LIMIT}" ]] || fail "--account and --time are required when self-submitting from a login node or a CPU-only Slurm session"
      if in_slurm_allocation; then
        log "INFO" "Current Slurm session does not expose a usable GPU; self-submitting a new GPU job."
      fi
      submit_self
      return 0
    fi
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "INFO" "Dry run inside allocation mode; skipping GPU preflight."
  else
    osc_prepare_gpu_environment
    osc_require_gpu_allocation "run_assignment_osc.sh"
  fi

  run_pipeline

  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "SUMMARY" "Dry run complete for ${STAGE_COUNT} stages. Q7 profile: ${Q7_PROFILE}."
  else
    log "SUMMARY" "Completed ${STAGE_COUNT} stages. Q7 profile: ${Q7_PROFILE}."
  fi
}

main "$@"
