#!/usr/bin/env bash
set -euo pipefail

# Run the repo-owned Q7 helper flow on OSC. The wrapper first repairs or
# bootstraps the TLlib clone, then runs `doctor`, then launches the selected
# full pipeline profile with the resolved device configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TLIB_REPO_DIR="${TLIB_REPO_DIR:-${REPO_ROOT}/external/Transfer-Learning-Library}"
PREFLIGHT_SCRIPT="${REPO_ROOT}/osc_gpu_preflight.sh"
SETUP_SCRIPT="${SCRIPT_DIR}/setup_tllib_osc.sh"
PYTHON_BIN="${PYTHON_BIN:-${TLIB_REPO_DIR}/.venv/bin/python}"
ALLOW_CPU="${ALLOW_CPU:-0}"
FORCE="${FORCE:-0}"
PROFILE="${PROFILE:-smoke}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  # A missing runtime means setup must build the environment from scratch.
  echo "TLlib runtime missing at ${PYTHON_BIN}; running full setup."
  env INSTALL_TORCH=1 INSTALL_DETECTRON2=1 \
    bash "${SETUP_SCRIPT}" "${TLIB_REPO_DIR}"
else
  # When the environment already exists, prefer the cheaper repair-only path so
  # compatibility fixes are refreshed before doctor or training starts.
  echo "TLlib repair-only preflight (verification failure stops the run):"
  env REPAIR_ONLY=1 \
    bash "${SETUP_SCRIPT}" "${TLIB_REPO_DIR}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: ${PYTHON_BIN} was not found or is not executable after Q7 setup."
  exit 1
fi

if [[ "${ALLOW_CPU}" != "1" ]]; then
  # Require a real OSC GPU allocation before benchmark-style execution.
  # shellcheck disable=SC1090
  source "${PREFLIGHT_SCRIPT}"
  osc_require_gpu_allocation "question_7_transfer_learning/run_tllib_osc.sh"
fi

DEVICE="$("${PYTHON_BIN}" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")"
# Require CUDA by default unless CPU execution was requested explicitly.
if [[ "${DEVICE}" != "cuda" && "${ALLOW_CPU}" != "1" ]]; then
  echo "Error: CUDA is not available in ${PYTHON_BIN}."
  echo "A GPU allocation exists, but the TLlib runtime still resolved to CPU."
  echo "Rebuild the environment on the allocated node, or set ALLOW_CPU=1 for CPU validation."
  exit 2
fi

# Keep one resolved device string for the Python driver call below.
RUN_DEVICE="${DEVICE}"
if [[ "${RUN_DEVICE}" != "cuda" ]]; then
  echo "Warning: CUDA is unavailable. Continuing with MODEL.DEVICE=${RUN_DEVICE}."
  if [[ "${PROFILE}" == "benchmark" ]]; then
    echo "Warning: benchmark on CPU can be extremely slow. For local validation, prefer PROFILE=smoke."
  fi
fi

# Forward any extra CLI flags to the Python driver.
EXTRA_ARGS=()
if [[ "$#" -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi
if [[ "${FORCE}" == "1" ]]; then
  EXTRA_ARGS=(--force "${EXTRA_ARGS[@]}")
fi

# Keep the explicit environment check in the shell transcript before the
# longer-running training stages begin.
# Run the environment check before starting the full pipeline.
echo "TLlib doctor check:"
"${PYTHON_BIN}" "${SCRIPT_DIR}/demo_tllib_object_detection.py" \
  --repo-dir "${TLIB_REPO_DIR}" \
  --mode doctor

echo
echo "TLlib ${PROFILE} pipeline:"
# Run the full pipeline with the resolved profile and device.
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/demo_tllib_object_detection.py" \
    --repo-dir "${TLIB_REPO_DIR}" \
    --mode full-pipeline \
    --profile "${PROFILE}" \
    --download-datasets \
    --device "${RUN_DEVICE}" \
    "${EXTRA_ARGS[@]}"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/demo_tllib_object_detection.py" \
    --repo-dir "${TLIB_REPO_DIR}" \
    --mode full-pipeline \
    --profile "${PROFILE}" \
    --download-datasets \
    --device "${RUN_DEVICE}"
fi
