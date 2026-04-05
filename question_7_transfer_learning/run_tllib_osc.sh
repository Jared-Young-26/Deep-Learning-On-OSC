#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TLIB_REPO_DIR="${TLIB_REPO_DIR:-${REPO_ROOT}/external/Transfer-Learning-Library}"
PYTHON_BIN="${PYTHON_BIN:-${TLIB_REPO_DIR}/.venv/bin/python}"
ALLOW_CPU="${ALLOW_CPU:-0}"
FORCE="${FORCE:-0}"
PROFILE="${PROFILE:-benchmark}"

# This helper is the shortest way to run the full Question 7 wrapper on a prepared environment.
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: ${PYTHON_BIN} was not found or is not executable."
  echo "Run bash question_7_transfer_learning/setup_tllib_osc.sh first."
  exit 1
fi

DEVICE="$("${PYTHON_BIN}" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")"
# Benchmark runs are intended for the OSC GPU workflow. Requiring CUDA by default
# prevents accidentally launching a long CPU-only job unless the user opts in.
if [[ "${DEVICE}" != "cuda" && "${ALLOW_CPU}" != "1" ]]; then
  echo "Error: CUDA is not available in ${PYTHON_BIN}."
  echo "Run this benchmark helper on an OSC GPU node, or set ALLOW_CPU=1 to override."
  exit 2
fi

# Decide once whether this run is using GPU or a slower CPU fallback.
RUN_DEVICE="${DEVICE}"
if [[ "${RUN_DEVICE}" != "cuda" ]]; then
  echo "Warning: CUDA is unavailable. Continuing with MODEL.DEVICE=${RUN_DEVICE}."
  if [[ "${PROFILE}" == "benchmark" ]]; then
    echo "Warning: benchmark on CPU can be extremely slow. For local validation, prefer PROFILE=smoke."
  fi
fi

# Forward any extra CLI args to the Python wrapper so one shell helper can still
# expose flags like --force, custom output dirs, or smoke-profile tweaks.
EXTRA_ARGS=()
if [[ "$#" -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi
if [[ "${FORCE}" == "1" ]]; then
  EXTRA_ARGS=(--force "${EXTRA_ARGS[@]}")
fi

# Always run the doctor preflight first so missing dependencies fail fast.
echo "Question 7 doctor check:"
"${PYTHON_BIN}" "${SCRIPT_DIR}/demo_tllib_object_detection.py" \
  --repo-dir "${TLIB_REPO_DIR}" \
  --mode doctor

echo
echo "Question 7 ${PROFILE} pipeline:"
# The second call does the real work: prepare datasets if needed, train the
# baseline, run D-adapt phases, render visualizations, and write the summary.
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
