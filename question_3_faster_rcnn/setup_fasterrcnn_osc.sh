#!/usr/bin/env bash
set -euo pipefail

# Prepare the repo-local FasterRCNN clone and virtual environment used by the
# tracked Q3 wrapper. The setup keeps both the CUDA-backed PyTorch path and the
# optional TF2 CPU fallback under one reproducible OSC baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/FasterRCNN"

REPO_URL="${REPO_URL:-https://github.com/trzy/FasterRCNN.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
OS_NAME="$(uname -s)"
PYTHON_BIN="${PYTHON_BIN:-}"
TF2_NUMPY_SPEC="${TF2_NUMPY_SPEC:-numpy>=1.26,<2}"
SUPPORTED_PYTHON_VERSION="${SUPPORTED_PYTHON_VERSION:-3.9.18}"

# Install the TF2 fallback by default so one OSC setup command prepares both the
# CUDA PyTorch path and the CPU-capable fallback runtime.
if [[ -z "${INSTALL_TF2:-}" ]]; then
  INSTALL_TF2="1"
fi

# Prefer Python 3.9 when it is available on OSC.
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.9 >/dev/null 2>&1; then
    PYTHON_BIN="python3.9"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ "${REPO_DIR}" != /* ]]; then
  # Resolve relative clone targets before the rest of the script uses the path.
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

# Stop immediately if the requested interpreter is unavailable.
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found."
  echo "Set PYTHON_BIN to a valid Python ${SUPPORTED_PYTHON_VERSION} interpreter."
  exit 1
fi

# Clone or reuse the upstream repository before rebuilding the environment.
# Create the parent directory before cloning into it.
mkdir -p "$(dirname "${REPO_DIR}")"

# Reuse the existing clone when it is already present.
if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "Using existing clone at ${REPO_DIR}"
else
  echo "Cloning ${REPO_URL} into ${REPO_DIR}"
  git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Capture the selected interpreter version before reusing or rebuilding the environment.
SELECTED_PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"
echo "Using Python interpreter: ${PYTHON_BIN} (${SELECTED_PYTHON_VERSION})"

if [[ "${SELECTED_PYTHON_VERSION}" != "${SUPPORTED_PYTHON_VERSION}" ]]; then
  echo "Error: FasterRCNN OSC setup requires Python ${SUPPORTED_PYTHON_VERSION}, but selected ${SELECTED_PYTHON_VERSION}."
  echo "Load Python ${SUPPORTED_PYTHON_VERSION} on OSC or set PYTHON_BIN to a Python ${SUPPORTED_PYTHON_VERSION} executable."
  exit 1
fi

if [[ -x ".venv/bin/python" ]]; then
  # Refuse to reuse an environment built with a different Python minor version.
  EXISTING_VENV_VERSION="$(.venv/bin/python -c 'import sys; print(sys.version.split()[0])')"
  if [[ "${EXISTING_VENV_VERSION}" != "${SELECTED_PYTHON_VERSION}" ]]; then
    echo "Existing virtual environment uses Python ${EXISTING_VENV_VERSION}, but setup selected ${SELECTED_PYTHON_VERSION}."
    echo "Remove ${REPO_DIR}/.venv and rerun, or set PYTHON_BIN to match the existing environment."
    exit 1
  fi
fi

# Rebuild the virtual environment from the selected interpreter.
echo "Creating virtual environment at ${REPO_DIR}/.venv"
"${PYTHON_BIN}" -m venv --clear .venv

# Activate the environment before installing packages into it.
source .venv/bin/activate

# Update the packaging tools before installing the runtime stack.
python -m pip install --upgrade pip setuptools wheel

if [[ "${OS_NAME}" == "Darwin" ]]; then
  # Install a platform-compatible package set on macOS.
  echo "Detected macOS. The upstream FasterRCNN PyTorch requirements pin Linux CUDA wheels."
  echo "Installing portable PyTorch dependencies from PyPI instead."
  pip install \
    h5py==3.8.0 \
    imageio==2.26.0 \
    matplotlib==3.7.1 \
    numpy==1.24.2 \
    tqdm==4.65.0
  pip install torch torchvision
else
  # Use the upstream PyTorch requirements on Linux systems.
  pip install -r pytorch/requirements.txt
fi

if [[ "${INSTALL_TF2}" == "1" ]]; then
  # Keep the shared TF2/PyTorch environment on a NumPy 1.x line so the
  # compiled Matplotlib wheel remains compatible on OSC Python 3.9.
  echo "Pinning shared NumPy runtime to ${TF2_NUMPY_SPEC}."
  pip install "${TF2_NUMPY_SPEC}"
  TF2_CONSTRAINTS_FILE="$(mktemp)"
  trap 'rm -f "${TF2_CONSTRAINTS_FILE}"' EXIT
  printf '%s\n' "${TF2_NUMPY_SPEC}" > "${TF2_CONSTRAINTS_FILE}"

  # Install the TensorFlow dependency set when it is enabled.
  echo "Installing TensorFlow dependencies."
  pip install -c "${TF2_CONSTRAINTS_FILE}" -r tf2/requirements.txt

  # Reinstall the NumPy-sensitive compiled packages after the final NumPy choice
  # so the TF2 fallback is not left in a mixed NumPy-major state.
  echo "Reinstalling NumPy-sensitive wheels for the final TF2 runtime."
  pip install --force-reinstall -c "${TF2_CONSTRAINTS_FILE}" \
    "matplotlib==3.7.1" \
    h5py

  echo "Validating TF2 fallback environment."
  if ! VALIDATION_OUTPUT="$(
    CUDA_VISIBLE_DEVICES="-1" TF_CPP_MIN_LOG_LEVEL="2" "${REPO_DIR}/.venv/bin/python" - <<'PY'
import sys

import matplotlib
import matplotlib.pyplot  # noqa: F401
import numpy
import tensorflow as tf
import torch

print("Environment validation:")
print(f"  python: {sys.version.split()[0]}")
print(f"  numpy: {numpy.__version__}")
print(f"  matplotlib: {matplotlib.__version__}")
print(f"  tensorflow: {tf.__version__}")
print(f"  torch: {getattr(torch, '__version__', 'unknown')}")
PY
  )"; then
    echo "Error: TF2 fallback validation failed in ${REPO_DIR}/.venv." >&2
    echo "The repo-local FasterRCNN environment must import numpy, matplotlib.pyplot, and tensorflow together." >&2
    echo "Remove ${REPO_DIR}/.venv and rerun bash question_3_faster_rcnn/setup_fasterrcnn_osc.sh." >&2
    exit 1
  fi
  printf '%s\n' "${VALIDATION_OUTPUT}"
fi

# Finish by printing the shortest manual next-step sequence for the local clone.
# Print the next commands for the prepared environment.
cat <<EOF
Setup complete.

Next steps:
  1) source "${REPO_DIR}/.venv/bin/activate"
  2) cd "${SCRIPT_DIR}"
  3) bash download_models_fasterrcnn.sh
  4) cd "${SCRIPT_DIR}"
  5) python3.9 demo_fasterrcnn.py --repo-dir "${REPO_DIR}"
EOF
