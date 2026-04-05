#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/FasterRCNN"

REPO_URL="${REPO_URL:-https://github.com/trzy/FasterRCNN.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
OS_NAME="$(uname -s)"
PYTHON_BIN="${PYTHON_BIN:-}"

# Default to installing TF2 on macOS because that is the runnable local path for
# this assignment, while Linux/OSC users can rely on the PyTorch implementation.
if [[ -z "${INSTALL_TF2:-}" ]]; then
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    INSTALL_TF2="1"
  else
    INSTALL_TF2="0"
  fi
fi

# Prefer Python 3.11 when available because it is a stable target for both the
# venv tooling and the older upstream dependency pins used by this project.
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ "${REPO_DIR}" != /* ]]; then
  # Accept a relative repo target but normalize it before clone/setup begins.
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found."
  echo "Set PYTHON_BIN to a valid Python interpreter, ideally Python 3.11."
  exit 1
fi

# Clone once into external/ so the question folder only contains wrapper code.
mkdir -p "$(dirname "${REPO_DIR}")"

if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "Using existing clone at ${REPO_DIR}"
else
  echo "Cloning ${REPO_URL} into ${REPO_DIR}"
  git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Match the selected interpreter to the virtual environment to avoid subtle package
# conflicts between multiple Python versions on the same machine.
SELECTED_PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Using Python interpreter: ${PYTHON_BIN} (${SELECTED_PYTHON_VERSION})"

if [[ -x ".venv/bin/python" ]]; then
  EXISTING_VENV_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "${EXISTING_VENV_VERSION}" != "${SELECTED_PYTHON_VERSION}" ]]; then
    echo "Existing virtual environment uses Python ${EXISTING_VENV_VERSION}, but setup selected ${SELECTED_PYTHON_VERSION}."
    echo "Remove ${REPO_DIR}/.venv and rerun, or set PYTHON_BIN to match the existing environment."
    exit 1
  fi
fi

# Recreate the virtual environment from the chosen interpreter so the demo is
# reproducible and not affected by stale packages from previous attempts.
echo "Creating virtual environment at ${REPO_DIR}/.venv"
"${PYTHON_BIN}" -m venv --clear .venv
source .venv/bin/activate

# Install the minimal runtime that keeps the upstream repo usable on both OSC/Linux
# and a local macOS walkthrough machine.
python -m pip install --upgrade pip setuptools wheel

if [[ "${OS_NAME}" == "Darwin" ]]; then
  # On macOS install a portable CPU-friendly stack instead of the upstream Linux
  # requirements file, which assumes CUDA-specific wheels.
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
  # On Linux/OSC the upstream PyTorch requirements are the closest match to the
  # original project instructions, including CUDA-enabled packages when available.
  pip install -r pytorch/requirements.txt
fi

if [[ "${INSTALL_TF2}" == "1" ]]; then
  # TensorFlow is optional overall, but it is the key local demo path on macOS.
  echo "Installing TensorFlow dependencies."
  pip install -r tf2/requirements.txt
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    echo "Re-pinning NumPy below 2 for TensorFlow/Matplotlib compatibility on macOS."
    pip install "numpy<2"
  fi
fi

# Finish with the exact next commands needed for the demo run.
cat <<EOF
Setup complete.

Next steps:
  1) source "${REPO_DIR}/.venv/bin/activate"
  2) cd "${SCRIPT_DIR}"
  3) bash download_models_fasterrcnn.sh
  4) cd "${SCRIPT_DIR}"
  5) python3 demo_fasterrcnn.py --repo-dir "${REPO_DIR}"
EOF
