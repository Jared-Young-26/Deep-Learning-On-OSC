#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/FasterRCNN"

REPO_URL="${REPO_URL:-https://github.com/trzy/FasterRCNN.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
OS_NAME="$(uname -s)"
PYTHON_BIN="${PYTHON_BIN:-}"

# Install the TF2 fallback by default so one OSC setup command prepares both the
# CUDA PyTorch path and the CPU-capable fallback runtime.
if [[ -z "${INSTALL_TF2:-}" ]]; then
  INSTALL_TF2="1"
fi

# Prefer Python 3.11 when it is available.
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
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
  echo "Set PYTHON_BIN to a valid Python interpreter, ideally Python 3.11."
  exit 1
fi

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
SELECTED_PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Using Python interpreter: ${PYTHON_BIN} (${SELECTED_PYTHON_VERSION})"

if [[ -x ".venv/bin/python" ]]; then
  # Refuse to reuse an environment built with a different Python minor version.
  EXISTING_VENV_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
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
  # Install the TensorFlow dependency set when it is enabled.
  echo "Installing TensorFlow dependencies."
  pip install -r tf2/requirements.txt
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    echo "Re-pinning NumPy below 2 for TensorFlow/Matplotlib compatibility on macOS."
    pip install "numpy<2"
  fi
fi

# Print the next commands for the prepared environment.
cat <<EOF
Setup complete.

Next steps:
  1) source "${REPO_DIR}/.venv/bin/activate"
  2) cd "${SCRIPT_DIR}"
  3) bash download_models_fasterrcnn.sh
  4) cd "${SCRIPT_DIR}"
  5) python3 demo_fasterrcnn.py --repo-dir "${REPO_DIR}"
EOF
