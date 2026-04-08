#!/usr/bin/env bash
set -euo pipefail

# Prepare the repo-local YOLOv12 clone used by the tracked wrapper. This setup
# intentionally pins a narrower dependency set than the upstream project so the
# OSC path stays reproducible on the shared Python 3.9.18 baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/yolov12"

REPO_URL="${REPO_URL:-https://github.com/sunsmarterjie/yolov12.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
PLATFORM_SYSTEM="$(uname -s)"
PLATFORM_MACHINE="$(uname -m)"
SUPPORTED_PYTHON_VERSION="${SUPPORTED_PYTHON_VERSION:-3.9.18}"

# Pin only the packages required for the local YOLOv12 workflow.
CORE_PACKAGES=(
  "torch==2.2.2"
  "torchvision==0.17.2"
  "timm==1.0.14"
  "albumentations==2.0.4"
  "pycocotools==2.0.7"
  "PyYAML==6.0.1"
  "scipy==1.13.0"
  "opencv-python==4.9.0.80"
  "psutil==5.9.8"
  "py-cpuinfo==9.0.0"
  "huggingface-hub==0.23.2"
  "safetensors==0.4.3"
  "numpy==1.26.4"
  "supervision==0.22.0"
)

if [[ "${REPO_DIR}" != /* ]]; then
  # Resolve relative clone targets before the rest of the script uses the path.
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

# Prefer Python 3.9 when it is available, but allow a python3 fallback on OSC.
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.9 >/dev/null 2>&1; then
    PYTHON_BIN="python3.9"
  else
    PYTHON_BIN="python3"
    echo "python3.9 not found on PATH; falling back to python3."
  fi
fi

# Clone or reuse the upstream repository before rebuilding the environment.
# Create the parent directory before cloning into it.
mkdir -p "$(dirname "${REPO_DIR}")"

# Stop immediately if the requested interpreter is unavailable.
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found."
  echo "Set PYTHON_BIN to a valid Python ${SUPPORTED_PYTHON_VERSION} interpreter."
  exit 1
fi

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
  echo "Error: YOLOv12 OSC setup requires Python ${SUPPORTED_PYTHON_VERSION}, but selected ${SELECTED_PYTHON_VERSION}."
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

# Rebuild the repository-local virtual environment from the selected interpreter.
echo "Creating virtual environment at ${REPO_DIR}/.venv"
"${PYTHON_BIN}" -m venv --clear .venv

# Activate the environment before installing packages into it.
source .venv/bin/activate

# Update the packaging tools before installing project dependencies.
python -m pip install --upgrade pip setuptools wheel

# Install the pinned dependency set used by the Python entrypoint.
echo "Installing YOLOv12 core dependencies for ${PLATFORM_SYSTEM} ${PLATFORM_MACHINE}"
pip install "${CORE_PACKAGES[@]}"

# Install the local clone in editable mode.
pip install -e .

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "Skipping flash-attn install: the supported OSC baseline is Python ${SUPPORTED_PYTHON_VERSION}."
  echo "Re-enable flash-attn only after the repo's YOLOv12 CUDA stack is revalidated outside the enforced ${SUPPORTED_PYTHON_VERSION} baseline."
else
  echo "Skipping flash-attn install under the supported Python ${SUPPORTED_PYTHON_VERSION} OSC baseline."
fi

# Finish by printing the recommended OSC batch command for the prepared runtime.
# Print the next commands for the prepared environment.
cat <<EOF2
YOLOv12 setup complete.

Next steps:
  1) cd "${REPO_DIR}"
  2) source .venv/bin/activate
  3) cd "${SCRIPT_DIR}"
  4) bash "${REPO_ROOT}/osc_gpu_batch.sh" --account <OSC_ACCOUNT> --time 01:00:00 -- python3.9 demo_yolov12.py --repo-dir "${REPO_DIR}" --device 0
     # Add --device cpu when you want CPU inference explicitly.
EOF2
