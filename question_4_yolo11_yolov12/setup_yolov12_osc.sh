#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/yolov12"

REPO_URL="${REPO_URL:-https://github.com/sunsmarterjie/yolov12.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
PLATFORM_SYSTEM="$(uname -s)"
PLATFORM_MACHINE="$(uname -m)"

# These pins are the smallest package set that supports the local Question 4
# YOLOv12 demo reliably without mirroring the repo's much broader upstream extras.
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
  # Allow a relative destination path, but normalize it before clone/setup continues.
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

# Create the parent directory once so the clone target is valid even on a fresh repo.
mkdir -p "$(dirname "${REPO_DIR}")"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found. Set PYTHON_BIN to a valid Python 3.11 executable."
  exit 1
fi

# Clone once into external/ so the question directory only keeps the wrapper/demo code.
if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "Using existing clone at ${REPO_DIR}"
else
  echo "Cloning ${REPO_URL} into ${REPO_DIR}"
  git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Build the repo-local environment first, then layer only the packages this
# assignment demo actually needs from the much larger upstream stack.
# The pinned core packages are the subset this repo needs for local inference
# without pulling in every optional export, demo, or CUDA-only extra upstream.
echo "Creating virtual environment at ${REPO_DIR}/.venv"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

echo "Installing Question 4 YOLOv12 core dependencies for ${PLATFORM_SYSTEM} ${PLATFORM_MACHINE}"
pip install "${CORE_PACKAGES[@]}"

# Install package from source clone while skipping upstream-only extras such as
# the Linux flash-attn wheel, ONNX export packages, and the Gradio demo app.
pip install -e .

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  # flash-attn is optional and only worth attempting on the specific Linux CUDA
  # environment it was built for.
  if [[ "${PLATFORM_SYSTEM}" == "Linux" && "${PLATFORM_MACHINE}" == "x86_64" ]]; then
    pip install flash-attn --no-build-isolation
  else
    echo "Skipping flash-attn install: supported only on Linux x86_64 CUDA environments."
  fi
else
  echo "Skipping flash-attn install (set INSTALL_FLASH_ATTN=1 on supported Linux x86_64 CUDA nodes)."
fi

# Finish with the exact next steps for the local demo.
cat <<EOF2
YOLOv12 setup complete.

Next steps:
  1) cd "${REPO_DIR}"
  2) source .venv/bin/activate
  3) cd "${SCRIPT_DIR}"
  4) python3 demo_yolov12.py --repo-dir "${REPO_DIR}"
     # Add --device cpu on macOS or when you want CPU inference explicitly.
EOF2
