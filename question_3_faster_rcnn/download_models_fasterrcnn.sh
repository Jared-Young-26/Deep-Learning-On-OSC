#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="${1:-${REPO_ROOT}/external/FasterRCNN}"

if [[ "${REPO_DIR}" != /* ]]; then
  # Allow users to pass a relative repo location from either the repo root or the
  # question folder without changing the download logic below.
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Error: FasterRCNN repo not found at ${REPO_DIR}"
  echo "Run bash setup_fasterrcnn_osc.sh first."
  exit 1
fi

# Prefer the first downloader available so this helper works on more machines.
download_file() {
  local url="$1"
  local output_path="$2"

  if command -v wget >/dev/null 2>&1; then
    wget -O "${output_path}" "${url}"
    return
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output_path}"
    return
  fi

  echo "Error: neither wget nor curl is available for downloading model files."
  exit 1
}

# The model files live inside the upstream repo because that is where the wrapper
# and the original Faster R-CNN entrypoints both expect to find them.
cd "${REPO_DIR}"

# Download both frameworks' supported checkpoints so the demo can switch between
# local TF2 usage and CUDA-backed PyTorch usage without another setup pass.
# - vgg16_caffe.pth is the base VGG16 backbone initialization.
# - fasterrcnn_pytorch_vgg16.pth and fasterrcnn_pytorch_resnet50.pth are PyTorch detectors.
# - fasterrcnn_tf2.h5 is the TF2 detector used by the local compatibility wrapper.
download_file "http://trzy.org/files/fasterrcnn/vgg16_caffe.pth" "vgg16_caffe.pth"
download_file "http://trzy.org/files/fasterrcnn/fasterrcnn_pytorch_vgg16.pth" "fasterrcnn_pytorch_vgg16.pth"
download_file "http://trzy.org/files/fasterrcnn/fasterrcnn_tf2.h5" "fasterrcnn_tf2.h5"
download_file "http://trzy.org/files/fasterrcnn/fasterrcnn_pytorch_resnet50.pth" "fasterrcnn_pytorch_resnet50.pth"

echo "Model download complete."
echo "Files saved in ${REPO_DIR}"
