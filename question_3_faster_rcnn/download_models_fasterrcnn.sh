#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="${1:-${REPO_ROOT}/external/FasterRCNN}"

if [[ "${REPO_DIR}" != /* ]]; then
  # Resolve relative repo paths before the rest of the script uses them.
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

# Stop early if setup has not created the upstream clone yet.
if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Error: FasterRCNN repo not found at ${REPO_DIR}"
  echo "Run bash setup_fasterrcnn_osc.sh first."
  exit 1
fi

# Try the available download tools in order.
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

# Change into the upstream repository before writing the checkpoint files.
cd "${REPO_DIR}"

# Download the supported base and detector weights into the repository root.
download_file "http://trzy.org/files/fasterrcnn/vgg16_caffe.pth" "vgg16_caffe.pth"
download_file "http://trzy.org/files/fasterrcnn/fasterrcnn_pytorch_vgg16.pth" "fasterrcnn_pytorch_vgg16.pth"
download_file "http://trzy.org/files/fasterrcnn/fasterrcnn_tf2.h5" "fasterrcnn_tf2.h5"
download_file "http://trzy.org/files/fasterrcnn/fasterrcnn_pytorch_resnet50.pth" "fasterrcnn_pytorch_resnet50.pth"

echo "Model download complete."
echo "Files saved in ${REPO_DIR}"
