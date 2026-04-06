#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/Transfer-Learning-Library"

REPO_URL="${REPO_URL:-https://github.com/thuml/Transfer-Learning-Library.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DETECTRON2_PIP_SPEC="${DETECTRON2_PIP_SPEC:-git+https://github.com/facebookresearch/detectron2.git}"
DETECTRON2_BUILD_NINJA="${DETECTRON2_BUILD_NINJA:-1}"

# Allow a relative target path at invocation time, then normalize it before the
# rest of the script reuses the resolved repository location.
if [[ "${REPO_DIR}" != /* ]]; then
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

mkdir -p "$(dirname "${REPO_DIR}")"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found."
  echo "Set PYTHON_BIN to a valid Python executable."
  exit 1
fi

# Clone once into external/ so this directory keeps only local scripts, data, and results.
if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "Using existing clone at ${REPO_DIR}"
else
  echo "Cloning ${REPO_URL} into ${REPO_DIR}"
  git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Build the repository-local virtual environment that the helper scripts will reuse.
echo "Creating virtual environment at ${REPO_DIR}/.venv"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

# Install the baseline TLlib requirements first, then add only the extra packages
# required by this object-detection workflow.
python -m pip install --upgrade pip "setuptools<82" wheel

# Torch is optional here because many environments already provide a working
# installation, and the correct wheel depends on CUDA and platform details.
if [[ "${INSTALL_TORCH}" == "1" ]]; then
  pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
fi

if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
fi

# Install TLlib from the local source clone.
pip install -e .

# Install the minimal extra packages required by `source_only.py`.
# The full object_detection requirements also pull in mmcv, which is not needed
# for this workflow and is heavier to build on many systems.
pip install timm

# Detectron2 is the critical native dependency for the downstream Python driver,
# but it stays optional here because the correct build path varies by host.
if [[ "${INSTALL_DETECTRON2}" == "1" ]]; then
  if [[ "${DETECTRON2_BUILD_NINJA}" == "1" ]]; then
    pip install ninja
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    # Apple toolchains often need explicit SDK and include flags for Detectron2's
    # native extensions, so apply them only on macOS.
    SDKROOT="$(xcrun --show-sdk-path)"
    CPLUS_INCLUDE_PATH="${SDKROOT}/usr/include/c++/v1" \
    CPPFLAGS="-isysroot ${SDKROOT} -I${SDKROOT}/usr/include/c++/v1" \
    CXXFLAGS="-isysroot ${SDKROOT} -I${SDKROOT}/usr/include/c++/v1" \
    LDFLAGS="-isysroot ${SDKROOT}" \
    CC=clang CXX=clang++ \
      pip install --no-build-isolation "${DETECTRON2_PIP_SPEC}"
  else
    pip install --no-build-isolation "${DETECTRON2_PIP_SPEC}"
  fi
fi

# Report the next steps plus whether Detectron2 is ready, since that is the most
# common blocker for the downstream Python driver.
if python -c "import detectron2" >/dev/null 2>&1; then
  DETECTRON2_STATUS="installed"
else
  DETECTRON2_STATUS="missing"
fi

cat <<EOF
TLlib setup complete.

Next steps:
  1) cd "${REPO_DIR}"
  2) source .venv/bin/activate
  3) cd "${SCRIPT_DIR}"
  4) python3 demo_tllib_object_detection.py --repo-dir "${REPO_DIR}" --mode doctor
  5) bash run_tllib_osc.sh

Optional:
  INSTALL_TORCH=1 bash setup_tllib_osc.sh
  INSTALL_DETECTRON2=1 bash setup_tllib_osc.sh

Object detection notes:
  - Installed for source_only.py: torch/tllib base deps + timm
  - Full-pipeline runs also need detectron2 because TLlib's object-detection
    scripts are Detectron2-based under the hood
  - detectron2 status: ${DETECTRON2_STATUS}
  - If detectron2 is still missing, the demo wrapper will stop with a preflight
    error before training starts. Install detectron2 manually for your Python /
    Torch / CUDA platform, or rerun with INSTALL_DETECTRON2=1 if that matches
    your environment.
EOF
