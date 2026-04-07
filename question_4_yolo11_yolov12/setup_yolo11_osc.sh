#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/ultralytics"

REPO_URL="${REPO_URL:-https://github.com/ultralytics/ultralytics.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-}"
SUPPORTED_PYTHON_VERSION="${SUPPORTED_PYTHON_VERSION:-3.9.18}"

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

# Create the parent directory before cloning into it.
mkdir -p "$(dirname "${REPO_DIR}")"

# Stop immediately if the requested interpreter is unavailable.
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found. Set PYTHON_BIN to a valid Python ${SUPPORTED_PYTHON_VERSION} executable."
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
  echo "Error: YOLO11 OSC setup requires Python ${SUPPORTED_PYTHON_VERSION}, but selected ${SELECTED_PYTHON_VERSION}."
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

# Rebuild the repository-local virtual environment used by the Python entrypoint.
echo "Creating virtual environment at ${REPO_DIR}/.venv"
"${PYTHON_BIN}" -m venv --clear .venv

# Activate the environment before installing packages into it.
source .venv/bin/activate

# Update the packaging tools before installing the project itself.
python -m pip install --upgrade pip setuptools wheel

# Install the clone in editable mode so imports resolve to this checkout.
pip install -e .

# Print the next commands for the prepared environment.
cat <<EOF2
YOLO11 setup complete.

Next steps:
  1) cd "${REPO_DIR}"
  2) source .venv/bin/activate
  3) cd "${SCRIPT_DIR}"
  4) bash "${REPO_ROOT}/osc_gpu_batch.sh" --account <OSC_ACCOUNT> --time 01:00:00 -- python3.9 demo_yolo11.py --repo-dir "${REPO_DIR}" --device 0
     # Use --device cpu instead when you want CPU inference explicitly.
EOF2
