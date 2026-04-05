#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/ultralytics"

REPO_URL="${REPO_URL:-https://github.com/ultralytics/ultralytics.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${REPO_DIR}" != /* ]]; then
  # Allow a relative destination path, but normalize it before clone/setup continues.
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

# Create the parent directory once so the clone target is valid even on a fresh repo.
mkdir -p "$(dirname "${REPO_DIR}")"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found. Set PYTHON_BIN to a valid Python executable."
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

# Build the repo-local environment that the question 4 wrapper will call directly.
# This keeps the assignment wrapper isolated from whatever global Python packages
# may already exist on the machine.
echo "Creating virtual environment at ${REPO_DIR}/.venv"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Install the cloned repository in editable mode so imports resolve from this
# checkout while still allowing local source updates.
pip install -e .

# Finish with the exact next steps for the local demo.
cat <<EOF2
YOLO11 setup complete.

Next steps:
  1) cd "${REPO_DIR}"
  2) source .venv/bin/activate
  3) cd "${SCRIPT_DIR}"
  4) python3 demo_yolo11.py --repo-dir "${REPO_DIR}"
EOF2
