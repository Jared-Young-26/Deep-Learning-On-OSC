#!/usr/bin/env bash
set -euo pipefail

# Prepare the repo-local Ultralytics environment used by the Q5 segmentation
# wrappers. Unlike Q4, this setup may also bootstrap the dataset and pretrained
# checkpoint because the tracked Q5 CLI expects those artifacts to exist.

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

# Clone or reuse the upstream repository before rebuilding the environment.
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

SELECTED_PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"
echo "Using Python interpreter: ${PYTHON_BIN} (${SELECTED_PYTHON_VERSION})"

if [[ "${SELECTED_PYTHON_VERSION}" != "${SUPPORTED_PYTHON_VERSION}" ]]; then
  echo "Error: YOLO11 segmentation OSC setup requires Python ${SUPPORTED_PYTHON_VERSION}, but selected ${SELECTED_PYTHON_VERSION}."
  echo "Load Python ${SUPPORTED_PYTHON_VERSION} on OSC or set PYTHON_BIN to a Python ${SUPPORTED_PYTHON_VERSION} executable."
  exit 1
fi

# Reuse the virtual environment when it already exists.
if [[ -d "${REPO_DIR}/.venv" ]]; then
  REUSED_VENV=1
  if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
    EXISTING_VENV_VERSION="$("${REPO_DIR}/.venv/bin/python" -c 'import sys; print(sys.version.split()[0])')"
    if [[ "${EXISTING_VENV_VERSION}" != "${SELECTED_PYTHON_VERSION}" ]]; then
      echo "Existing virtual environment uses Python ${EXISTING_VENV_VERSION}, but setup selected ${SELECTED_PYTHON_VERSION}."
      echo "Remove ${REPO_DIR}/.venv and rerun, or set PYTHON_BIN to match the existing environment."
      exit 1
    fi
  fi
  echo "Reusing existing virtual environment at ${REPO_DIR}/.venv"
else
  REUSED_VENV=0
  echo "Creating virtual environment at ${REPO_DIR}/.venv"
  "${PYTHON_BIN}" -m venv .venv
fi

# Activate the environment before installing or checking packages.
source .venv/bin/activate

# Keep the packaging bootstrap tools below versions known to work with this
# OSC-ready Ultralytics segmentation path.
# Verify the packaging tool versions before reinstalling them.
if python - <<'PY'
from importlib.metadata import PackageNotFoundError, version


def parse(value: str) -> tuple[int, ...]:
    parts = []
    for chunk in value.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


try:
    pip_ok = parse(version("pip")) < (27,)
    setuptools_ok = parse(version("setuptools")) < (82,)
    wheel_ok = True if version("wheel") else False
except PackageNotFoundError:
    raise SystemExit(1)

raise SystemExit(0 if pip_ok and setuptools_ok and wheel_ok else 1)
PY
then
  echo "Packaging bootstrap tools already compatible."
else
  python -m pip install --disable-pip-version-check --upgrade "pip<27" "setuptools<82" wheel
fi

# Remove stale site-packages copies so imports resolve to this checkout rather
# than to any previously installed Ultralytics wheel.
# Remove stale Ultralytics package metadata before reinstalling the local clone.
python - <<'PY'
import shutil
import sysconfig
from pathlib import Path

purelib = Path(sysconfig.get_paths()["purelib"])
for path in sorted(purelib.glob("ultralytics*")):
    if path.name == "ultralytics" and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.name.startswith("ultralytics-") and (path.suffix == ".dist-info" or path.suffix == ".egg-info"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
PY
# Check that Shapely is new enough for the segmentation utilities.
if python - <<'PY'
from importlib.metadata import PackageNotFoundError, version


def parse(value: str) -> tuple[int, ...]:
    parts = []
    for chunk in value.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


try:
    shapely_ok = parse(version("shapely")) >= (2, 0, 0)
except PackageNotFoundError:
    shapely_ok = False

raise SystemExit(0 if shapely_ok else 1)
PY
then
  echo "Shapely already compatible."
else
  pip install --disable-pip-version-check --upgrade "shapely>=2.0.0"
fi

# Keep the Q5 setup self-contained by ensuring the dataset downloader is present
# before the optional auto-bootstrap step runs.
# Check that gdown is available before running the dataset bootstrap.
if python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    version("gdown")
except PackageNotFoundError:
    raise SystemExit(1)

raise SystemExit(0)
PY
then
  echo "gdown already installed."
else
  pip install --disable-pip-version-check --upgrade "gdown>=5,<6"
fi

# Reinstall the local source tree into the active environment.
if [[ "${REUSED_VENV}" == "1" ]]; then
  pip install --disable-pip-version-check --no-build-isolation --no-deps --ignore-installed .
else
  pip install --disable-pip-version-check --no-build-isolation --ignore-installed . "shapely>=2.0.0"
fi

# Confirm that the active environment imports the local Ultralytics package.
echo "Verifying Ultralytics import..."
python -c "import ultralytics; print(ultralytics.__file__)"

AUTO_BOOTSTRAP_DATASET="${AUTO_BOOTSTRAP_DATASET:-1}"
if [[ "${AUTO_BOOTSTRAP_DATASET}" == "1" ]]; then
  # Run dataset preparation immediately when automatic bootstrap is enabled.
  echo "Auto-bootstrapping the Q5 dataset and pretrained checkpoint..."
  python "${SCRIPT_DIR}/bootstrap_isaid_seg.py" --repo-dir "${REPO_DIR}" --download-dataset
else
  echo "Skipping automatic Q5 dataset bootstrap because AUTO_BOOTSTRAP_DATASET=${AUTO_BOOTSTRAP_DATASET}."
fi

# Finish by printing the shortest follow-up commands for manual bootstrap,
# training, and inference.
# Print the next commands for the prepared environment.
cat <<EOF2
YOLO11 segmentation setup complete.

You can run this setup from either directory:
  Repo root:
    bash question_5_semantic_segmentation/setup_yolo11_osc.sh
  Inside question_5_semantic_segmentation:
    bash setup_yolo11_osc.sh

Next steps:
  1) If you need to rerun dataset preparation manually:
     ${REPO_DIR}/.venv/bin/python "${SCRIPT_DIR}/bootstrap_isaid_seg.py" --repo-dir "${REPO_DIR}" --download-dataset

  2) Fine-tune on iSAID and save question_5_semantic_segmentation/models/isaid_seg/best.pt:
     bash "${REPO_ROOT}/osc_gpu_batch.sh" --account <OSC_ACCOUNT> --time 04:00:00 -- ${REPO_DIR}/.venv/bin/python "${SCRIPT_DIR}/train_isaid_seg.py" --repo-dir "${REPO_DIR}" --device 0

  3) Run the forward-only segmentation demo on your satellite images:
     ${REPO_DIR}/.venv/bin/python "${SCRIPT_DIR}/demo_yolo_segmentation.py" --repo-dir "${REPO_DIR}" --device cpu

Repo-local demo input folder:
  ${SCRIPT_DIR}/inputs/satellite_images

Repo-local demo output folder:
  ${SCRIPT_DIR}/outputs/satellite_results

Set AUTO_BOOTSTRAP_DATASET=0 if you want setup without the dataset download step.
EOF2
