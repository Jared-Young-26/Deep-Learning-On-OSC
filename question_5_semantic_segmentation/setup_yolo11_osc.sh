#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/ultralytics"

REPO_URL="${REPO_URL:-https://github.com/ultralytics/ultralytics.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${REPO_DIR}" != /* ]]; then
  # Accept relative clone targets from either the repo root or the question folder.
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

mkdir -p "$(dirname "${REPO_DIR}")"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found. Set PYTHON_BIN to a valid Python executable."
  exit 1
fi

# Clone once into external/ so this question folder only keeps the project-facing
# wrapper scripts, datasets, and outputs.
if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "Using existing clone at ${REPO_DIR}"
else
  echo "Cloning ${REPO_URL} into ${REPO_DIR}"
  git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Reuse the environment when possible because Question 5 may be run repeatedly
# across bootstrap, training, and forward-only demo steps.
if [[ -d "${REPO_DIR}/.venv" ]]; then
  REUSED_VENV=1
  echo "Reusing existing virtual environment at ${REPO_DIR}/.venv"
else
  REUSED_VENV=0
  echo "Creating virtual environment at ${REPO_DIR}/.venv"
  "${PYTHON_BIN}" -m venv .venv
fi

source .venv/bin/activate

# Keep packaging tools in a range known to work with this editable Ultralytics setup.
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

# OBB utilities depend on a recent Shapely, so validate or upgrade it explicitly.
# Remove any older ultralytics wheel first so the editable install below is the
# version Python actually imports during bootstrap/train/demo.
pip uninstall -y ultralytics >/dev/null 2>&1 || true
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

# Install the local source tree itself, then prove the import resolves from this venv.
if [[ "${REUSED_VENV}" == "1" ]]; then
  pip install --disable-pip-version-check --no-build-isolation --no-deps --force-reinstall .
else
  pip install --disable-pip-version-check --no-build-isolation --force-reinstall . "shapely>=2.0.0"
fi

echo "Verifying Ultralytics import..."
python -c "import ultralytics; print(ultralytics.__file__)"

# End with the exact three-step workflow the rest of Question 5 expects.
cat <<EOF2
YOLO11 OBB setup complete for Question 5.

You can run this setup from either directory:
  Repo root:
    bash question_5_semantic_segmentation/setup_yolo11_osc.sh
  Inside question_5_semantic_segmentation:
    bash setup_yolo11_osc.sh

Next steps:
  1) Bootstrap DOTAv1 and the reusable pretrained checkpoint:
     ${REPO_DIR}/.venv/bin/python "${SCRIPT_DIR}/bootstrap_dota_obb.py" --repo-dir "${REPO_DIR}"

  2) Fine-tune on DOTAv1 and save question_5_semantic_segmentation/models/dota_obb/best.pt:
     ${REPO_DIR}/.venv/bin/python "${SCRIPT_DIR}/train_dota_obb.py" --repo-dir "${REPO_DIR}"

  3) Run the forward-only demo on your satellite images:
     ${REPO_DIR}/.venv/bin/python "${SCRIPT_DIR}/demo_yolo_segmentation.py" --repo-dir "${REPO_DIR}" --device cpu

Repo-local demo input folder:
  ${SCRIPT_DIR}/inputs/satellite_images

Repo-local demo output folder:
  ${SCRIPT_DIR}/outputs/satellite_results
EOF2
