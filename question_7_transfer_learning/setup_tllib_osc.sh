#!/usr/bin/env bash
set -euo pipefail

# Prepare the repo-local TLlib checkout used by the tracked Q7 wrapper. This
# setup script also owns the small compatibility repairs that keep TLlib usable
# on the OSC-validated dependency stack.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_REPO_DIR="${REPO_ROOT}/external/Transfer-Learning-Library"

REPO_URL="${REPO_URL:-https://github.com/thuml/Transfer-Learning-Library.git}"
REPO_DIR="${1:-${DEFAULT_REPO_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-0}"
REPAIR_ONLY="${REPAIR_ONLY:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DETECTRON2_PIP_SPEC="${DETECTRON2_PIP_SPEC:-git+https://github.com/facebookresearch/detectron2.git}"
DETECTRON2_BUILD_NINJA="${DETECTRON2_BUILD_NINJA:-1}"
DETECTRON2_CC="${DETECTRON2_CC:-}"
DETECTRON2_CXX="${DETECTRON2_CXX:-}"
SUPPORTED_PYTHON_VERSION="${SUPPORTED_PYTHON_VERSION:-3.9.18}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.9 >/dev/null 2>&1; then
    PYTHON_BIN="python3.9"
  else
    PYTHON_BIN="python3"
  fi
fi

compiler_banner() {
  # Print one short compiler banner for diagnostics without failing setup.
  local compiler="${1:-}"
  if [[ -z "${compiler}" ]]; then
    return 0
  fi
  "${compiler}" --version 2>/dev/null | head -n 1 || true
}

compiler_looks_nvhpc() {
  # Detect NVHPC-style compiler wrappers because Detectron2 builds more
  # reliably on OSC when GCC is selected explicitly instead.
  local banner
  banner="$(compiler_banner "${1:-}")"
  [[ "${banner}" == *"NVIDIA"* || "${banner}" == *"NVHPC"* || "${banner}" == *"PGI"* || "${banner}" == *"nvc++"* || "${banner}" == *"nvc "* ]]
}

apply_tllib_torchvision_compat() {
  # Apply the local source edits that keep the checked-out TLlib revision
  # compatible with the repo's validated torchvision / detectron2 stack.
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import re

repo_dir = Path.cwd()

resnet_path = repo_dir / "tllib" / "vision" / "models" / "resnet.py"
resnet_text = resnet_path.read_text()
expected_import = "from torchvision.models.resnet import BasicBlock, Bottleneck"
stale_import = "from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls"
new_import = expected_import + "\n"
old_model_urls = """WEIGHTS_ENUMS = {\n    'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,\n    'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,\n    'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,\n    'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,\n    'resnet152': models.ResNet152_Weights.IMAGENET1K_V1,\n    'resnext50_32x4d': models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,\n    'resnext101_32x8d': models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1,\n    'wide_resnet50_2': models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,\n    'wide_resnet101_2': models.Wide_ResNet101_2_Weights.IMAGENET1K_V1,\n}\n"""
new_model_urls = """WEIGHTS_ENUMS = {\n    'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,\n    'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,\n    'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,\n    'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,\n    'resnet152': models.ResNet152_Weights.IMAGENET1K_V1,\n    'resnext50_32x4d': models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1,\n    'resnext101_32x8d': models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1,\n    'wide_resnet50_2': models.Wide_ResNet50_2_Weights.IMAGENET1K_V1,\n    'wide_resnet101_2': models.Wide_ResNet101_2_Weights.IMAGENET1K_V1,\n}\n\n# torchvision removed `model_urls`, but several TLlib modules still import it.\n# Recreate the same mapping from the modern weights enums so older TLlib code\n# can keep calling `load_state_dict_from_url(model_urls[arch])`.\nmodel_urls = {arch: weights.url for arch, weights in WEIGHTS_ENUMS.items()}\n"""
old_pretrained = """        if model_urls is not None:\n            pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)\n        else:\n            pretrained_dict = WEIGHTS_ENUMS[arch].get_state_dict(progress=progress)\n"""
new_pretrained = """        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)\n"""
model_urls_line = "model_urls = {arch: weights.url for arch, weights in WEIGHTS_ENUMS.items()}"
all_block_pattern = r"__all__ = \[.*?\]\n"

resnet_text = re.sub(
    r"try:\n\s+from torchvision\.models\.resnet import .*model_urls.*\nexcept ImportError:\n\s+from torchvision\.models\.resnet import .*BasicBlock.*Bottleneck.*\n\s+model_urls = None\n",
    new_import,
    resnet_text,
)
resnet_text = re.sub(
    r"^from torchvision\.models\.resnet import .*model_urls.*$",
    expected_import,
    resnet_text,
    flags=re.MULTILINE,
)

if old_model_urls in resnet_text and model_urls_line not in resnet_text:
    resnet_text = resnet_text.replace(old_model_urls, new_model_urls)
elif model_urls_line not in resnet_text:
    weights_match = re.search(r"WEIGHTS_ENUMS = \{\n(?:    .+\n)+\}\n", resnet_text)
    if weights_match is not None:
        resnet_text = (
            resnet_text[: weights_match.end()]
            + "\n"
            + "# torchvision removed `model_urls`, but several TLlib modules still import it.\n"
            + "# Recreate the same mapping from the modern weights enums so older TLlib code\n"
            + "# can keep calling `load_state_dict_from_url(model_urls[arch])`.\n"
            + model_urls_line
            + "\n"
            + resnet_text[weights_match.end():]
        )
    else:
        all_match = re.search(all_block_pattern, resnet_text, flags=re.DOTALL)
        if all_match is None:
            raise SystemExit(
                "Error: TLlib torchvision compatibility verification failed.\n"
                f"File: {resnet_path}\n"
                "  - could not locate an insertion point for the replacement model_urls mapping\n"
                "First 20 lines:\n"
                + "\n".join(f"{idx + 1:>4}: {line}" for idx, line in enumerate(resnet_text.splitlines()[:20]))
            )
        resnet_text = (
            resnet_text[: all_match.end()]
            + "\n"
            + new_model_urls
            + resnet_text[all_match.end():]
        )
if old_pretrained in resnet_text:
    resnet_text = resnet_text.replace(old_pretrained, new_pretrained)

verification_errors = []
if expected_import not in resnet_text:
    verification_errors.append(f"missing expected import: {expected_import}")
if stale_import in resnet_text:
    verification_errors.append(f"stale import still present: {stale_import}")
if model_urls_line not in resnet_text:
    verification_errors.append(f"missing local model_urls mapping: {model_urls_line}")
if verification_errors:
    raise SystemExit(
        "Error: TLlib torchvision compatibility verification failed.\n"
        f"File: {resnet_path}\n"
        + "\n".join(f"  - {error}" for error in verification_errors)
        + "\nFirst 20 lines:\n"
        + "\n".join(f"{idx + 1:>4}: {line}" for idx, line in enumerate(resnet_text.splitlines()[:20]))
    )

resnet_path.write_text(resnet_text)

deeplab_path = repo_dir / "tllib" / "vision" / "models" / "segmentation" / "deeplabv2.py"
deeplab_text = deeplab_path.read_text()
deeplab_text = deeplab_text.replace(
    "from torchvision.models.utils import load_state_dict_from_url\n",
    "from torch.hub import load_state_dict_from_url\n",
)
deeplab_path.write_text(deeplab_text)

backbone_init_path = repo_dir / "tllib" / "vision" / "models" / "object_detection" / "backbone" / "__init__.py"
backbone_init_text = backbone_init_path.read_text()
old_backbone_import = "from .vgg import VGG, build_vgg_fpn_backbone\n"
new_backbone_import = """try:\n    from .vgg import VGG, build_vgg_fpn_backbone\nexcept ModuleNotFoundError as exc:\n    if exc.name != \"mmcv\":\n        raise\n\n    VGG = None\n    build_vgg_fpn_backbone = None\n"""
if old_backbone_import in backbone_init_text and "ModuleNotFoundError" not in backbone_init_text:
    backbone_init_text = new_backbone_import
backbone_init_path.write_text(backbone_init_text)

imagelist_path = repo_dir / "tllib" / "vision" / "datasets" / "imagelist.py"
imagelist_text = imagelist_path.read_text()
imagelist_text = imagelist_text.replace(
    "from torch.utils.data.dataset import Dataset, T_co, IterableDataset\n",
    "from torch.utils.data.dataset import Dataset, IterableDataset\n",
)
if "Dataset[T_co]" in imagelist_text and "TypeVar" not in imagelist_text:
    imagelist_text = imagelist_text.replace(
        "from typing import Optional, Callable, Tuple, Any, List, Iterable\n",
        "from typing import Optional, Callable, Tuple, Any, List, Iterable, TypeVar\n",
    )
if "Dataset[T_co]" in imagelist_text and "T_co = TypeVar('T_co', covariant=True)" not in imagelist_text:
    imagelist_text = imagelist_text.replace(
        "from torchvision.datasets.folder import default_loader\n",
        "from torchvision.datasets.folder import default_loader\n\nT_co = TypeVar('T_co', covariant=True)\n",
    )
imagelist_path.write_text(imagelist_text)

object_detection_dataset_path = repo_dir / "tllib" / "vision" / "datasets" / "object_detection" / "__init__.py"
object_detection_dataset_text = object_detection_dataset_path.read_text()
object_detection_dataset_text = object_detection_dataset_text.replace(
    "import xml.etree.ElementTree as ET\n\nfrom detectron2.data import (\n",
    "import xml.etree.ElementTree as ET\n\nfrom PIL import Image\nfrom detectron2.data import (\n",
)
object_detection_dataset_text = object_detection_dataset_text.replace(
    "fileids = np.loadtxt(f, dtype=np.str)\n",
    "fileids = np.loadtxt(f, dtype=str)\n",
)
if "def _resolve_image_size(image_path, xml_height, xml_width):\n" not in object_detection_dataset_text:
    object_detection_dataset_text = object_detection_dataset_text.replace(
        "def parse_root_and_file_name(path):\n"
        "    path_list = path.split('/')\n"
        "    dataset_root = '/'.join(path_list[:-1])\n"
        "    file_name = path_list[-1]\n"
        "    if dataset_root == '':\n"
        "        dataset_root = '.'\n"
        "    return dataset_root, file_name\n",
        "def parse_root_and_file_name(path):\n"
        "    path_list = path.split('/')\n"
        "    dataset_root = '/'.join(path_list[:-1])\n"
        "    file_name = path_list[-1]\n"
        "    if dataset_root == '':\n"
        "        dataset_root = '.'\n"
        "    return dataset_root, file_name\n\n\n"
        "def _resolve_image_size(image_path, xml_height, xml_width):\n"
        "    try:\n"
        "        with PathManager.open(image_path, \"rb\") as image_file:\n"
        "            with Image.open(image_file) as image:\n"
        "                image_width, image_height = image.size\n"
        "    except Exception:\n"
        "        return xml_height, xml_width, False\n\n"
        "    corrected = image_height != xml_height or image_width != xml_width\n"
        "    return image_height, image_width, corrected\n",
    )
object_detection_dataset_text = object_detection_dataset_text.replace(
    "    dicts = []\n"
    "    skip_classes = set()\n",
    "    dicts = []\n"
    "    skip_classes = set()\n"
    "    corrected_size_count = 0\n"
    "    corrected_size_examples = []\n",
)
object_detection_dataset_text = object_detection_dataset_text.replace(
    "        r = {\n"
    "            \"file_name\": jpeg_file,\n"
    "            \"image_id\": fileid,\n"
    "            \"height\": int(tree.findall(\"./size/height\")[0].text),\n"
    "            \"width\": int(tree.findall(\"./size/width\")[0].text),\n"
    "        }\n",
    "        xml_height = int(tree.findall(\"./size/height\")[0].text)\n"
    "        xml_width = int(tree.findall(\"./size/width\")[0].text)\n"
    "        image_height, image_width, corrected_size = _resolve_image_size(\n"
    "            jpeg_file, xml_height, xml_width\n"
    "        )\n"
    "        if corrected_size:\n"
    "            corrected_size_count += 1\n"
    "            if len(corrected_size_examples) < 5:\n"
    "                corrected_size_examples.append(\n"
    "                    f\"{fileid}: xml {xml_height}x{xml_width} -> jpeg {image_height}x{image_width}\"\n"
    "                )\n\n"
    "        r = {\n"
    "            \"file_name\": jpeg_file,\n"
    "            \"image_id\": fileid,\n"
    "            \"height\": image_height,\n"
    "            \"width\": image_width,\n"
    "        }\n",
)
if "Corrected image-size metadata for " not in object_detection_dataset_text:
    object_detection_dataset_text = object_detection_dataset_text.replace(
        "    print(\"Skip classes:\", list(skip_classes))\n"
        "    return dicts\n",
        "    print(\"Skip classes:\", list(skip_classes))\n"
        "    if corrected_size_count:\n"
        "        print(\n"
        "            \"Corrected image-size metadata for \"\n"
        "            f\"{corrected_size_count} VOC annotations using JPEG headers. \"\n"
        "            f\"Examples: {', '.join(corrected_size_examples)}\"\n"
        "        )\n"
        "    return dicts\n",
    )
object_detection_dataset_path.write_text(object_detection_dataset_text)

dadapt_proposal_path = repo_dir / "tllib" / "alignment" / "d_adapt" / "proposal.py"
dadapt_proposal_text = dadapt_proposal_path.read_text()
dadapt_proposal_text = dadapt_proposal_text.replace(".astype(np.float)", ".astype(float)")
dadapt_proposal_path.write_text(dadapt_proposal_text)

print(f"Applied TLlib torchvision compatibility patches and verified {resnet_path}.")
PY
}

# Allow a relative target path at invocation time, then normalize it before the
# rest of the script reuses the resolved repository location.
if [[ "${REPO_DIR}" != /* ]]; then
  REPO_DIR="${PWD}/${REPO_DIR}"
fi

mkdir -p "$(dirname "${REPO_DIR}")"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found."
  echo "Set PYTHON_BIN to a valid Python ${SUPPORTED_PYTHON_VERSION} executable."
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
apply_tllib_torchvision_compat

REPO_PYTHON="${REPO_DIR}/.venv/bin/python"

if [[ "${REPAIR_ONLY}" == "1" ]]; then
  echo "Repair-only mode enabled. Skipping virtualenv rebuild and package installation."
  if [[ -x "${REPO_PYTHON}" ]]; then
    EXISTING_VENV_VERSION="$("${REPO_PYTHON}" -c 'import sys; print(sys.version.split()[0])')"
    echo "Existing repo-local Python: ${REPO_PYTHON} (${EXISTING_VENV_VERSION})"
    if [[ "${EXISTING_VENV_VERSION}" != "${SUPPORTED_PYTHON_VERSION}" ]]; then
      echo "Note: repo-local Python does not match the supported OSC version ${SUPPORTED_PYTHON_VERSION}."
    fi
    if "${REPO_PYTHON}" -c "import detectron2" >/dev/null 2>&1; then
      DETECTRON2_STATUS="installed"
    else
      DETECTRON2_STATUS="missing"
    fi
  else
    echo "Note: repo-local Python entrypoint is missing at ${REPO_PYTHON}."
    DETECTRON2_STATUS="missing"
  fi

  cat <<EOF
TLlib repair-only preflight complete.

Repair summary:
  - patched TLlib torchvision compatibility shims
  - repo-local python: $(if [[ -x "${REPO_PYTHON}" ]]; then printf '%s' "${REPO_PYTHON}"; else printf '<missing>'; fi)
  - detectron2 status: ${DETECTRON2_STATUS}

If the repo-local Python is missing, rerun full setup:
  INSTALL_TORCH=1 INSTALL_DETECTRON2=1 bash "${SCRIPT_DIR}/setup_tllib_osc.sh" "${REPO_DIR}"
EOF
  exit 0
fi

# Capture the selected interpreter version before reusing or rebuilding the environment.
SELECTED_PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(sys.version.split()[0])')"
echo "Using Python interpreter: ${PYTHON_BIN} (${SELECTED_PYTHON_VERSION})"

if [[ "${SELECTED_PYTHON_VERSION}" != "${SUPPORTED_PYTHON_VERSION}" ]]; then
  echo "Error: TLlib OSC setup requires Python ${SUPPORTED_PYTHON_VERSION}, but selected ${SELECTED_PYTHON_VERSION}."
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

# Build the repository-local virtual environment that the helper scripts will reuse.
echo "Creating virtual environment at ${REPO_DIR}/.venv"
"${PYTHON_BIN}" -m venv --clear .venv
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
    # Detectron2 expects a GNU C/C++ toolchain on Linux. OSC shells can expose
    # NVHPC compilers by default, which cause the noisy warning pattern and
    # eventual native-extension build failure seen with detectron2.
    LINUX_CC="${DETECTRON2_CC:-${CC:-}}"
    LINUX_CXX="${DETECTRON2_CXX:-${CXX:-}}"

    if [[ -z "${LINUX_CC}" ]] && command -v gcc >/dev/null 2>&1; then
      LINUX_CC="$(command -v gcc)"
    fi
    if [[ -z "${LINUX_CXX}" ]] && command -v g++ >/dev/null 2>&1; then
      LINUX_CXX="$(command -v g++)"
    fi

    if compiler_looks_nvhpc "${LINUX_CC}" || compiler_looks_nvhpc "${LINUX_CXX}"; then
      if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
        LINUX_CC="$(command -v gcc)"
        LINUX_CXX="$(command -v g++)"
      fi
    fi

    if [[ -z "${LINUX_CC}" || -z "${LINUX_CXX}" ]]; then
      echo "Error: Detectron2 build on Linux/OSC requires gcc and g++."
      echo "Set DETECTRON2_CC and DETECTRON2_CXX explicitly, or load a GNU toolchain and rerun."
      exit 1
    fi

    if compiler_looks_nvhpc "${LINUX_CC}" || compiler_looks_nvhpc "${LINUX_CXX}"; then
      echo "Error: Detectron2 build resolved to an NVHPC/PGI compiler, which is unsupported here."
      echo "Resolved compiler banners:"
      echo "  CC : $(compiler_banner "${LINUX_CC}")"
      echo "  CXX: $(compiler_banner "${LINUX_CXX}")"
      echo "Load a GNU toolchain first, or rerun with:"
      echo "  DETECTRON2_CC=/path/to/gcc DETECTRON2_CXX=/path/to/g++ bash question_7_transfer_learning/setup_tllib_osc.sh"
      exit 1
    fi

    echo "Building detectron2 with CC=${LINUX_CC}"
    echo "Building detectron2 with CXX=${LINUX_CXX}"
    CC="${LINUX_CC}" CXX="${LINUX_CXX}" \
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
  4) python3.9 demo_tllib_object_detection.py --repo-dir "${REPO_DIR}" --mode doctor
  5) bash "${REPO_ROOT}/osc_gpu_batch.sh" --account <OSC_ACCOUNT> --time 04:00:00 -- bash run_tllib_osc.sh

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
