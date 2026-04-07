#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT_SCRIPT="${SCRIPT_DIR}/osc_gpu_preflight.sh"

ACCOUNT=""
TIME_LIMIT=""
CLUSTER=""
NODES="1"
GPUS_PER_NODE="1"
JOB_NAME="osc-gpu-job"
WORKDIR="$(pwd)"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time <HH:MM:SS> [options] -- <command...>

Options:
  --account <value>        Required OSC project/account.
  --time <HH:MM:SS>        Required Slurm walltime.
  --cluster <name>         Optional OSC cluster.
  --nodes <count>          Slurm node count. Default: 1.
  --gpus-per-node <count>  GPUs per node. Default: 1.
  --job-name <name>        Optional Slurm job name.
  --workdir <path>         Directory to cd into before running the command.
  --dry-run                Print the generated sbatch script instead of submitting it.
EOF
}

quote_command() {
  local quoted=()
  local part
  for part in "$@"; do
    quoted+=("$(printf '%q' "${part}")")
  done
  printf '%s' "${quoted[*]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      ACCOUNT="${2:-}"
      shift 2
      ;;
    --time)
      TIME_LIMIT="${2:-}"
      shift 2
      ;;
    --cluster)
      CLUSTER="${2:-}"
      shift 2
      ;;
    --nodes)
      NODES="${2:-}"
      shift 2
      ;;
    --gpus-per-node)
      GPUS_PER_NODE="${2:-}"
      shift 2
      ;;
    --job-name)
      JOB_NAME="${2:-}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Error: unrecognized option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${ACCOUNT}" || -z "${TIME_LIMIT}" ]]; then
  echo "Error: --account and --time are required." >&2
  usage >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Error: supply the command to run after --." >&2
  usage >&2
  exit 1
fi

COMMAND=("$@")
WORKDIR="$(cd "${WORKDIR}" && pwd)"
COMMAND_STRING="$(quote_command "${COMMAND[@]}")"
TMP_SCRIPT="$(mktemp "/tmp/osc-gpu-batch.XXXXXX")"
trap 'rm -f "${TMP_SCRIPT}"' EXIT

cat >"${TMP_SCRIPT}" <<EOF
#!/usr/bin/env bash
#SBATCH --account=${ACCOUNT}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --job-name=${JOB_NAME}
EOF

if [[ -n "${CLUSTER}" ]]; then
  printf '#SBATCH --cluster=%s\n' "${CLUSTER}" >>"${TMP_SCRIPT}"
fi

cat >>"${TMP_SCRIPT}" <<EOF

set -euo pipefail

cd $(printf '%q' "${WORKDIR}")

if command -v module >/dev/null 2>&1; then
  module load cuda || true
fi

source $(printf '%q' "${PREFLIGHT_SCRIPT}")
osc_require_gpu_allocation "${COMMAND[0]}"

exec ${COMMAND_STRING}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${TMP_SCRIPT}"
  exit 0
fi

sbatch "${TMP_SCRIPT}"
