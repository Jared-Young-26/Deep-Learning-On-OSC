#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFLIGHT_SCRIPT="${SCRIPT_DIR}/osc_gpu_preflight.sh"

ACCOUNT=""
TIME_LIMIT=""
CLUSTER=""
NODES="1"
GPUS_PER_NODE="1"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  bash osc_gpu_interactive.sh --account <OSC_ACCOUNT> --time <HH:MM:SS> [options]

Options:
  --account <value>        Required OSC project/account.
  --time <HH:MM:SS>        Required Slurm walltime.
  --cluster <name>         Optional OSC cluster.
  --nodes <count>          Slurm node count. Default: 1.
  --gpus-per-node <count>  GPUs per node. Default: 1.
  --dry-run                Print the salloc/srun command instead of starting it.
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
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
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

SHELL_SNIPPET="source $(printf '%q' "${PREFLIGHT_SCRIPT}"); osc_prepare_gpu_environment; osc_require_gpu_allocation interactive-shell; exec /bin/bash -l"

ALLOC_COMMAND=(
  salloc
  "--account=${ACCOUNT}"
  "--time=${TIME_LIMIT}"
  "--nodes=${NODES}"
  "--gpus-per-node=${GPUS_PER_NODE}"
)

if [[ -n "${CLUSTER}" ]]; then
  ALLOC_COMMAND+=("--cluster=${CLUSTER}")
fi

ALLOC_COMMAND+=(srun --pty /bin/bash -lc "${SHELL_SNIPPET}")

if [[ "${DRY_RUN}" == "1" ]]; then
  quote_command "${ALLOC_COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${ALLOC_COMMAND[@]}"
