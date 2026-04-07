#!/usr/bin/env bash

OSC_GPU_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSC_GPU_BATCH_LAUNCHER="${OSC_GPU_HELPER_DIR}/osc_gpu_batch.sh"
OSC_GPU_INTERACTIVE_LAUNCHER="${OSC_GPU_HELPER_DIR}/osc_gpu_interactive.sh"

osc_prepare_gpu_environment() {
  if ! command -v module >/dev/null 2>&1; then
    return 0
  fi
  module load cuda >/dev/null 2>&1 || true
}

osc_gpu_allocation_hint() {
  local target="${1:-the requested command}"
  cat >&2 <<EOF
Request an OSC GPU allocation before running ${target}.

Batch example:
  bash osc_gpu_batch.sh --account <OSC_ACCOUNT> --time 01:00:00 -- <command...>

Interactive example:
  bash osc_gpu_interactive.sh --account <OSC_ACCOUNT> --time 01:00:00

These launchers live at:
  ${OSC_GPU_BATCH_LAUNCHER}
  ${OSC_GPU_INTERACTIVE_LAUNCHER}
EOF
}

osc_slurm_gpu_signals_present() {
  local signal="${SLURM_GPUS_ON_NODE:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
  [[ -n "${signal}" && "${signal}" != "NoDevFiles" ]]
}

osc_visible_gpu_present() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

osc_require_gpu_allocation() {
  local target="${1:-the requested command}"

  osc_prepare_gpu_environment

  if [[ -z "${SLURM_JOB_ID:-}" && -z "${SLURM_STEP_ID:-}" ]]; then
    echo "Error: no Slurm allocation is active for ${target}." >&2
    osc_gpu_allocation_hint "${target}"
    return 1
  fi

  if ! osc_slurm_gpu_signals_present; then
    echo "Error: this Slurm job does not expose any GPU resources for ${target}." >&2
    echo "Submit the job with --gpus-per-node=1 (or a higher count when needed)." >&2
    osc_gpu_allocation_hint "${target}"
    return 1
  fi

  if ! osc_visible_gpu_present; then
    echo "Error: nvidia-smi does not report a visible GPU for ${target}." >&2
    echo "The CUDA module was loaded, but the current allocation still looks CPU-only." >&2
    osc_gpu_allocation_hint "${target}"
    return 1
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  osc_require_gpu_allocation "${1:-the requested command}"
fi
