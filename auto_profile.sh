#!/usr/bin/env bash

set -euo pipefail

DEFAULT_SCRIPT="inference_baseline.py"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <n (1-50)> [python_file (default: $DEFAULT_SCRIPT)]" >&2
  exit 1
fi

MAX_N=$1
MAX_EE_HEAD=13
# PY_SCRIPT=${2:-$DEFAULT_SCRIPT}


if ! [[ "$MAX_N" =~ ^[0-9]+$ ]] || (( MAX_N < 1 || MAX_N > 50 )); then
  echo "n must be an integer between 1 and 50." >&2
  exit 1
fi

# if ! [[ "$MAX_EE_HEAD" =~ ^[0-9]+$ ]] || (( MAX_EE_HEAD < 0 || MAX_EE_HEAD > 13 )); then
#   echo "ee_head must be an integer between 0 and 13." >&2
#   exit 1
# fi

activate_default_env() {
  # Try conda first if available.
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base:-}" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_base/etc/profile.d/conda.sh"
      conda activate default
      return
    fi
  fi

  # If conda isn't present, try mamba via its profile scripts (no shell hook needed).
  if command -v mamba >/dev/null 2>&1; then
    local mamba_base
    mamba_base="$(mamba info --base 2>/dev/null || true)"
    if [[ -n "${mamba_base:-}" && -f "$mamba_base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$mamba_base/etc/profile.d/conda.sh"
      conda activate default
      return
    fi
    if [[ -n "${mamba_base:-}" && -f "$mamba_base/etc/profile.d/mamba.sh" ]]; then
      # shellcheck disable=SC1090
      source "$mamba_base/etc/profile.d/mamba.sh"
      mamba activate default
      return
    fi
  fi

  echo "Could not activate the 'default' env. Run 'mamba init' (or 'conda init bash'), open a new shell, then rerun this script." >&2
  exit 1
}

activate_default_env

for ((i=1; i<=MAX_N; i++)); do
for ((EE_HEAD=0; EE_HEAD<=MAX_EE_HEAD; EE_HEAD++)); do
  echo "Running $DEFAULT_SCRIPT with -b $i ($i/$MAX_N) -e $EE_HEAD ..."
  python "$DEFAULT_SCRIPT" -b "$i" -e "$EE_HEAD" -m resnet50ee
done
done