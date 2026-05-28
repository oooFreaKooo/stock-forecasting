#!/usr/bin/env bash
# Source before running radar CLI on macOS (LightGBM requires libomp)
# Usage: source scripts/env.sh   (from repo root, or via absolute path)

_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
RADAR_ROOT="$(cd "${_ENV_SH_DIR}/.." && pwd)"
export RADAR_ROOT

if [[ "$(uname)" == "Darwin" ]] && [[ -d "/opt/homebrew/opt/libomp/lib" ]]; then
  export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH:-}"
fi

# Avoid OpenMP/PyTorch thread fights that hang or crash Chronos after LightGBM inference.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -d "${RADAR_ROOT}/.venv" ]]; then
  # shellcheck source=/dev/null
  source "${RADAR_ROOT}/.venv/bin/activate"
fi

if [[ -f "${RADAR_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${RADAR_ROOT}/.env"
  set +a
fi
