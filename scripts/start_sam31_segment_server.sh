#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

: "${SAM31_SEGMENT_BACKEND_HOST:=127.0.0.1}"
: "${SAM31_SEGMENT_BACKEND_PORT:=8766}"
: "${SAM3_DEVICE:=cuda}"
: "${SAM31_PRELOAD_ON_START:=true}"

if [[ -z "${SAM3_CHECKPOINT_PATH:-}" ]]; then
  echo "ERROR: SAM3_CHECKPOINT_PATH must point to a local SAM3.1 checkpoint." >&2
  echo "Example:" >&2
  echo "  SAM3_CHECKPOINT_PATH=/path/to/sam3.1/checkpoint.pt $0" >&2
  exit 2
fi

export SAM3_CHECKPOINT_PATH
export SAM3_DEVICE
export SAM31_PRELOAD_ON_START
if [[ -n "${SAM3_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export SAM3_CUDA_VISIBLE_DEVICES
fi

exec ./.venv/bin/python -m vision_backends.sam31_segment_server \
  --host "${SAM31_SEGMENT_BACKEND_HOST}" \
  --port "${SAM31_SEGMENT_BACKEND_PORT}"
