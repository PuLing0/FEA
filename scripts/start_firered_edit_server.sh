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

: "${FIRERED_CUDA_VISIBLE_DEVICES:=5,6,3,4}"
: "${FIRERED_NUM_INFERENCE_STEPS:=8}"
: "${FIRERED_FUSE_LORA:=true}"
: "${FIRERED_PRELOAD_ON_START:=true}"
: "${FIRERED_EDIT_BACKEND_HOST:=127.0.0.1}"
: "${FIRERED_EDIT_BACKEND_PORT:=8765}"

export FIRERED_CUDA_VISIBLE_DEVICES
export FIRERED_NUM_INFERENCE_STEPS
export FIRERED_FUSE_LORA
export FIRERED_PRELOAD_ON_START

exec ./.venv/bin/python -m vision_backends.firered_edit_server \
  --host "${FIRERED_EDIT_BACKEND_HOST}" \
  --port "${FIRERED_EDIT_BACKEND_PORT}"
