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

FIRERED_LOG_DIR="${FIRERED_LOG_DIR:-generated/service_logs}"
SAM31_LOG_DIR="${SAM31_LOG_DIR:-generated/service_logs}"
mkdir -p "${FIRERED_LOG_DIR}" "${SAM31_LOG_DIR}"

FIRERED_LOG_PATH="${FIRERED_LOG_PATH:-${FIRERED_LOG_DIR}/firered_edit_server.log}"
SAM31_LOG_PATH="${SAM31_LOG_PATH:-${SAM31_LOG_DIR}/sam31_segment_server.log}"

cleanup() {
  local exit_code=$?
  if [[ -n "${FIRERED_PID:-}" ]] && kill -0 "${FIRERED_PID}" 2>/dev/null; then
    kill "${FIRERED_PID}" 2>/dev/null || true
    wait "${FIRERED_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SAM31_PID:-}" ]] && kill -0 "${SAM31_PID}" 2>/dev/null; then
    kill "${SAM31_PID}" 2>/dev/null || true
    wait "${SAM31_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}

trap cleanup INT TERM EXIT

./scripts/start_firered_edit_server.sh >"${FIRERED_LOG_PATH}" 2>&1 &
FIRERED_PID=$!

./scripts/start_sam31_segment_server.sh >"${SAM31_LOG_PATH}" 2>&1 &
SAM31_PID=$!

echo "FireRed edit service PID: ${FIRERED_PID}"
echo "FireRed edit log: ${FIRERED_LOG_PATH}"
echo "SAM3.1 segment service PID: ${SAM31_PID}"
echo "SAM3.1 segment log: ${SAM31_LOG_PATH}"
echo "Press Ctrl+C to stop both services."

wait -n "${FIRERED_PID}" "${SAM31_PID}"
