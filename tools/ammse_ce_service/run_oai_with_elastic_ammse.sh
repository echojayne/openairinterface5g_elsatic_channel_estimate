#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OAI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NR_SOFTMODEM="${NR_SOFTMODEM:-${OAI_ROOT}/cmake_targets/ran_build_local/build/nr-softmodem}"
SOCKET_PATH="${OAI_AMMSE_CE_SOCKET:-/tmp/oai_ammse_ce.sock}"
METHOD="${OAI_AMMSE_CE_METHOD:-strujepa}"
WIDTH="${OAI_AMMSE_CE_WIDTH:-1.0}"
DEPTH="${OAI_AMMSE_CE_DEPTH:-1.0}"
SUBNET_FILE="${OAI_AMMSE_CE_SUBNET_FILE:-/tmp/oai_ammse_ce_subnet.txt}"
SERVICE_LOG="${OAI_AMMSE_CE_SERVICE_LOG:-/tmp/oai_ammse_ce_service.log}"
SERVICE_CSV="${OAI_AMMSE_CE_SERVICE_CSV:-/tmp/oai_ammse_ce_service.csv}"
START_TIMEOUT_S="${OAI_AMMSE_CE_START_TIMEOUT_S:-120}"

mkdir -p "$(dirname "${SOCKET_PATH}")" "$(dirname "${SERVICE_LOG}")" "$(dirname "${SERVICE_CSV}")" "$(dirname "${SUBNET_FILE}")"
printf 'width=%s depth=%s\n' "${WIDTH}" "${DEPTH}" > "${SUBNET_FILE}"

service_args=(
  "${SCRIPT_DIR}/serve_elastic_ammse_ce.py"
  --method "${METHOD}"
  --socket "${SOCKET_PATH}"
  --width "${WIDTH}"
  --depth "${DEPTH}"
  --log-csv "${SERVICE_CSV}"
)

if [[ -n "${OAI_AMMSE_CE_CHECKPOINT:-}" ]]; then
  service_args+=(--checkpoint "${OAI_AMMSE_CE_CHECKPOINT}")
fi
if [[ -n "${OAI_AMMSE_CE_RUN_DIR:-}" ]]; then
  service_args+=(--run-dir "${OAI_AMMSE_CE_RUN_DIR}")
fi
if [[ -n "${OAI_AMMSE_CE_DEVICE:-}" ]]; then
  service_args+=(--device "${OAI_AMMSE_CE_DEVICE}")
fi
if [[ -n "${OAI_AMMSE_CE_NOISE_POWER_DB:-}" ]]; then
  service_args+=(--noise-power-db "${OAI_AMMSE_CE_NOISE_POWER_DB}")
fi

rm -f "${SOCKET_PATH}"
"${PYTHON_BIN}" "${service_args[@]}" > "${SERVICE_LOG}" 2>&1 &
service_pid=$!

cleanup() {
  kill "${service_pid}" >/dev/null 2>&1 || true
  wait "${service_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 $((START_TIMEOUT_S * 20))); do
  if [[ -S "${SOCKET_PATH}" ]]; then
    break
  fi
  if ! kill -0 "${service_pid}" >/dev/null 2>&1; then
    echo "A-MMSE CE service exited before socket became ready. Log: ${SERVICE_LOG}" >&2
    exit 1
  fi
  sleep 0.05
done

if [[ ! -S "${SOCKET_PATH}" ]]; then
  echo "A-MMSE CE service socket was not created: ${SOCKET_PATH}. Log: ${SERVICE_LOG}" >&2
  exit 1
fi

export OAI_AMMSE_CE_SOCKET="${SOCKET_PATH}"
export OAI_AMMSE_CE_WIDTH="${WIDTH}"
export OAI_AMMSE_CE_DEPTH="${DEPTH}"
export OAI_AMMSE_CE_SUBNET_FILE="${SUBNET_FILE}"

"${NR_SOFTMODEM}" "$@"
