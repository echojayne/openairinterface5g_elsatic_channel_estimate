#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OAI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NR_SOFTMODEM="${NR_SOFTMODEM:-${OAI_ROOT}/cmake_targets/ran_build_local/build/nr-softmodem}"
OAI_BUILD_DIR="$(cd "$(dirname "${NR_SOFTMODEM}")" && pwd)"
SOCKET_PATH="${OAI_AMMSE_CE_SOCKET:-/tmp/oai_ammse_ce.sock}"
METHOD="${OAI_AMMSE_CE_METHOD:-strujepa}"
WIDTH="${OAI_AMMSE_CE_WIDTH:-1.0}"
DEPTH="${OAI_AMMSE_CE_DEPTH:-1.0}"
SUBNET_FILE="${OAI_AMMSE_CE_SUBNET_FILE:-/tmp/oai_ammse_ce_subnet.txt}"
SERVICE_LOG="${OAI_AMMSE_CE_SERVICE_LOG:-/tmp/oai_ammse_ce_service.log}"
SERVICE_CSV="${OAI_AMMSE_CE_SERVICE_CSV:-/tmp/oai_ammse_ce_service.csv}"
START_TIMEOUT_S="${OAI_AMMSE_CE_START_TIMEOUT_S:-120}"

mkdir -p "$(dirname "${SOCKET_PATH}")" "$(dirname "${SERVICE_LOG}")" "$(dirname "${SERVICE_CSV}")" "$(dirname "${SUBNET_FILE}")"

replace_regular_file() {
  local path="$1"
  if [[ -e "${path}" || -L "${path}" ]]; then
    if [[ ! -f "${path}" && ! -L "${path}" ]]; then
      echo "Refusing to replace non-regular file: ${path}" >&2
      exit 1
    fi
    rm -f -- "${path}" || {
      echo "Cannot replace ${path}; remove it or set a different OAI_AMMSE_CE_* path." >&2
      exit 1
    }
  fi
}

replace_regular_file "${SUBNET_FILE}"
replace_regular_file "${SERVICE_LOG}"
replace_regular_file "${SERVICE_CSV}"
if [[ -n "${OAI_AMMSE_CE_METRICS:-}" ]]; then
  mkdir -p "$(dirname "${OAI_AMMSE_CE_METRICS}")"
  replace_regular_file "${OAI_AMMSE_CE_METRICS}"
fi
if [[ -n "${OAI_CE_NMSE_CSV:-}" ]]; then
  mkdir -p "$(dirname "${OAI_CE_NMSE_CSV}")"
  replace_regular_file "${OAI_CE_NMSE_CSV}"
fi
printf 'width=%s depth=%s\n' "${WIDTH}" "${DEPTH}" > "${SUBNET_FILE}"

if ! "${PYTHON_BIN}" - <<'PY'
import numpy  # noqa: F401
import torch  # noqa: F401
PY
then
  echo "Python interpreter cannot import numpy and torch: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the absolute path of the Python environment used for the A-MMSE service." >&2
  exit 1
fi

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
if [[ -n "${OAI_AMMSE_CE_PRINT_EVERY:-}" ]]; then
  service_args+=(--print-every "${OAI_AMMSE_CE_PRINT_EVERY}")
fi

softmodem_extra_args=()

args_contain_exact() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

rfsim_channel_index="${OAI_AMMSE_CE_RFSIM_CHANNEL_INDEX:-1}"
rfsim_channel_model="${OAI_AMMSE_CE_RFSIM_CHANNEL_MODEL:-}"
rfsim_noise_power_db="${OAI_AMMSE_CE_RFSIM_NOISE_POWER_DB:-}"
rfsim_max_doppler_hz="${OAI_AMMSE_CE_RFSIM_MAX_DOPPLER_HZ:-}"
if [[ -z "${rfsim_max_doppler_hz}" && -n "${OAI_AMMSE_CE_RFSIM_SPEED_KMH:-}" ]]; then
  rfsim_carrier_hz="${OAI_AMMSE_CE_RFSIM_CARRIER_HZ:-3619200000}"
  rfsim_max_doppler_hz="$("${PYTHON_BIN}" - "${OAI_AMMSE_CE_RFSIM_SPEED_KMH}" "${rfsim_carrier_hz}" <<'PY'
import sys
speed_kmh = float(sys.argv[1])
carrier_hz = float(sys.argv[2])
print(f"{speed_kmh / 3.6 * carrier_hz / 299792458.0:.6f}")
PY
)"
fi

enable_rfsim_chanmod="${OAI_AMMSE_CE_ENABLE_RFSIM_CHANMOD:-}"
if [[ -z "${enable_rfsim_chanmod}" ]]; then
  if [[ -n "${rfsim_channel_model}" || -n "${rfsim_noise_power_db}" || -n "${rfsim_max_doppler_hz}" ]]; then
    enable_rfsim_chanmod=1
  else
    enable_rfsim_chanmod=0
  fi
fi
if [[ "${enable_rfsim_chanmod}" != "0" && "${enable_rfsim_chanmod,,}" != "false" && "${enable_rfsim_chanmod,,}" != "no" ]]; then
  if ! args_contain_exact "--rfsimulator.[0].options" "$@"; then
    softmodem_extra_args+=("--rfsimulator.[0].options" "chanmod")
  fi
fi
if [[ -n "${rfsim_channel_model}" ]]; then
  softmodem_extra_args+=("--channelmod.modellist_rfsimu_1.[${rfsim_channel_index}].type" "${rfsim_channel_model}")
fi
if [[ -n "${rfsim_noise_power_db}" ]]; then
  softmodem_extra_args+=("--channelmod.modellist_rfsimu_1.[${rfsim_channel_index}].noise_power_dB" "${rfsim_noise_power_db}")
fi
if [[ -n "${rfsim_max_doppler_hz}" ]]; then
  softmodem_extra_args+=("--channelmod.modellist_rfsimu_1.[${rfsim_channel_index}].max_Doppler" "${rfsim_max_doppler_hz}")
fi

print_service_log_tail() {
  if [[ -s "${SERVICE_LOG}" ]]; then
    echo "--- tail -80 ${SERVICE_LOG} ---" >&2
    tail -n 80 "${SERVICE_LOG}" >&2 || true
  fi
}

rm -f -- "${SOCKET_PATH}"
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
    print_service_log_tail
    exit 1
  fi
  sleep 0.05
done

if [[ ! -S "${SOCKET_PATH}" ]]; then
  echo "A-MMSE CE service socket was not created: ${SOCKET_PATH}. Log: ${SERVICE_LOG}" >&2
  print_service_log_tail
  exit 1
fi

export LD_LIBRARY_PATH="${OAI_BUILD_DIR}:${LD_LIBRARY_PATH:-}"
export OAI_AMMSE_CE_SOCKET="${SOCKET_PATH}"
export OAI_AMMSE_CE_WIDTH="${WIDTH}"
export OAI_AMMSE_CE_DEPTH="${DEPTH}"
export OAI_AMMSE_CE_SUBNET_FILE="${SUBNET_FILE}"

"${NR_SOFTMODEM}" "$@" "${softmodem_extra_args[@]}"
