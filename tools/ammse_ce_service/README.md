# Elastic A-MMSE CE Service

This directory contains the OAI runtime hook and Python service used to replace
the default gNB uplink PUSCH channel estimate with an elastic A-MMSE estimate.
The gNB sends the sparse DMRS-position channel grid to the service over a Unix
domain socket. The service loads one full elastic checkpoint, selects the
requested width/depth subnet for each request, and returns a dense channel grid
for the scheduled PUSCH resources.

## Files

- `serve_elastic_ammse_ce.py`: Unix-socket inference service for the elastic
  A-MMSE checkpoint.
- `ammse_runtime/`: self-contained A-MMSE model, elastic Transformer wrapper,
  and input preprocessing code used by the service at inference time.
- `run_oai_with_elastic_ammse.sh`: wrapper that starts the service, exports the
  required OAI environment variables, then launches `nr-softmodem`.
- `configs/rayleigh8.env`: sourceable bash/zsh config for the default Rayleigh8
  RFsim evaluation run.
- `checkpoints/strujepa_best.pt`: optional local StruJEPA elastic A-MMSE
  checkpoint path. Large checkpoints are intentionally not committed to this
  public fork.

## Requirements

Build OAI normally with the gNB PHY code in this branch. The hook is disabled
unless `OAI_AMMSE_CE_SOCKET` is set, so a normal OAI run keeps the stock channel
estimator.

The Python service needs:

- Python 3
- `numpy`
- `torch`

No external StruJEPA checkout is required for deployment. The inference-time
model and elastic runtime code live in `tools/ammse_ce_service/ammse_runtime`.

## Checkpoint

The StruJEPA elastic A-MMSE checkpoint is hosted outside this public fork:

```text
https://drive.google.com/file/d/1ctv2xhcmAQQqUn-MK4KIZ_gcfd1cS1C2/view?usp=drive_link
```

Put a trained checkpoint at `tools/ammse_ce_service/checkpoints/strujepa_best.pt`
or pass one explicitly:

```bash
export OAI_AMMSE_CE_CHECKPOINT=/path/to/strujepa_best.pt
```

## Full Startup Flow

Run the following from the OAI repository root unless noted otherwise:

```bash
cd /path/to/openairinterface5g
```

1. Build the gNB, nrUE, and RFsim targets.

The wrapper defaults to `cmake_targets/ran_build_local/build/nr-softmodem`, so
this build layout works without extra environment variables:

```bash
cmake -S . -B cmake_targets/ran_build_local/build -GNinja
cmake --build cmake_targets/ran_build_local/build --target \
  nr-softmodem nr-uesoftmodem rfsimulator params_libconfig
```

If you build with OAI's standard `cmake_targets/build_oai` flow instead, set
`NR_SOFTMODEM` before starting the wrapper and use the matching
`nr-uesoftmodem` path in the UE command.

2. Download the checkpoint.

Download the Google Drive checkpoint above, then put it at the default service
path:

```bash
mkdir -p tools/ammse_ce_service/checkpoints
# Save the downloaded file as:
ls -lh tools/ammse_ce_service/checkpoints/strujepa_best.pt
```

Alternatively keep the checkpoint anywhere and pass its absolute path with
`OAI_AMMSE_CE_CHECKPOINT`.

3. Load the runtime config.

```bash
source tools/ammse_ce_service/configs/rayleigh8.env

"${PYTHON_BIN}" -c 'import numpy, torch'
```

The config file derives paths from the repository checkout, sets the active
Python interpreter path, points to the default checkpoint location, enables
Rayleigh8 RFsim channel modeling, and creates a timestamped run directory under
`tools/ammse_ce_service/runs/`. Source it from the active Python environment;
passing plain `python3` through `sudo` can select the system Python, which may
not have `numpy` or `torch`.

Override a setting after sourcing the file, for example:

```bash
export OAI_AMMSE_CE_RFSIM_SPEED_KMH=30
export OAI_AMMSE_CE_RFSIM_NOISE_POWER_DB=-80
export OAI_AMMSE_CE_NOISE_POWER_DB="${OAI_AMMSE_CE_RFSIM_NOISE_POWER_DB}"
```

4. Start the gNB through the A-MMSE wrapper.

The wrapper starts `serve_elastic_ammse_ce.py`, waits for the Unix socket, exports
the socket/subnet variables, and then starts `nr-softmodem`.

```bash
sudo -E tools/ammse_ce_service/run_oai_with_elastic_ammse.sh \
  --rfsim \
  --phy-test \
  --noS1 \
  -O ci-scripts/conf_files/gnb.band78.106prb.rfsim.phytest-strujepa.conf \
  '--rfsimulator.[0].serveraddr' server \
  --T_stdout 2 \
  --T_nowait
```

Use `sudo -E` if your OAI run needs sudo; it preserves the variables loaded from
`configs/rayleigh8.env`. The wrapper also accepts
`--ammse-config tools/ammse_ce_service/configs/rayleigh8.env` or
`OAI_AMMSE_CE_CONFIG_FILE=...` if you want it to source a config file directly.
By default this wrapper keeps all generated files under
`tools/ammse_ce_service/runs/<run_id>/` rather than `/tmp`, and mirrors the gNB
stdout/stderr to
`${OAI_AMMSE_CE_OUTPUT_DIR}/logs/nr-softmodem.log`.

5. Start the nrUE in another terminal.

```bash
cd /path/to/openairinterface5g
source tools/ammse_ce_service/configs/rayleigh8.env

sudo -E env \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${OAI_BUILD_DIR}/nr-uesoftmodem" \
  --rfsim \
  --phy-test \
  --noS1 \
  -O ci-scripts/conf_files/nrue.strujepa.rfsim.conf \
  -r 106 \
  --numerology 1 \
  --band 78 \
  -C 3619200000 \
  '--rfsimulator.[0].serveraddr' 127.0.0.1 \
  --T_stdout 2 \
  --T_nowait \
  --T_port 2023
```

The `--rfsimulator.[0].serveraddr` option is quoted because `zsh` treats
unquoted square brackets as filename glob syntax and fails with
`zsh: no matches found`.

If your PHY-test workflow uses pre-generated RRC files, add:

```bash
--reconfig-file /path/to/reconfig.raw --rbconfig-file /path/to/rbconfig.raw
```

6. Change the elastic subnet while OAI is running.

```bash
printf 'width=0.5 depth=0.75\n' | sudo tee "${OAI_AMMSE_CE_OUTPUT_DIR}/runtime/subnet.txt"
```

The gNB polls this file and sends the latest width/depth with each A-MMSE CE
request.

7. Check runtime outputs.

```bash
tail -f "${OAI_AMMSE_CE_OUTPUT_DIR}/logs/nr-softmodem.log"
tail -f "${OAI_AMMSE_CE_OUTPUT_DIR}/logs/service.log"
tail -f "${OAI_AMMSE_CE_OUTPUT_DIR}/metrics/c_timing.csv"
tail -f "${OAI_AMMSE_CE_OUTPUT_DIR}/metrics/ce_nmse.csv"
```

Stop the gNB wrapper with `Ctrl-C`; its cleanup handler also stops the Python
A-MMSE service.

## Basic Run

```bash
source tools/ammse_ce_service/configs/rayleigh8.env

OAI_AMMSE_CE_METHOD=strujepa \
OAI_AMMSE_CE_WIDTH=0.25 \
OAI_AMMSE_CE_DEPTH=0.25 \
tools/ammse_ce_service/run_oai_with_elastic_ammse.sh <nr-softmodem args>
```

The wrapper defaults to:

- `NR_SOFTMODEM=cmake_targets/ran_build_local/build/nr-softmodem`
- `OAI_AMMSE_CE_OUTPUT_ROOT=tools/ammse_ce_service/runs`
- `OAI_AMMSE_CE_OUTPUT_DIR=${OAI_AMMSE_CE_OUTPUT_ROOT}/<timestamp>_<pid>`
- `OAI_AMMSE_CE_SOCKET=${OAI_AMMSE_CE_OUTPUT_DIR}/runtime/sock`
- `OAI_AMMSE_CE_SUBNET_FILE=${OAI_AMMSE_CE_OUTPUT_DIR}/runtime/subnet.txt`
- `OAI_AMMSE_CE_SERVICE_LOG=${OAI_AMMSE_CE_OUTPUT_DIR}/logs/service.log`
- `OAI_AMMSE_CE_SOFTMODEM_LOG=${OAI_AMMSE_CE_OUTPUT_DIR}/logs/nr-softmodem.log`
- `OAI_AMMSE_CE_SERVICE_CSV=${OAI_AMMSE_CE_OUTPUT_DIR}/metrics/service.csv`
- `OAI_AMMSE_CE_METRICS=${OAI_AMMSE_CE_OUTPUT_DIR}/metrics/c_timing.csv`
- `OAI_CE_NMSE_CSV=${OAI_AMMSE_CE_OUTPUT_DIR}/metrics/ce_nmse.csv`

The run directory is organized by purpose:

- `logs/`: Python service log and mirrored `nr-softmodem` terminal output.
- `metrics/`: Python service timing CSV, C-side timing CSV, and CE NMSE CSV.
- `runtime/`: Unix socket and elastic subnet control file.
- `cache/`: per-run Python cache directories used by matplotlib/XDG clients.

Use `OAI_AMMSE_CE_CHECKPOINT` to load a different checkpoint, or
`OAI_AMMSE_CE_RUN_DIR` with `OAI_AMMSE_CE_METHOD=static|strujepa|dynabert|ofa`
to load checkpoints from a local run directory copied into this OAI checkout.

## Runtime Subnet Selection

The gNB polls the subnet file and sends the current width/depth in each A-MMSE
CE request. Update the file while OAI is running:

```bash
printf 'width=0.5 depth=0.75\n' > "${OAI_AMMSE_CE_OUTPUT_DIR}/runtime/subnet.txt"
```

The parser also accepts:

```text
0.5 0.75
```

The polling interval defaults to 100 ms and can be changed with
`OAI_AMMSE_CE_SUBNET_POLL_US`.

## Supported PUSCH Shape

The current OAI hook is scoped to the RFsim training/deployment shape used for
the bundled checkpoint:

- 1 layer
- 1 gNB RX antenna
- 50 RB
- PUSCH start symbol 0
- 13 PUSCH symbols

If a scheduled PUSCH does not match this shape, the hook logs one warning and
falls back to the stock OAI channel estimate.

## Metrics and NMSE Logging

The wrapper writes C-side request timing and status by default:

```bash
tail -f "${OAI_AMMSE_CE_OUTPUT_DIR}/metrics/c_timing.csv"
```

The metrics CSV writes one row for each A-MMSE request:

- `c_ce_total_us`: C-side wall time for the PUSCH channel-estimation block,
  including native DMRS estimation, optional A-MMSE replacement, and NMSE queue
  enqueue.
- `c_service_wall_us`: C-side wall time spent sending the request to the Python
  service and receiving the dense CE grid back.
- `python_total_us`: Python service time from after the request header is read
  through model execution and response packing.
- `python_model_us`: Python model inference time only.
- `ipc_gap_us`: `c_service_wall_us - python_total_us`. Treat this as the
  socket/process scheduling/serialization gap, not a pure IPC-only number.
- `status`: `0` means the A-MMSE replacement was applied. Negative values mean
  the hook attempted a request but the service response was not usable.

The Python service writes matching request latency and subnet information to
`OAI_AMMSE_CE_SERVICE_CSV`. Its `total_us` includes response sending; the
`python_total_us` column is the value returned to the C process.

To print the same C/Python timing fields in the gNB terminal while it is
running, set:

```bash
export OAI_AMMSE_CE_TIMING_PRINT_EVERY=100
```

Use `1` to print every A-MMSE request. Use `0` or leave it unset to disable live
timing prints. `OAI_AMMSE_CE_TIMING_PRINT=1` is a shortcut for printing every
request unless `OAI_AMMSE_CE_TIMING_PRINT_EVERY` is also set.

## Channel and Print Controls

The wrapper accepts environment variables for the common RFsim and logging
controls:

- `OAI_AMMSE_CE_PRINT_EVERY`: Python service progress print interval in served
  requests. Set `0` to disable periodic service prints.
- `OAI_AMMSE_CE_TIMING_PRINT_EVERY`: C-side A-MMSE timing log interval in
  requests. Set `1` to print every request, or `0` to disable.
- `OAI_CE_NMSE_PERIOD`: C-side NMSE log/CSV period in PUSCH events.
- `OAI_AMMSE_CE_ENABLE_RFSIM_CHANMOD`: set `1` to append
  `--rfsimulator.[0].options chanmod`.
- `OAI_AMMSE_CE_RFSIM_CHANNEL_MODEL`: RFsim uplink channel model, for example
  `Rayleigh8`.
- `OAI_AMMSE_CE_RFSIM_SPEED_KMH`: mobile speed. The wrapper converts it to
  `max_Doppler` using `OAI_AMMSE_CE_RFSIM_CARRIER_HZ`, default
  `3619200000`. Set it to `0` for static-channel validation or to a non-zero
  value, for example `30`, for mobility stress tests.
- `OAI_AMMSE_CE_RFSIM_MAX_DOPPLER_HZ`: explicit Doppler override. If set, it
  takes precedence over `OAI_AMMSE_CE_RFSIM_SPEED_KMH`.
- `OAI_AMMSE_CE_RFSIM_NOISE_POWER_DB`: RFsim channel `noise_power_dB` for
  `rfsimu_channel_ue0`. Keep `OAI_AMMSE_CE_NOISE_POWER_DB` aligned with this
  value so the A-MMSE model receives the same assumed noise level.
- `OAI_AMMSE_CE_RFSIM_CHANNEL_INDEX`: channelmod list index to override. The
  default `1` is `rfsimu_channel_ue0`, which is the uplink channel used by the
  gNB A-MMSE hook and NMSE logger.
- `OAI_AMMSE_CE_TEE_SOFTMODEM_LOG`: set `0` to disable mirroring
  `nr-softmodem` stdout/stderr to `logs/nr-softmodem.log`.

For RFsim experiments, online CE NMSE logging can be enabled without replacing
the estimator:

```bash
export OAI_CE_NMSE_ENABLE=1
tail -f "${OAI_AMMSE_CE_OUTPUT_DIR}/metrics/ce_nmse.csv"
```

The NMSE logger compares the applied channel estimate, the stock OAI
interpolated estimate, and raw DMRS-position LS estimates against the RFsim true
channel for the scheduled PUSCH grid. `ammse_all_re` and `oai_inter_all_re` use
the same all-RE replay convention as `rayleigh8_pareto_all_re_grid.png`:
`oai_inter_all_re` follows the T-tracer `raw-inter-ce-fd-data` layout, so REs
not emitted by OAI's extractor remain zero. `raw_dmrs` is the raw LS estimate on
the DMRS support only, while `raw_dmrs_all_re` is the sparse raw-DMRS all-RE
baseline used by the replay plots. The gNB command must enable
`'--rfsimulator.[0].options' chanmod`; otherwise RFsim does not allocate the
`rfsimu_channel_ue0`/`rfsimu_channel_enB0` descriptors and the logger reports
`CE NMSE logger has no RFsim true channel`. A-MMSE inference still runs in that
case, but strict true-channel NMSE is unavailable.

The NMSE CSV contains a `ref_source` column:

- `rfsim_ts`: strict RFsim reference. The gNB matched the PUSCH receive
  timestamp to a timestamped RFsim shared-memory channel snapshot and copied
  that snapshot into the NMSE sample at the channel-estimation entry point.
- `current`: fallback RFsim descriptor. This is only meaningful for static or
  effectively static channels. Time-varying RFsim strict NMSE should use
  `rfsim_ts` rows.

For time-varying RFsim channels, the logger no longer evaluates against the
later "current" descriptor. If no timestamped RFsim snapshot is available for
the PUSCH receive timestamp, or if the raw DMRS-position estimate is empty, that
sample is skipped. This keeps online NMSE aligned with the channel snapshot used
by the received PUSCH samples and avoids the misleading `0 dB` rows that appear
when the UE is not yet connected or has already exited.

## Useful RFsim Configs

The companion RFsim configs are:

- `ci-scripts/conf_files/gnb.band78.106prb.rfsim.phytest-strujepa.conf`
- `ci-scripts/conf_files/nrue.strujepa.rfsim.conf`
- `ci-scripts/conf_files/strujepa_channelmod_rfsimu.conf`

`strujepa_channelmod_rfsimu.conf` defines the RFsim channel model names used by
the true-channel trace and NMSE logger.

## Troubleshooting

- If model loading is slow, increase `OAI_AMMSE_CE_START_TIMEOUT_S`.
- If the gNB cannot connect to the service, check `OAI_AMMSE_CE_SERVICE_LOG` and
  confirm that the Unix socket path exists.
- If the checkpoint is missing, copy it to
  `tools/ammse_ce_service/checkpoints/strujepa_best.pt` or set
  `OAI_AMMSE_CE_CHECKPOINT`.
