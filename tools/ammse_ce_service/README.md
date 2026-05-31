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
cd /home/users/dky/openairinterface5g
```

1. Build the gNB, nrUE, and RFsim targets.

The wrapper defaults to `cmake_targets/ran_build_local/build/nr-softmodem`, so
this build layout works without extra environment variables:

```bash
cmake -S . -B cmake_targets/ran_build_local/build -GNinja
cmake --build cmake_targets/ran_build_local/build --target nr-softmodem nr-uesoftmodem rfsimulator
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

3. Configure the Python service environment.

```bash
export PYTHON_BIN=python3
export OAI_AMMSE_CE_CHECKPOINT=/home/users/dky/openairinterface5g/tools/ammse_ce_service/checkpoints/strujepa_best.pt

"${PYTHON_BIN}" -c 'import numpy, torch'
```

The last command is a quick dependency check for the Python service. The
service imports its own runtime code from `tools/ammse_ce_service`.

4. Start the gNB through the A-MMSE wrapper.

The wrapper starts `serve_elastic_ammse_ce.py`, waits for the Unix socket, exports
the socket/subnet variables, and then starts `nr-softmodem`.

```bash
sudo -E env \
  PYTHON_BIN="${PYTHON_BIN}" \
  OAI_AMMSE_CE_METHOD=strujepa \
  OAI_AMMSE_CE_CHECKPOINT="${OAI_AMMSE_CE_CHECKPOINT}" \
  OAI_AMMSE_CE_WIDTH=0.25 \
  OAI_AMMSE_CE_DEPTH=0.25 \
  OAI_AMMSE_CE_NOISE_POWER_DB=-70 \
  OAI_AMMSE_CE_METRICS=/tmp/oai_ammse_ce_metrics.csv \
  OAI_CE_NMSE_ENABLE=1 \
  OAI_CE_NMSE_CSV=/tmp/oai_ce_nmse.csv \
  tools/ammse_ce_service/run_oai_with_elastic_ammse.sh \
    --rfsim \
    --phy-test \
    --noS1 \
    -O ci-scripts/conf_files/gnb.band78.106prb.rfsim.phytest-strujepa.conf \
    '--rfsimulator.[0].serveraddr' server \
    --T_stdout 2 \
    --T_nowait
```

Use `sudo -E env ...` if your OAI run needs sudo; it preserves the variables the
wrapper and Python service need. The wrapper removes stale default `/tmp`
subnet, service-log, service-CSV, metrics-CSV, and NMSE-CSV files before
starting, because root may be unable to truncate user-owned files in sticky
directories such as `/tmp` on systems with `fs.protected_regular` enabled.

5. Start the nrUE in another terminal.

```bash
cd /home/users/dky/openairinterface5g

sudo -E cmake_targets/ran_build_local/build/nr-uesoftmodem \
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
printf 'width=0.5 depth=0.75\n' | sudo tee /tmp/oai_ammse_ce_subnet.txt
```

The gNB polls this file and sends the latest width/depth with each A-MMSE CE
request.

7. Check runtime outputs.

```bash
tail -f /tmp/oai_ammse_ce_service.log
tail -f /tmp/oai_ammse_ce_metrics.csv
tail -f /tmp/oai_ce_nmse.csv
```

Stop the gNB wrapper with `Ctrl-C`; its cleanup handler also stops the Python
A-MMSE service.

## Basic Run

```bash
OAI_AMMSE_CE_METHOD=strujepa \
OAI_AMMSE_CE_WIDTH=0.25 \
OAI_AMMSE_CE_DEPTH=0.25 \
tools/ammse_ce_service/run_oai_with_elastic_ammse.sh <nr-softmodem args>
```

The wrapper defaults to:

- `NR_SOFTMODEM=cmake_targets/ran_build_local/build/nr-softmodem`
- `OAI_AMMSE_CE_SOCKET=/tmp/oai_ammse_ce.sock`
- `OAI_AMMSE_CE_SUBNET_FILE=/tmp/oai_ammse_ce_subnet.txt`
- `OAI_AMMSE_CE_SERVICE_LOG=/tmp/oai_ammse_ce_service.log`
- `OAI_AMMSE_CE_SERVICE_CSV=/tmp/oai_ammse_ce_service.csv`

Use `OAI_AMMSE_CE_CHECKPOINT` to load a different checkpoint, or
`OAI_AMMSE_CE_RUN_DIR` with `OAI_AMMSE_CE_METHOD=static|strujepa|dynabert|ofa`
to load checkpoints from a local run directory copied into this OAI checkout.

## Runtime Subnet Selection

The gNB polls the subnet file and sends the current width/depth in each A-MMSE
CE request. Update the file while OAI is running:

```bash
printf 'width=0.5 depth=0.75\n' > /tmp/oai_ammse_ce_subnet.txt
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

Set `OAI_AMMSE_CE_METRICS` to write C-side request timing and status:

```bash
export OAI_AMMSE_CE_METRICS=/tmp/oai_ammse_ce_metrics.csv
```

The Python service writes request latency and subnet information to
`OAI_AMMSE_CE_SERVICE_CSV`.

For RFsim experiments, online CE NMSE logging can be enabled without replacing
the estimator:

```bash
export OAI_CE_NMSE_ENABLE=1
export OAI_CE_NMSE_CSV=/tmp/oai_ce_nmse.csv
```

The NMSE logger compares the applied channel estimate, the stock OAI
interpolated estimate, and raw DMRS-position LS estimates against the RFsim true
channel for the scheduled PUSCH grid.

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
