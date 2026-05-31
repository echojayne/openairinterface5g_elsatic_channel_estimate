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
- the StruJEPA training/evaluation source tree

Set `STRUJEPA_ROOT` if the StruJEPA tree is not at `/home/users/dky/StruJEPA`:

```bash
export STRUJEPA_ROOT=/path/to/StruJEPA
```

Put a trained checkpoint at `tools/ammse_ce_service/checkpoints/strujepa_best.pt`
or pass one explicitly:

```bash
export OAI_AMMSE_CE_CHECKPOINT=/path/to/strujepa_best.pt
```

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
`OAI_AMMSE_CE_RUN_DIR` with `OAI_AMMSE_CE_METHOD=static|strujepa|dynabert|ofa|matformer`
to load checkpoints from a StruJEPA run directory.

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
