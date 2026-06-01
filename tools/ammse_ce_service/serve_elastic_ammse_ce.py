#!/usr/bin/env python3
"""Serve an elastic A-MMSE CE checkpoint to OAI through a Unix-domain socket.

The OAI request header carries width/depth as fixed-point PPM fields. The
service loads one full elastic checkpoint and selects the requested subnet per
PUSCH request.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any

SERVICE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = SERVICE_DIR / "runs" / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_DIR / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from torch import nn

from ammse_runtime.elastic import STACK_PATHS, MultiStackTorchEncoderWrapper, call_model_for_spec
from ammse_runtime.model import AMMSERankAdaptiveConfig, AMMSERankAdaptiveModel
from ammse_runtime.utils import EPS, complex_vector_to_channels, resolve_device


MAGIC = 0x414D4D53
VERSION = 1
REQUEST = struct.Struct("<16I")
RESPONSE = struct.Struct("<8I")
DYNAMIC_CHECKPOINT_REL = {
    "strujepa": "strujepa/checkpoints/strujepa_best.pt",
    "dynabert": "dynabert/checkpoints/dynabert_best.pt",
    "ofa": "ofa/checkpoints/ofa_best.pt",
}
DEFAULT_RUN_DIR = SERVICE_DIR / "runs" / "oai_elastic_ammse_rfsim_ce"
DEFAULT_STRUJEPA_CHECKPOINT = SERVICE_DIR / "checkpoints" / "strujepa_best.pt"


def recv_exact(conn: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise EOFError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_all(conn: socket.socket, data: bytes | memoryview) -> None:
    conn.sendall(data)


def static_checkpoint_from_summary(run_dir: Path) -> Path:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    return Path(summary["methods"]["static"]["checkpoint"])


def checkpoint_path_for(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    if args.method == "strujepa" and DEFAULT_STRUJEPA_CHECKPOINT.exists():
        return DEFAULT_STRUJEPA_CHECKPOINT.resolve()
    if args.method == "static":
        return static_checkpoint_from_summary(args.run_dir).expanduser().resolve()
    return (args.run_dir / DYNAMIC_CHECKPOINT_REL[args.method]).expanduser().resolve()


def load_model(method: str, checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any], np.ndarray, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint["model_config"])
    base = AMMSERankAdaptiveModel(AMMSERankAdaptiveConfig(**model_config))
    if method in {"strujepa", "dynabert", "ofa"}:
        model: nn.Module = MultiStackTorchEncoderWrapper(base, stack_paths=STACK_PATHS)
    elif method == "static":
        model = base
    else:
        raise ValueError(f"unsupported method: {method}")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return (
        model,
        model_config,
        np.asarray(checkpoint["pilot_mask"], dtype=np.bool_),
        float(checkpoint["noise_reference_db"]),
    )


def parse_request_header(data: bytes) -> dict[str, int]:
    values = REQUEST.unpack(data)
    keys = (
        "magic",
        "version",
        "header_bytes",
        "request_id",
        "frame",
        "slot",
        "rb_size",
        "nr_symbols",
        "start_symbol",
        "nb_re",
        "grid_elems",
        "nr_layers",
        "nb_rx_ant",
        "nvar",
        "width_ppm",
        "depth_ppm",
    )
    return dict(zip(keys, values, strict=True))


def c16_payload_to_grid(payload: bytes, *, nb_re: int, nr_symbols: int) -> np.ndarray:
    values = np.frombuffer(payload, dtype="<i2")
    expected = int(nb_re) * int(nr_symbols) * 2
    if values.size != expected:
        raise ValueError(f"expected {expected} int16 payload values, got {values.size}")
    pairs = values.reshape(-1, 2)
    complex_vec = pairs[:, 0].astype(np.float32) + 1j * pairs[:, 1].astype(np.float32)
    return complex_vec.reshape((int(nb_re), int(nr_symbols)), order="F")


def grid_to_c16_payload(grid: np.ndarray) -> bytes:
    vector = np.asarray(grid).reshape(-1, order="F")
    interleaved = np.empty((vector.size, 2), dtype="<i2")
    interleaved[:, 0] = np.clip(np.rint(vector.real), -32768, 32767).astype("<i2")
    interleaved[:, 1] = np.clip(np.rint(vector.imag), -32768, 32767).astype("<i2")
    return interleaved.tobytes(order="C")


def predict_grid(
    *,
    method: str,
    model: nn.Module,
    model_config: dict[str, Any],
    pilot_mask: np.ndarray,
    noise_reference_db: float,
    noise_power_db: float,
    width: float,
    depth: float,
    raw_grid: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    flat_mask = pilot_mask.reshape(-1, order="F")
    pilot_vector = raw_grid.reshape(-1, order="F")[flat_mask]
    scale = float(np.sqrt(np.mean(np.abs(pilot_vector) ** 2) + EPS))
    pilot_channels = complex_vector_to_channels(pilot_vector / scale)
    pilot_tensor = torch.from_numpy(pilot_channels[None]).to(device=device, dtype=torch.float32)
    noise_var = torch.tensor(
        [10.0 ** ((float(noise_power_db) - float(noise_reference_db)) / 10.0)],
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        pred_norm = call_model_for_spec(
            method=method,
            model=model,
            pilot_vector=pilot_tensor,
            noise_var=noise_var,
            width=float(width),
            depth=float(depth),
            model_config=model_config,
        )
    pred = pred_norm.detach().cpu().numpy()[0]
    return (pred[0] + 1j * pred[1]) * scale


def serve(args: argparse.Namespace) -> int:
    args.run_dir = args.run_dir.expanduser().resolve()
    checkpoint = checkpoint_path_for(args)
    device = resolve_device(str(args.device))
    model, model_config, pilot_mask, noise_reference_db = load_model(args.method, checkpoint, device)
    socket_path = Path(args.socket).expanduser()
    if socket_path.exists():
        socket_path.unlink()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = None
    if args.log_csv:
        log_path = Path(args.log_csv).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        log_handle.write(
            "request_id,frame,slot,grid_elems,model_us,python_total_us,total_us,method,label,width,depth,noise_power_db\n"
        )
        log_handle.flush()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    print(
        json.dumps(
            {
                "event": "ready",
                "socket": str(socket_path),
                "method": args.method,
                "checkpoint": str(checkpoint),
                "width": args.width,
                "depth": args.depth,
                "device": str(device),
                "noise_power_db": args.noise_power_db,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    served = 0
    try:
        while args.max_requests <= 0 or served < args.max_requests:
            conn, _ = server.accept()
            with conn:
                while args.max_requests <= 0 or served < args.max_requests:
                    try:
                        header = parse_request_header(recv_exact(conn, REQUEST.size))
                    except (EOFError, ConnectionResetError):
                        break
                    request_t0 = time.perf_counter()
                    status = 0
                    payload = b""
                    request_width = float(args.width)
                    request_depth = float(args.depth)
                    if header["magic"] != MAGIC or header["version"] != VERSION or header["header_bytes"] != REQUEST.size:
                        status = 1
                    if status == 0:
                        payload = recv_exact(conn, int(header["grid_elems"]) * 4)
                    try:
                        if status == 0:
                            raw_grid = c16_payload_to_grid(payload, nb_re=header["nb_re"], nr_symbols=header["nr_symbols"])
                            request_width = float(header["width_ppm"]) / 1_000_000.0 if int(header["width_ppm"]) > 0 else request_width
                            request_depth = float(header["depth_ppm"]) / 1_000_000.0 if int(header["depth_ppm"]) > 0 else request_depth
                            model_t0 = time.perf_counter()
                            prediction = predict_grid(
                                method=args.method,
                                model=model,
                                model_config=model_config,
                                pilot_mask=pilot_mask,
                                noise_reference_db=noise_reference_db,
                                noise_power_db=float(args.noise_power_db),
                                width=request_width,
                                depth=request_depth,
                                raw_grid=raw_grid,
                                device=device,
                            )
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            model_us = int(round((time.perf_counter() - model_t0) * 1.0e6))
                            response_payload = grid_to_c16_payload(prediction)
                        else:
                            model_us = 0
                            response_payload = b""
                    except Exception as exc:  # noqa: BLE001
                        print(json.dumps({"event": "request_error", "error": str(exc)}, ensure_ascii=True), flush=True)
                        status = 2
                        model_us = 0
                        response_payload = b""
                    python_total_us = int(round((time.perf_counter() - request_t0) * 1.0e6))
                    resp = RESPONSE.pack(
                        MAGIC,
                        VERSION,
                        RESPONSE.size,
                        int(header["request_id"]),
                        int(status),
                        int(header["grid_elems"]) if status == 0 else 0,
                        int(model_us),
                        int(python_total_us),
                    )
                    send_all(conn, resp)
                    if status == 0:
                        send_all(conn, response_payload)
                    total_us = int(round((time.perf_counter() - request_t0) * 1.0e6))
                    served += 1
                    if log_handle is not None:
                        log_handle.write(
                            f"{header['request_id']},{header['frame']},{header['slot']},{header['grid_elems']},"
                            f"{model_us},{python_total_us},{total_us},{args.method},"
                            f"w{request_width:g}_d{request_depth:g},{request_width},{request_depth},{args.noise_power_db}\n"
                        )
                        log_handle.flush()
                    if args.print_every > 0 and served % int(args.print_every) == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "served",
                                    "count": served,
                                    "request_id": header["request_id"],
                                    "model_us": model_us,
                                    "python_total_us": python_total_us,
                                    "total_us": total_us,
                                    "width": request_width,
                                    "depth": request_depth,
                                },
                                ensure_ascii=True,
                            ),
                            flush=True,
                        )
    finally:
        server.close()
        if socket_path.exists():
            socket_path.unlink()
        if log_handle is not None:
            log_handle.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--method", choices=("static", "strujepa", "dynabert", "ofa"), default="strujepa")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--socket", default=str(SERVICE_DIR / "runs" / "manual" / "runtime" / "sock"))
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--noise-power-db", type=float, default=-70.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--log-csv", default="")
    args = parser.parse_args()
    if not args.label:
        args.label = f"w{args.width:g}_d{args.depth:g}"
    return args


if __name__ == "__main__":
    raise SystemExit(serve(parse_args()))
