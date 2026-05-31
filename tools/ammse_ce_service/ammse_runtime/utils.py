"""Small runtime helpers used by the elastic A-MMSE CE socket service."""

from __future__ import annotations

import numpy as np
import torch


EPS = 1.0e-12


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def complex_vector_to_channels(vector: np.ndarray) -> np.ndarray:
    return np.stack((vector.real, vector.imag), axis=0).astype(np.float32, copy=False)
