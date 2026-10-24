"""Detección automática del mejor dispositivo disponible para PyTorch."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Selecciona CUDA, MPS o CPU, en ese orden de preferencia."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")
