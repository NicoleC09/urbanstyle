"""Inicialización reproducible de generadores pseudoaleatorios."""

from __future__ import annotations

import random


def seed_everything(seed: int) -> None:
    """Fija semillas de Python, NumPy y PyTorch si están instalados."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
