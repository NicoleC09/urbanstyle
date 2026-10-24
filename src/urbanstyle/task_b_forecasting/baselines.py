"""Baselines sencillos para pronóstico un paso adelante."""

from __future__ import annotations

import numpy as np


def naive_forecast(history: np.ndarray) -> float:
    """Predice el último valor observado."""
    if len(history) == 0:
        raise ValueError("Se requiere al menos una observación histórica.")
    return float(history[-1])


def moving_average_forecast(history: np.ndarray, window: int = 4) -> float:
    """Predice la media de las últimas observaciones disponibles."""
    if len(history) == 0 or window < 1:
        raise ValueError("La historia y el tamaño de la media móvil deben ser positivos.")
    return float(np.mean(history[-window:]))


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Calcula MAE y RMSE entre valores reales y predichos."""
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Actual y predicho deben tener la misma forma no vacía.")
    errors = actual.astype(float) - predicted.astype(float)
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors**2)))
