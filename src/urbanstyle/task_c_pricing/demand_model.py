"""Modelo de demanda con elasticidad, estacionalidad y ruido reproducible."""

from __future__ import annotations

import math


def expected_demand(
    price: float,
    reference_price: float,
    forecast_units: float,
    elasticity: float,
    seasonal_factor: float = 1.0,
    segment_multiplier: float = 1.0,
) -> float:
    """Calcula unidades esperadas con una curva de elasticidad log-log.

    El pronóstico de Task B fija la demanda base; si no está disponible, el
    entorno proporciona la media semanal histórica. Con elasticidad negativa,
    la demanda esperada es monótonamente decreciente respecto al precio.
    """
    if price <= 0 or reference_price <= 0:
        raise ValueError("El precio actual y el precio de referencia deben ser positivos.")
    if forecast_units < 0:
        raise ValueError("El pronóstico de unidades no puede ser negativo.")
    if elasticity >= 0:
        raise ValueError("La elasticidad debe ser negativa para una demanda decreciente.")
    return (
        forecast_units
        * (price / reference_price) ** elasticity
        * max(0.0, seasonal_factor)
        * max(0.0, segment_multiplier)
    )


def seasonal_multiplier(step: int, amplitude: float) -> float:
    """Multiplicador sinusoidal suave para introducir estacionalidad ligera."""
    return 1.0 + amplitude * math.sin(2 * math.pi * step / 16)
