"""Políticas deterministas de referencia para evaluación de precios."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .env import DynamicPricingEnv

Policy = Callable[[np.ndarray, DynamicPricingEnv], int]


def theoretical_price_multiplier(
    elasticity: float,
    cost_fraction: float,
    price_min_multiplier: float,
    price_max_multiplier: float,
) -> float:
    """Devuelve el óptimo isoelástico de margen, limitado al dominio de precios."""
    elasticity = float(elasticity)
    if elasticity >= -1.0 or abs(elasticity + 1.0) < 1e-9:
        unconstrained = float("inf")
    else:
        unconstrained = elasticity / (elasticity + 1.0) * float(cost_fraction)
    return float(np.clip(unconstrained, price_min_multiplier, price_max_multiplier))


def theoretical_optimal_price(environment: DynamicPricingEnv) -> float:
    """Precio isoelástico de margen máximo, acotado a los límites del entorno."""
    return theoretical_price_multiplier(
        environment.product.elasticity,
        environment.unit_cost_fraction,
        environment.price_min_multiplier,
        environment.price_max_multiplier,
    )


def fixed_price_action(target_multiplier: float, environment: DynamicPricingEnv) -> int:
    """Mueve el precio un paso hacia un nivel fijo bajo, medio o alto."""
    target = float(
        np.clip(
            target_multiplier,
            environment.price_min_multiplier,
            environment.price_max_multiplier,
        )
    )
    if environment.price < target - 1e-9:
        return 2
    if environment.price > target + 1e-9:
        return 0
    return 1


def low_price_policy(_: np.ndarray, environment: DynamicPricingEnv) -> int:
    """Política que converge al límite inferior de precio."""
    return fixed_price_action(environment.price_min_multiplier, environment)


def medium_price_policy(_: np.ndarray, environment: DynamicPricingEnv) -> int:
    """Política que mantiene el precio de referencia."""
    return fixed_price_action(1.0, environment)


def high_price_policy(_: np.ndarray, environment: DynamicPricingEnv) -> int:
    """Política que converge al límite superior de precio."""
    return fixed_price_action(environment.price_max_multiplier, environment)


def theoretical_price_policy(_: np.ndarray, environment: DynamicPricingEnv) -> int:
    """Acerca al óptimo de margen isoelástico, acotado por los límites de precio."""
    return fixed_price_action(theoretical_optimal_price(environment), environment)


STATIC_POLICIES: dict[str, Policy] = {
    "Precio bajo": low_price_policy,
    "Precio medio": medium_price_policy,
    "Precio alto": high_price_policy,
    "Óptimo teórico": theoretical_price_policy,
}
