"""Entorno Gymnasium de precio dinámico para UrbanStyle.

Supuestos: cada episodio representa una categoría y un segmento. El pronóstico
de Task B, si se proporciona, fija la demanda base; en su ausencia se usa la
media semanal histórica. La demanda esperada sigue una curva de elasticidad
log-log calibrada o con fallback documentado, recibe estacionalidad ligera y
ruido gaussiano truncado para evitar unidades negativas. No se simulan inventario
ni faltantes. La recompensa es margen bruto del producto (precio menos costo
unitario configurado, por unidades vendidas). El entorno conserva también el
ingreso como métrica, pero no como recompensa de aprendizaje. El precio está
limitado alrededor de su referencia histórica.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .demand_model import expected_demand, seasonal_multiplier


@dataclass(frozen=True)
class ProductPricingData:
    """Parámetros históricos y de demanda de una categoría."""

    name: str
    reference_price: float
    baseline_units: float
    elasticity: float
    elasticity_method: str = "fallback"


class DynamicPricingEnv(gym.Env[np.ndarray, int]):
    """Entorno discreto de tres acciones para fijar el precio de una categoría.

    La observación contiene precio relativo, segmento, ventas recientes y
    demanda base normalizados. Acciones: bajar, mantener y subir el precio.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        products: list[ProductPricingData],
        *,
        episode_steps: int = 16,
        price_step_fraction: float = 0.05,
        price_min_multiplier: float = 0.6,
        price_max_multiplier: float = 1.5,
        demand_noise_std: float = 0.08,
        seasonality_amplitude: float = 0.06,
        recent_sales_window: int = 4,
        segment_multipliers: tuple[float, ...] = (0.92, 1.0, 1.08),
        unit_cost_fraction: float = 0.55,
        forecast_units: dict[str, float] | None = None,
    ):
        super().__init__()
        if not products:
            raise ValueError("El entorno requiere al menos un producto.")
        if episode_steps < 1 or recent_sales_window < 1:
            raise ValueError("El horizonte y la ventana de ventas deben ser positivos.")
        if not 0 < price_min_multiplier < 1 < price_max_multiplier:
            raise ValueError("Los límites de precio deben rodear el precio de referencia.")
        if not 0 <= unit_cost_fraction < 1:
            raise ValueError("La fracción de coste debe estar entre 0 y 1.")
        if not segment_multipliers or any(value <= 0 for value in segment_multipliers):
            raise ValueError("Los multiplicadores de segmento deben ser positivos.")

        self.products = products
        self.episode_steps = episode_steps
        self.price_step_fraction = price_step_fraction
        self.price_min_multiplier = price_min_multiplier
        self.price_max_multiplier = price_max_multiplier
        self.demand_noise_std = demand_noise_std
        self.seasonality_amplitude = seasonality_amplitude
        self.recent_sales_window = recent_sales_window
        self.segment_multipliers = segment_multipliers
        self.unit_cost_fraction = unit_cost_fraction
        self.forecasts = forecast_units or {}
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.zeros(3 + recent_sales_window, dtype=np.float32),
            high=np.ones(3 + recent_sales_window, dtype=np.float32),
            dtype=np.float32,
        )
        self.product: ProductPricingData
        self.segment_index = 0
        self.price = 1.0
        self.step_index = 0
        self.recent_sales: deque[float] = deque(maxlen=recent_sales_window)
        self.forecast = 1.0

    def _observation(self) -> np.ndarray:
        segment = self.segment_index / max(1, len(self.segment_multipliers) - 1)
        price_range = self.price_max_multiplier - self.price_min_multiplier
        normalized_price = (self.price - self.price_min_multiplier) / price_range
        normalized_forecast = self.forecast / (self.forecast + 10.0)
        recent = [value / max(1.0, self.forecast * 2.0) for value in self.recent_sales]
        recent = [0.0] * (self.recent_sales_window - len(recent)) + recent
        values = [normalized_price, segment, normalized_forecast, *recent]
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Comienza episodio con producto y segmento reproducibles."""
        super().reset(seed=seed)
        options = options or {}
        product_name = options.get("product")
        matching = [product for product in self.products if product.name == product_name]
        self.product = matching[0] if matching else self.np_random.choice(self.products)
        self.segment_index = int(
            options.get("segment", self.np_random.integers(len(self.segment_multipliers)))
        )
        if not 0 <= self.segment_index < len(self.segment_multipliers):
            raise ValueError("El índice de segmento está fuera de rango.")
        initial_price = float(options.get("initial_price", 1.0))
        self.price = float(
            np.clip(initial_price, self.price_min_multiplier, self.price_max_multiplier)
        )
        self.step_index = 0
        self.forecast = max(
            0.1,
            float(self.forecasts.get(self.product.name, self.product.baseline_units)),
        )
        self.recent_sales = deque(
            [self.forecast] * self.recent_sales_window,
            maxlen=self.recent_sales_window,
        )
        return self._observation(), {
            "product": self.product.name,
            "segment": self.segment_index,
            "price": self.product.reference_price,
            "price_index": self.price,
            "forecast_units": self.forecast,
            "elasticity": self.product.elasticity,
            "elasticity_method": self.product.elasticity_method,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Actualiza precio, demanda, ingreso y estado del episodio."""
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")
        direction = int(action) - 1
        self.price = float(
            np.clip(
                self.price + direction * self.price_step_fraction,
                self.price_min_multiplier,
                self.price_max_multiplier,
            )
        )
        seasonal = seasonal_multiplier(self.step_index, self.seasonality_amplitude)
        mean_demand = expected_demand(
            self.price,
            1.0,
            self.forecast,
            self.product.elasticity,
            seasonal,
            self.segment_multipliers[self.segment_index],
        )
        noise = float(self.np_random.normal(0.0, self.demand_noise_std))
        units = max(0, int(round(mean_demand * (1.0 + noise))))
        unit_cost = self.unit_cost_fraction * self.product.reference_price
        revenue = units * self.price * self.product.reference_price
        profit = units * (self.price * self.product.reference_price - unit_cost)
        self.recent_sales.append(float(units))
        self.step_index += 1
        terminated = False
        truncated = self.step_index >= self.episode_steps
        observation = self._observation()
        info = {
            "product": self.product.name,
            "price": self.price * self.product.reference_price,
            "price_index": self.price,
            "units_sold": units,
            "expected_units": mean_demand,
            "revenue": revenue,
            "profit": profit,
            "reward": profit,
            "action": int(action),
            "step": self.step_index,
            "elasticity": self.product.elasticity,
        }
        return observation, float(profit), terminated, truncated, info
