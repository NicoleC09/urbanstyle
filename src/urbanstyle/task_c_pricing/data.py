"""Carga de productos, precios y elasticidades preparados para Task C."""

from __future__ import annotations

import json

import pandas as pd
from src.urbanstyle.task_c_pricing.env import DynamicPricingEnv, ProductPricingData
from src.urbanstyle.utils.config import resolve_path


def mean_weekly_units(weekly: pd.DataFrame, category: str) -> float:
    """Suma géneros dentro de cada semana y devuelve demanda media semanal."""
    category_weeks = weekly.loc[weekly["Product Category"] == category]
    if category_weeks.empty:
        return 0.0
    if "week" in category_weeks.columns:
        return float(category_weeks.groupby("week")["units"].sum().mean())
    return float(category_weeks["units"].mean())


def load_forecast_units(config: dict) -> dict[str, float]:
    """Agrega el pronóstico LSTM de ambos géneros por categoría, si existe."""
    prediction_path = resolve_path(config["paths"]["outputs"]) / "predictions.csv"
    if not prediction_path.is_file():
        return {}
    predictions = pd.read_csv(prediction_path)
    required = {"category", "gender", "lstm"}
    if not required.issubset(predictions.columns):
        return {}
    forecasts = predictions.groupby("category")["lstm"].sum()
    return {
        category: max(0.1, float(forecasts[category]))
        for category in config["data"]["categories"]
        if category in forecasts
    }


def load_product_data(config: dict) -> list[ProductPricingData]:
    """Combina estadísticas de precio, volumen semanal y elasticidad calibrada."""
    processed_dir = resolve_path(config["paths"]["processed_data"])
    calibration_path = processed_dir / "calibration.json"
    weekly_path = processed_dir / "weekly_sales.csv"
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Falta {calibration_path}; ejecuta python scripts/prepare_data.py."
        )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    weekly = pd.read_csv(weekly_path) if weekly_path.is_file() else None
    products: list[ProductPricingData] = []
    for category in config["data"]["categories"]:
        stats = calibration.get("price_statistics", {}).get(category)
        elasticity_info = calibration.get("elasticity", {}).get(category)
        if not stats or not elasticity_info:
            continue
        if weekly is not None and {"Product Category", "units"}.issubset(weekly.columns):
            baseline_units = mean_weekly_units(weekly, category)
        else:
            raw_path = resolve_path(config["paths"]["raw_data"])
            if not raw_path.is_file():
                raise FileNotFoundError(
                    f"Falta la serie semanal y el CSV original: {raw_path}. "
                    "Ejecuta python scripts/prepare_data.py."
                )
            raw = pd.read_csv(raw_path)
            baseline_units = float(
                raw.loc[raw["Product Category"] == category]
                .groupby("Date")["Quantity"]
                .sum()
                .mean()
            )
        products.append(
            ProductPricingData(
                name=category,
                reference_price=float(stats["median"]),
                baseline_units=max(0.1, baseline_units),
                elasticity=float(elasticity_info["elasticity"]),
                elasticity_method=str(elasticity_info["method"]),
            )
        )
    if not products:
        raise ValueError("No hay productos calibrados para entrenar Task C.")
    return products


def make_environment(
    config: dict,
    products: list[ProductPricingData],
    forecast_units: dict[str, float] | None = None,
) -> DynamicPricingEnv:
    """Construye el entorno desde los parámetros de config.yaml."""
    pricing = config["pricing"]
    return DynamicPricingEnv(
        products,
        episode_steps=int(pricing["episode_steps"]),
        price_step_fraction=float(pricing["price_step_fraction"]),
        price_min_multiplier=float(pricing["price_min_multiplier"]),
        price_max_multiplier=float(pricing["price_max_multiplier"]),
        demand_noise_std=float(pricing["demand_noise_std"]),
        seasonality_amplitude=float(pricing["seasonality_amplitude"]),
        recent_sales_window=int(pricing["recent_sales_window"]),
        segment_multipliers=tuple(float(value) for value in pricing["segment_multipliers"]),
        unit_cost_fraction=float(pricing["unit_cost_fraction"]),
        forecast_units=forecast_units,
    )
