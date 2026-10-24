"""Valida el CSV original y genera series semanales y calibraciones."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import linregress
from src.urbanstyle.utils.config import load_config, resolve_path
from src.urbanstyle.utils.logging import configure_logging

LOGGER = logging.getLogger("urbanstyle.prepare_data")
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


class DataValidationError(ValueError):
    """Error de esquema o calidad en los datos de entrada."""


def validate_dataset(frame: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Comprueba el esquema y la calidad mínima, y devuelve columnas tipadas."""
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise DataValidationError(
            "Faltan columnas obligatorias en el CSV: " + ", ".join(missing_columns)
        )

    frame = frame.loc[:, required_columns].copy()
    exact_duplicates = int(frame.duplicated().sum())
    if exact_duplicates:
        LOGGER.warning("Se eliminarán %d filas duplicadas exactas.", exact_duplicates)
        frame = frame.drop_duplicates().copy()

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in ("Transaction ID", "Age", "Quantity", "Price per Unit", "Total Amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    null_counts = frame[required_columns].isna().sum()
    invalid_columns = null_counts[null_counts > 0]
    if not invalid_columns.empty:
        details = ", ".join(f"{column}: {count}" for column, count in invalid_columns.items())
        raise DataValidationError(f"Hay fechas, tipos inválidos o valores nulos ({details}).")

    duplicate_ids = int(frame["Transaction ID"].duplicated().sum())
    if duplicate_ids:
        raise DataValidationError(
            f"Se encontraron {duplicate_ids} identificadores de transacción repetidos."
        )

    for column in ("Quantity", "Price per Unit", "Total Amount"):
        if (frame[column] < 0).any():
            raise DataValidationError(f"La columna '{column}' contiene valores negativos.")
    if frame.empty:
        raise DataValidationError("El CSV no contiene transacciones después de validarlo.")
    return frame


def build_weekly_series(
    frame: pd.DataFrame,
    categories: list[str],
    genders: list[str],
    frequency: str,
) -> pd.DataFrame:
    """Agrega ventas y unidades por semana para cada categoría y género."""
    weekly_frame = frame.copy()
    weekly_frame["week"] = weekly_frame["Date"].dt.to_period(frequency).dt.end_time.dt.normalize()
    grouped = (
        weekly_frame.groupby(["week", "Product Category", "Gender"], observed=True)
        .agg(sales=("Total Amount", "sum"), units=("Quantity", "sum"))
        .reset_index()
    )
    weeks = pd.date_range(grouped["week"].min(), grouped["week"].max(), freq=frequency)
    index = pd.MultiIndex.from_product(
        [weeks, categories, genders], names=["week", "Product Category", "Gender"]
    )
    result = (
        grouped.set_index(["week", "Product Category", "Gender"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    result["sales"] = result["sales"].astype(float)
    result["units"] = result["units"].astype(float)
    return result


def drop_partial_edge_weeks(frame: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, list[str]]:
    """Descarta periodos de borde no cubiertos por completo por el rango temporal.

    La completitud se determina con los límites calendario del periodo, no con
    cuántos días tienen transacciones, porque un día completo puede no registrar ventas.
    """
    if frame.empty:
        return frame.copy(), []
    dates = pd.to_datetime(frame["Date"])
    first_period = dates.min().to_period(frequency)
    last_period = dates.max().to_period(frequency)
    partial_periods = []
    first_is_partial = dates.min().normalize() > first_period.start_time.normalize()
    last_is_partial = dates.max().normalize() < last_period.end_time.normalize()
    if first_period == last_period:
        if first_is_partial or last_is_partial:
            partial_periods.append(first_period)
    else:
        if first_is_partial:
            partial_periods.append(first_period)
        if last_is_partial:
            partial_periods.append(last_period)
    excluded = {period.end_time.normalize() for period in partial_periods}
    complete = frame.loc[~dates.dt.to_period(frequency).dt.end_time.dt.normalize().isin(excluded)]
    labels = [period.end_time.normalize().date().isoformat() for period in partial_periods]
    return complete.copy(), labels


def estimate_elasticity(
    frame: pd.DataFrame,
    categories: list[str],
    fallback: float,
    minimum_observations: int,
    significance_level: float,
) -> dict[str, dict[str, Any]]:
    """Estima elasticidad log-log por categoría, con fallback explícito."""
    estimates: dict[str, dict[str, Any]] = {}
    for category in categories:
        category_rows = frame.loc[frame["Product Category"] == category].copy()
        daily = (
            category_rows.groupby("Date")
            .agg(units=("Quantity", "sum"), price=("Price per Unit", "mean"))
            .query("units > 0 and price > 0")
        )
        reason: str | None = None
        elasticity = fallback
        p_value: float | None = None
        r_squared: float | None = None

        if len(daily) < minimum_observations:
            reason = f"Solo hay {len(daily)} observaciones diarias válidas."
        else:
            try:
                regression = linregress(np.log(daily["price"]), np.log(daily["units"]))
            except ValueError:
                reason = "No hay variación suficiente de precios para estimar la regresión."
            else:
                p_value = float(regression.pvalue)
                r_squared = float(regression.rvalue**2)
                if p_value >= significance_level:
                    reason = f"La pendiente no es significativa (p={p_value:.4f})."
                elif regression.slope >= 0:
                    reason = "La pendiente estimada no representa una demanda decreciente."
                else:
                    elasticity = float(regression.slope)

        estimates[category] = {
            "elasticity": elasticity,
            "method": "fallback" if reason else "log_log_ols",
            "observations": int(len(daily)),
            "p_value": p_value,
            "r_squared": r_squared,
            "fallback_reason": reason,
        }
        if reason:
            LOGGER.warning(
                "%s: se usa elasticidad de fallback %.2f. %s",
                category,
                fallback,
                reason,
            )
    return estimates


def prepare_data(config_path: Path, input_path: Path | None = None) -> tuple[Path, Path]:
    """Valida el dataset y guarda las series y parámetros derivados."""
    config = load_config(config_path)
    raw_path = input_path or resolve_path(config["paths"]["raw_data"])
    if not raw_path.is_file():
        raise FileNotFoundError(
            "No se encontró el dataset. Descarga el CSV de Kaggle “Retail Sales Dataset” "
            "(mohammadtalib786) y colócalo en: "
            f"{raw_path}"
        )

    data_config = config["data"]
    frame = pd.read_csv(raw_path)
    frame = validate_dataset(frame, data_config["required_columns"])
    categories = data_config["categories"]
    genders = data_config["genders"]
    unexpected_categories = sorted(set(frame["Product Category"]) - set(categories))
    unexpected_genders = sorted(set(frame["Gender"]) - set(genders))
    if unexpected_categories or unexpected_genders:
        details = []
        if unexpected_categories:
            details.append("categorías no configuradas: " + ", ".join(unexpected_categories))
        if unexpected_genders:
            details.append("géneros no configurados: " + ", ".join(unexpected_genders))
        raise DataValidationError("El CSV contiene " + "; ".join(details) + ".")

    weekly_frame, excluded_weeks = drop_partial_edge_weeks(frame, data_config["week_frequency"])
    if weekly_frame.empty:
        raise DataValidationError(
            "No quedan semanas calendario completas para construir el pronóstico."
        )
    weekly = build_weekly_series(weekly_frame, categories, genders, data_config["week_frequency"])
    category_sales = frame.groupby("Product Category")["Total Amount"].sum()
    total_sales = float(category_sales.sum())
    shares = {
        category: float(category_sales.get(category, 0.0) / total_sales) if total_sales else 0.0
        for category in categories
    }
    price_stats = {
        category: {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        for category, values in frame.groupby("Product Category")["Price per Unit"]
    }
    pricing_config = config["pricing"]
    elasticities = estimate_elasticity(
        frame,
        categories,
        float(pricing_config["elasticity_fallback"]),
        int(pricing_config["elasticity_min_observations"]),
        float(pricing_config["elasticity_significance_level"]),
    )

    output_dir = resolve_path(config["paths"]["processed_data"])
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly_path = output_dir / "weekly_sales.csv"
    calibration_path = output_dir / "calibration.json"
    weekly.to_csv(weekly_path, index=False, date_format="%Y-%m-%d")
    calibration = {
        "source": str(raw_path.relative_to(PROJECT_ROOT))
        if raw_path.is_relative_to(PROJECT_ROOT)
        else str(raw_path),
        "transactions": int(len(frame)),
        "date_range": {
            "start": frame["Date"].min().date().isoformat(),
            "end": frame["Date"].max().date().isoformat(),
        },
        "weekly_series_boundary_policy": (
            "Se excluyen los periodos W-SUN inicial/final cuando el rango de fechas no cubre "
            "desde el lunes hasta el domingo completos; no se infiere completitud por actividad "
            "diaria."
        ),
        "excluded_partial_week_ends": excluded_weeks,
        "weekly_series_date_range": {
            "start": weekly["week"].min().date().isoformat(),
            "end": weekly["week"].max().date().isoformat(),
        },
        "category_sales_share": shares,
        "price_statistics": price_stats,
        "elasticity": elasticities,
        "elasticity_fallback_assumption": (
            "Se usa el valor configurable si hay pocas observaciones, la regresión log-log no "
            "es significativa o la pendiente no indica demanda decreciente."
        ),
    }
    calibration_path.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Series semanales guardadas en %s", weekly_path)
    LOGGER.info("Calibraciones guardadas en %s", calibration_path)
    return weekly_path, calibration_path


def main() -> int:
    """Punto de entrada de la preparación de datos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Ruta a config.yaml")
    parser.add_argument("--input", type=Path, help="Sobrescribe la ruta del CSV de entrada")
    args = parser.parse_args()
    configure_logging()
    try:
        weekly_path, calibration_path = prepare_data(args.config, args.input)
    except (DataValidationError, FileNotFoundError, KeyError, ValueError) as error:
        LOGGER.error("Preparación cancelada: %s", error)
        return 1
    LOGGER.info("Preparación terminada: %s | %s", weekly_path, calibration_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
