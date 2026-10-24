"""División temporal, escalado y ventanas para las seis series semanales."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SERIES_COLUMNS = ("Product Category", "Gender")
TARGET_COLUMN = "units"


@dataclass(frozen=True)
class SeriesSplit:
    """Datos, escaladores y límites temporales para entrenamiento y evaluación."""

    series_keys: tuple[tuple[str, str], ...]
    series_labels: tuple[str, ...]
    weeks: tuple[pd.Timestamp, ...]
    values: np.ndarray
    normalized: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    train_end: int
    validation_end: int
    lookback: int


def load_weekly_series(path: str | Path) -> pd.DataFrame:
    """Carga y valida las series generadas por prepare_data.py."""
    weekly_path = Path(path)
    if not weekly_path.is_file():
        raise FileNotFoundError(
            f"Faltan las series semanales en {weekly_path}. "
            "Ejecuta primero python scripts/prepare_data.py."
        )
    frame = pd.read_csv(weekly_path, parse_dates=["week"])
    required = {"week", "Product Category", "Gender", TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Faltan columnas en weekly_sales.csv: " + ", ".join(sorted(missing)))
    if frame[list(required)].isna().any().any():
        raise ValueError("Las series semanales contienen valores nulos.")
    if (frame[TARGET_COLUMN] < 0).any():
        raise ValueError("Las ventas semanales no pueden ser negativas.")
    return frame.sort_values([*SERIES_COLUMNS, "week"]).reset_index(drop=True)


def prepare_series(
    frame: pd.DataFrame,
    *,
    categories: list[str],
    genders: list[str],
    lookback: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> SeriesSplit:
    """Construye matriz serie×semana y ajusta media/desvío solo en train."""
    if lookback < 1:
        raise ValueError("La longitud de ventana debe ser positiva.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Las fracciones de train y validación deben estar entre 0 y 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train más validación debe dejar un tramo de test positivo.")

    series_keys = tuple((category, gender) for category in categories for gender in genders)
    series_labels = tuple(f"{category} · {gender}" for category, gender in series_keys)
    weeks = tuple(pd.DatetimeIndex(sorted(frame["week"].unique())))
    if len(weeks) <= lookback + 2:
        raise ValueError("No hay suficientes semanas para las ventanas y los tres splits.")

    indexed = frame.pivot(index="week", columns=SERIES_COLUMNS, values=TARGET_COLUMN)
    indexed = indexed.reindex(index=weeks, columns=pd.MultiIndex.from_tuples(series_keys))
    if indexed.isna().any().any():
        raise ValueError("Faltan semanas en alguna serie; prepara el dataset con relleno de ceros.")
    values = indexed.to_numpy(dtype=np.float32).T
    train_end = max(lookback + 1, int(len(weeks) * train_fraction))
    validation_end = max(train_end + 1, int(len(weeks) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(weeks) - 1)
    means = values[:, :train_end].mean(axis=1, keepdims=True)
    scales = values[:, :train_end].std(axis=1, keepdims=True)
    scales = np.where(scales < 1e-8, 1.0, scales)
    normalized = ((values - means) / scales).astype(np.float32)
    return SeriesSplit(
        series_keys=series_keys,
        series_labels=series_labels,
        weeks=weeks,
        values=values,
        normalized=normalized,
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        train_end=train_end,
        validation_end=validation_end,
        lookback=lookback,
    )


def build_windows(
    split: SeriesSplit, target_start: int, target_end: int, lookback: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genera ventanas cuyos objetivos pertenecen a [target_start, target_end)."""
    window = split.lookback if lookback is None else lookback
    if window < 1 or window > target_start:
        raise ValueError("La ventana debe ser positiva y no superar el índice del primer objetivo.")
    if not target_start < target_end <= len(split.weeks):
        raise ValueError("Los límites temporales del split no son válidos.")
    contexts: list[np.ndarray] = []
    targets: list[float] = []
    series_ids: list[int] = []
    target_indices: list[int] = []
    for series_id, series in enumerate(split.normalized):
        for target_index in range(target_start, target_end):
            contexts.append(series[target_index - window : target_index])
            targets.append(float(series[target_index]))
            series_ids.append(series_id)
            target_indices.append(target_index)
    return (
        np.asarray(contexts, dtype=np.float32),
        np.asarray(series_ids, dtype=np.int64),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_indices, dtype=np.int64),
    )
