"""Entrenamiento standalone del LSTM global y evaluación de baselines."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.urbanstyle.task_b_forecasting.baselines import (
    moving_average_forecast,
    naive_forecast,
    regression_metrics,
)
from src.urbanstyle.task_b_forecasting.dataset import (
    SeriesSplit,
    build_windows,
    load_weekly_series,
    prepare_series,
)
from src.urbanstyle.task_b_forecasting.lstm_model import GlobalDemandLSTM
from src.urbanstyle.utils.config import PROJECT_ROOT, load_config, resolve_path
from src.urbanstyle.utils.seed import seed_everything
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger("urbanstyle.task_b.train")


def _tensor_data(windows: np.ndarray, series_ids: np.ndarray, targets: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(windows),
        torch.from_numpy(series_ids),
        torch.from_numpy(targets),
    )


def _predict(
    model: GlobalDemandLSTM,
    split: SeriesSplit,
    target_start: int,
    target_end: int,
    lookback: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows, series_ids, targets, indices = build_windows(split, target_start, target_end, lookback)
    model.eval()
    with torch.no_grad():
        predicted = (
            model(
                torch.from_numpy(windows).to(device),
                torch.from_numpy(series_ids).to(device),
            )
            .cpu()
            .numpy()
        )
    return predicted, targets, series_ids, indices


def _denormalize(values: np.ndarray, series_ids: np.ndarray, split: SeriesSplit) -> np.ndarray:
    return values * split.scales[series_ids, 0] + split.means[series_ids, 0]


def _evaluate_contexts(
    model: GlobalDemandLSTM,
    split: SeriesSplit,
    context_windows: list[int],
    moving_average_window: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for window in context_windows:
        predicted_scaled, actual_scaled, series_ids, indices = _predict(
            model,
            split,
            split.validation_end,
            len(split.weeks),
            window,
            device,
        )
        actual = _denormalize(actual_scaled, series_ids, split)
        predicted = np.maximum(0.0, _denormalize(predicted_scaled, series_ids, split))
        for series_id, label in enumerate(split.series_labels):
            mask = series_ids == series_id
            series_actual = actual[mask]
            series_predicted = predicted[mask]
            series_indices = indices[mask]
            naive = np.asarray([split.values[series_id, index - 1] for index in series_indices])
            moving = np.asarray(
                [
                    moving_average_forecast(
                        split.values[series_id, max(0, index - moving_average_window) : index],
                        moving_average_window,
                    )
                    for index in series_indices
                ]
            )
            for model_name, estimates in (
                ("LSTM", series_predicted),
                ("Media móvil", moving),
                ("Ingenuo", naive),
            ):
                mae, rmse = regression_metrics(series_actual, estimates)
                metric_rows.append(
                    {
                        "series": label,
                        "category": split.series_keys[series_id][0],
                        "gender": split.series_keys[series_id][1],
                        "model": model_name,
                        "context_weeks": window,
                        "mae": mae,
                        "rmse": rmse,
                    }
                )
            for _offset, (index, observed, estimate, avg, last) in enumerate(
                zip(series_indices, series_actual, series_predicted, moving, naive, strict=True)
            ):
                prediction_rows.append(
                    {
                        "series": label,
                        "category": split.series_keys[series_id][0],
                        "gender": split.series_keys[series_id][1],
                        "week": split.weeks[index].date().isoformat(),
                        "context_weeks": window,
                        "split": "test",
                        "actual": float(observed),
                        "lstm": float(estimate),
                        "moving_average": float(avg),
                        "naive": float(last),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def train_forecaster(
    config: dict,
    *,
    max_epochs: int | None = None,
    seed: int | None = None,
    data_path: str | Path | None = None,
) -> dict[str, Path]:
    """Entrena con early stopping y exporta métricas, predicciones y checkpoint."""
    task = config["task_b"]
    seed = int(config["project"]["seed"] if seed is None else seed)
    max_epochs = int(task["max_epochs"] if max_epochs is None else max_epochs)
    if max_epochs < 1:
        raise ValueError("El número de épocas debe ser positivo.")
    seed_everything(seed)
    random.seed(seed)
    weekly_path = (
        Path(data_path)
        if data_path
        else resolve_path(config["paths"]["processed_data"]) / "weekly_sales.csv"
    )
    frame = load_weekly_series(weekly_path)
    split = prepare_series(
        frame,
        categories=config["data"]["categories"],
        genders=config["data"]["genders"],
        lookback=int(task["lookback"]),
        train_fraction=float(task["train_fraction"]),
        validation_fraction=float(task["validation_fraction"]),
    )
    train_windows, train_ids, train_targets, _ = build_windows(
        split, split.lookback, split.train_end
    )
    val_windows, val_ids, val_targets, _ = build_windows(
        split, split.train_end, split.validation_end
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalDemandLSTM(
        num_series=len(split.series_keys),
        embedding_size=int(task["embedding_size"]),
        hidden_size=int(task["hidden_size"]),
        num_layers=int(task["num_layers"]),
        dropout=float(task["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task["learning_rate"]),
        weight_decay=float(task["weight_decay"]),
    )
    loss_function = nn.MSELoss()
    loader = DataLoader(
        _tensor_data(train_windows, train_ids, train_targets),
        batch_size=int(task["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_x = torch.from_numpy(val_windows).to(device)
    val_series = torch.from_numpy(val_ids).to(device)
    val_y = torch.from_numpy(val_targets).to(device)
    patience = int(task["early_stopping_patience"])
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    training_history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        for batch_windows, batch_ids, batch_targets in loader:
            batch_windows = batch_windows.to(device)
            batch_ids = batch_ids.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_windows, batch_ids)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(val_x, val_series), val_y).cpu())
        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(batch_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("El entrenamiento no produjo un checkpoint válido.")
    model.load_state_dict(best_state)
    context_windows = sorted({int(window) for window in task["context_windows"]} | {split.lookback})
    invalid_windows = [window for window in context_windows if window >= split.validation_end]
    if invalid_windows:
        raise ValueError(f"Ventanas de contexto demasiado largas: {invalid_windows}")
    metrics, test_predictions = _evaluate_contexts(
        model,
        split,
        context_windows,
        int(task["moving_average_window"]),
        device,
    )
    primary_context = split.lookback
    future_predictions: list[dict[str, Any]] = []
    for series_id, label in enumerate(split.series_labels):
        normalized_context = split.normalized[series_id, -primary_context:]
        with torch.no_grad():
            normalized_prediction = float(
                model(
                    torch.from_numpy(normalized_context[None, :]).to(device),
                    torch.tensor([series_id], dtype=torch.long, device=device),
                ).cpu()[0]
            )
        predicted = max(
            0.0,
            normalized_prediction * float(split.scales[series_id, 0])
            + float(split.means[series_id, 0]),
        )
        history = split.values[series_id]
        future_predictions.append(
            {
                "series": label,
                "category": split.series_keys[series_id][0],
                "gender": split.series_keys[series_id][1],
                "week": (split.weeks[-1] + pd.Timedelta(days=7)).date().isoformat(),
                "lstm": predicted,
                "moving_average": moving_average_forecast(
                    history, int(task["moving_average_window"])
                ),
                "naive": naive_forecast(history),
                "context_weeks": primary_context,
            }
        )

    model_dir = resolve_path(config["paths"]["models"])
    output_dir = resolve_path(config["paths"]["outputs"])
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "lstm.pt"
    torch.save(
        {
            "state_dict": best_state,
            "series_keys": split.series_keys,
            "series_labels": split.series_labels,
            "means": split.means,
            "scales": split.scales,
            "lookback": split.lookback,
            "context_windows": context_windows,
            "model_config": {
                "num_series": len(split.series_keys),
                "embedding_size": int(task["embedding_size"]),
                "hidden_size": int(task["hidden_size"]),
                "num_layers": int(task["num_layers"]),
                "dropout": float(task["dropout"]),
            },
        },
        checkpoint_path,
    )
    metrics_path = output_dir / "forecast_metrics.csv"
    test_path = output_dir / "forecast_test_predictions.csv"
    forecast_path = output_dir / "predictions.csv"
    history_path = output_dir / "forecast_training_history.csv"
    report_path = output_dir / "forecast_metrics.json"
    metrics.to_csv(metrics_path, index=False)
    test_predictions.to_csv(test_path, index=False)
    pd.DataFrame(future_predictions).to_csv(forecast_path, index=False)
    pd.DataFrame(training_history).to_csv(history_path, index=False)
    report_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "epochs_completed": len(training_history),
                "best_validation_mse_normalized": best_loss,
                "train_weeks": split.train_end,
                "validation_weeks": split.validation_end - split.train_end,
                "test_weeks": len(split.weeks) - split.validation_end,
                "normalization": "media y desvío ajustados solo con el split train, por serie",
                "metrics": metrics.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plot_path = output_dir / "forecast_training_curve.png"
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.plot(
        [row["epoch"] for row in training_history],
        [row["train_loss"] for row in training_history],
        label="Train",
    )
    axis.plot(
        [row["epoch"] for row in training_history],
        [row["validation_loss"] for row in training_history],
        label="Validación",
    )
    axis.set(title="LSTM global · pérdida MSE normalizada", xlabel="Época", ylabel="MSE")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    LOGGER.info(
        "Mejor checkpoint guardado en %s tras %d épocas", checkpoint_path, len(training_history)
    )
    return {
        "model": checkpoint_path,
        "metrics": metrics_path,
        "metrics_json": report_path,
        "predictions": forecast_path,
        "test_predictions": test_path,
        "training_history": history_path,
        "training_plot": plot_path,
    }


def train_from_project(
    *, max_epochs: int | None = None, seed: int | None = None, config_path: str | Path | None = None
) -> dict[str, Path]:
    """Carga config.yaml y entrena el modelo semanal."""
    return train_forecaster(
        load_config(config_path or PROJECT_ROOT / "config.yaml"),
        max_epochs=max_epochs,
        seed=seed,
    )


def evaluate_saved_forecaster(
    config: dict, *, checkpoint_path: str | Path | None = None
) -> dict[str, Path]:
    """Reevalúa un checkpoint existente y refresca métricas sin entrenar."""
    task = config["task_b"]
    checkpoint_path = (
        Path(checkpoint_path)
        if checkpoint_path
        else resolve_path(config["paths"]["models"]) / "lstm.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Falta el modelo {checkpoint_path}; ejecuta python scripts/train_lstm.py."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weekly_path = resolve_path(config["paths"]["processed_data"]) / "weekly_sales.csv"
    frame = load_weekly_series(weekly_path)
    split = prepare_series(
        frame,
        categories=config["data"]["categories"],
        genders=config["data"]["genders"],
        lookback=int(checkpoint["lookback"]),
        train_fraction=float(task["train_fraction"]),
        validation_fraction=float(task["validation_fraction"]),
    )
    model_config = checkpoint["model_config"]
    model = GlobalDemandLSTM(**model_config)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    context_windows = [int(value) for value in checkpoint["context_windows"]]
    metrics, test_predictions = _evaluate_contexts(
        model,
        split,
        context_windows,
        int(task["moving_average_window"]),
        device,
    )
    output_dir = resolve_path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "forecast_metrics.csv"
    test_path = output_dir / "forecast_test_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    test_predictions.to_csv(test_path, index=False)
    return {"metrics": metrics_path, "test_predictions": test_path}
