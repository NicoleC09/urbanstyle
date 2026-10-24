"""Pruebas de split cronológico, normalización train-only y shapes del LSTM."""

import numpy as np
import pandas as pd
import torch
from src.urbanstyle.task_b_forecasting.dataset import build_windows, prepare_series
from src.urbanstyle.task_b_forecasting.lstm_model import GlobalDemandLSTM


def sample_weekly() -> pd.DataFrame:
    weeks = pd.date_range("2023-01-01", periods=20, freq="W-SUN")
    rows = []
    for category_index, category in enumerate(("Beauty", "Clothing", "Electronics")):
        for gender_index, gender in enumerate(("Female", "Male")):
            for index, week in enumerate(weeks):
                rows.append(
                    {
                        "week": week,
                        "Product Category": category,
                        "Gender": gender,
                        "sales": float(index + 10 * category_index + gender_index),
                        "units": float(index % 5 + gender_index),
                    }
                )
    return pd.DataFrame(rows)


def test_scaler_is_fitted_only_on_training_period() -> None:
    frame = sample_weekly()
    split = prepare_series(
        frame,
        categories=["Beauty", "Clothing", "Electronics"],
        genders=["Female", "Male"],
        lookback=4,
    )

    np.testing.assert_allclose(split.means[:, 0], split.values[:, : split.train_end].mean(axis=1))
    assert np.all(split.values[:, split.train_end :] != 999)
    assert split.train_end < split.validation_end < len(split.weeks)

    changed_future = frame.copy()
    future_week = sorted(frame["week"].unique())[int(len(frame["week"].unique()) * 0.8)]
    changed_future.loc[changed_future["week"] >= future_week, "units"] += 100_000
    changed_split = prepare_series(
        changed_future,
        categories=["Beauty", "Clothing", "Electronics"],
        genders=["Female", "Male"],
        lookback=4,
    )
    np.testing.assert_allclose(split.means, changed_split.means)
    np.testing.assert_allclose(split.scales, changed_split.scales)


def test_validation_windows_only_score_validation_targets() -> None:
    split = prepare_series(
        sample_weekly(),
        categories=["Beauty", "Clothing", "Electronics"],
        genders=["Female", "Male"],
        lookback=4,
    )
    windows, series_ids, targets, indices = build_windows(
        split, split.train_end, split.validation_end
    )

    assert windows.shape == (6 * (split.validation_end - split.train_end), 4)
    assert series_ids.min() == 0 and series_ids.max() == 5
    assert targets.shape == indices.shape
    assert indices.min() == split.train_end
    assert indices.max() < split.validation_end


def test_lstm_returns_one_prediction_per_sample() -> None:
    model = GlobalDemandLSTM(num_series=6, embedding_size=4, hidden_size=12)
    predictions = model(torch.zeros((5, 8)), torch.tensor([0, 1, 2, 3, 4]))

    assert predictions.shape == (5,)
    assert torch.isfinite(predictions).all()
