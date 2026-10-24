"""Pruebas iniciales de validación y agregación del dataset."""

import pandas as pd
import pytest
from scripts.prepare_data import (
    DataValidationError,
    build_weekly_series,
    drop_partial_edge_weeks,
    validate_dataset,
)

REQUIRED_COLUMNS = [
    "Transaction ID",
    "Date",
    "Customer ID",
    "Gender",
    "Age",
    "Product Category",
    "Quantity",
    "Price per Unit",
    "Total Amount",
]


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1, "2023-01-02", "C1", "Female", 25, "Beauty", 2, 10, 20],
            [2, "2023-01-09", "C2", "Male", 35, "Clothing", 1, 30, 30],
        ],
        columns=REQUIRED_COLUMNS,
    )


def test_validation_converts_dates_and_numeric_fields(sample_transactions: pd.DataFrame) -> None:
    clean = validate_dataset(sample_transactions, REQUIRED_COLUMNS)

    assert pd.api.types.is_datetime64_any_dtype(clean["Date"])
    assert pd.api.types.is_numeric_dtype(clean["Quantity"])
    assert len(clean) == 2


def test_validation_rejects_missing_columns(sample_transactions: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="Faltan columnas"):
        validate_dataset(sample_transactions.drop(columns=["Gender"]), REQUIRED_COLUMNS)


def test_weekly_series_contains_six_combinations_and_zero_fills(
    sample_transactions: pd.DataFrame,
) -> None:
    clean = validate_dataset(sample_transactions, REQUIRED_COLUMNS)
    weekly = build_weekly_series(
        clean,
        categories=["Beauty", "Clothing", "Electronics"],
        genders=["Female", "Male"],
        frequency="W-SUN",
    )

    assert len(weekly) == 2 * 3 * 2
    empty_series = weekly.loc[
        (weekly["Product Category"] == "Electronics") & (weekly["Gender"] == "Female")
    ]
    assert empty_series["sales"].eq(0).all()
    assert empty_series["units"].eq(0).all()


def test_partial_calendar_edge_weeks_are_discarded_by_date_coverage() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                    "2023-01-07",
                    "2023-01-08",
                    "2023-12-31",
                    "2024-01-01",
                ]
            ),
            "units": range(10),
        }
    )

    complete, excluded = drop_partial_edge_weeks(frame, "W-SUN")

    assert complete["Date"].min() == pd.Timestamp("2023-01-02")
    assert complete["Date"].max() == pd.Timestamp("2023-12-31")
    assert excluded == ["2023-01-01", "2024-01-07"]


def test_single_partial_week_is_removed_completely() -> None:
    frame = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-03"])})

    complete, excluded = drop_partial_edge_weeks(frame, "W-SUN")

    assert complete.empty
    assert excluded == ["2024-01-07"]
