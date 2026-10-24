"""Pruebas de las reglas de selección y decisión del escenario integrado."""

from pathlib import Path

import pytest
from app.tabs import home_tab as home_module
from app.tabs.home_tab import HomeTab
from src.urbanstyle.integration.pipeline import (
    ScenarioResult,
    fashion_proxy_class,
    select_demand_category,
    suggested_discount,
)
from src.urbanstyle.utils.config import load_config


def test_scenario_selects_highest_forecast_with_stable_tie_break() -> None:
    assert select_demand_category({"Clothing": 12.0, "Beauty": 14.0}) == ("Beauty", 14.0)
    assert select_demand_category({"Clothing": 14.0, "Beauty": 14.0}) == ("Beauty", 14.0)


@pytest.mark.parametrize(
    ("price_index", "expected"),
    [(0.6, 40), (0.95, 5), (1.0, 5), (1.5, 5)],
)
def test_suggested_discount_is_bounded(price_index: float, expected: int) -> None:
    assert suggested_discount(price_index) == expected


def test_visual_proxy_class_is_configured_and_in_range() -> None:
    config = load_config()

    assert fashion_proxy_class("Clothing", config) == 3
    with pytest.raises(ValueError, match="No hay clase visual"):
        fashion_proxy_class("Unknown", config)


def test_home_runs_scenario_asynchronously_and_updates_summary(monkeypatch, tmp_path) -> None:
    config = load_config()
    poster = tmp_path / "poster.png"
    poster.write_bytes(b"png")
    result = ScenarioResult(
        category="Clothing",
        forecast_units=15.0,
        pricing_algorithm="PPO",
        action="Bajar",
        price=47.5,
        units_sold=16,
        revenue=760.0,
        discount=5,
        layout="Clásico",
        pathfinding_algorithm="A*",
        customers=24,
        max_congestion=24,
        congestion_cell=(0, 3),
        dwell_steps=25.0,
        fashion_class="Vestido",
        proxy_note="Imagen proxy.",
        poster_path=str(poster),
        generated_grid_path=str(tmp_path / "grid.png"),
        seed=42,
        traffic_heatmap=((1, 2, 5), (2, 3, 9)),
    )
    monkeypatch.setattr(home_module, "run_scenario", lambda config, **kwargs: result)
    home = HomeTab(config, Path.cwd())

    home.run_scenario()
    assert home.loading
    assert home._future is not None
    home._future.result(timeout=5)
    home.update()

    assert not home.loading
    assert home.result == result
    assert home.poster_path == poster
    home.close()


def test_scenario_result_serializes_heatmap_for_home_dashboard() -> None:
    result = ScenarioResult(
        category="Clothing",
        forecast_units=15.0,
        pricing_algorithm="PPO",
        action="Mantener",
        price=50.0,
        units_sold=10,
        revenue=500.0,
        discount=0,
        layout="Clásico",
        pathfinding_algorithm="A*",
        customers=24,
        max_congestion=9,
        congestion_cell=(2, 3),
        dwell_steps=25.0,
        fashion_class="Vestido",
        proxy_note="Imagen proxy.",
        poster_path="poster.png",
        generated_grid_path="grid.png",
        seed=42,
        traffic_heatmap=((1, 2, 5), (2, 3, 9)),
    )

    serialized = result.to_dict()

    assert serialized["traffic_heatmap"] == ((1, 2, 5), (2, 3, 9))


def test_home_surfaces_missing_artifact_error(monkeypatch) -> None:
    config = load_config()
    monkeypatch.setattr(
        home_module,
        "run_scenario",
        lambda config, **kwargs: (_ for _ in ()).throw(
            FileNotFoundError("Falta predictions.csv")
        ),
    )
    home = HomeTab(config, Path.cwd())
    home.run_scenario()
    assert home._future is not None
    with pytest.raises(FileNotFoundError):
        home._future.result(timeout=5)
    home.update()

    assert not home.loading
    assert home.error == "Falta predictions.csv"
    home.close()
