"""Pruebas del entorno Gymnasium de precios dinámicos."""

import numpy as np
import pandas as pd
import pytest
from src.urbanstyle.task_c_pricing.data import load_forecast_units, mean_weekly_units
from src.urbanstyle.task_c_pricing.demand_model import expected_demand
from src.urbanstyle.task_c_pricing.env import DynamicPricingEnv, ProductPricingData
from stable_baselines3.common.env_checker import check_env


@pytest.fixture
def environment() -> DynamicPricingEnv:
    return DynamicPricingEnv(
        [ProductPricingData("Clothing", 50.0, 8.0, -1.2, "fallback")],
        episode_steps=5,
        price_step_fraction=0.1,
        price_min_multiplier=0.6,
        price_max_multiplier=1.5,
        demand_noise_std=0.0,
    )


def test_environment_passes_sb3_gymnasium_checker(environment: DynamicPricingEnv) -> None:
    check_env(environment, warn=True)


def test_price_is_clamped_and_episode_truncates(environment: DynamicPricingEnv) -> None:
    environment.reset(seed=4, options={"product": "Clothing", "segment": 1})
    for _ in range(20):
        observation, reward, terminated, truncated, info = environment.step(2)

    assert info["price_index"] == pytest.approx(1.5)
    assert 0 <= observation[0] <= 1
    assert 0 < observation[2] < 1
    assert not terminated
    assert truncated
    assert reward == pytest.approx(info["profit"])
    assert info["profit"] == pytest.approx(
        info["units_sold"]
        * (info["price"] - environment.unit_cost_fraction * environment.product.reference_price)
    )


def test_price_is_clamped_at_lower_bound(environment: DynamicPricingEnv) -> None:
    environment.reset(seed=4, options={"product": "Clothing", "segment": 1})
    for _ in range(20):
        _, _, _, _, info = environment.step(0)

    assert info["price_index"] == pytest.approx(0.6)


def test_reset_clamps_initial_price_option(environment: DynamicPricingEnv) -> None:
    _, info = environment.reset(
        seed=3,
        options={"product": "Clothing", "segment": 1, "initial_price": 9.0},
    )

    assert info["price_index"] == pytest.approx(1.5)


def test_expected_demand_decreases_as_price_increases() -> None:
    quantities = [expected_demand(price, 100, 12, -1.2) for price in (60, 100, 150)]

    assert quantities[0] > quantities[1] > quantities[2]


def test_profit_reward_has_an_interior_theoretical_price() -> None:
    from src.urbanstyle.task_c_pricing.policies import (
        theoretical_optimal_price,
        theoretical_price_multiplier,
        theoretical_price_policy,
    )

    product = ProductPricingData("Clothing", 50.0, 8.0, -1.2)
    env = DynamicPricingEnv(
        [product],
        price_min_multiplier=0.6,
        price_max_multiplier=4.0,
        unit_cost_fraction=0.55,
        demand_noise_std=0,
    )
    env.reset(seed=2, options={"product": "Clothing", "segment": 1})
    target = product.elasticity / (product.elasticity + 1) * env.unit_cost_fraction
    optimum = theoretical_price_multiplier(
        product.elasticity,
        env.unit_cost_fraction,
        env.price_min_multiplier,
        env.price_max_multiplier,
    )

    assert 0.6 < target < 4.0
    assert optimum == pytest.approx(target)
    assert theoretical_optimal_price(env) == pytest.approx(target)
    assert theoretical_price_policy(np.zeros(env.observation_space.shape), env) == 2
    demand = np.asarray(
        [
            expected_demand(price, 1.0, 10.0, product.elasticity)
            for price in np.linspace(0.6, 4, 200)
        ]
    )
    multipliers = np.linspace(0.6, 4, 200)
    profits = (multipliers - env.unit_cost_fraction) * demand
    maximum_index = int(profits.argmax())
    assert 0 < maximum_index < len(profits) - 1
    assert multipliers[maximum_index] == pytest.approx(target, abs=0.02)


def test_low_price_is_degenerate_for_revenue_but_not_for_margin() -> None:
    elasticity = -1.2
    unit_cost_fraction = 0.55
    prices = np.asarray([0.6, 1.0, 1.5, 3.3, 4.0])
    demand = np.asarray([expected_demand(price, 1.0, 100.0, elasticity) for price in prices])
    revenue = prices * demand
    margin = (prices - unit_cost_fraction) * demand

    assert revenue[0] > revenue[1] > revenue[2]
    assert int(margin.argmax()) == 3


def test_baseline_units_sums_gender_series_per_week() -> None:
    weekly = pd.DataFrame(
        {
            "week": ["w1", "w1", "w2", "w2"],
            "Product Category": ["Clothing"] * 4,
            "Gender": ["Female", "Male"] * 2,
            "units": [3, 5, 4, 6],
        }
    )

    assert mean_weekly_units(weekly, "Clothing") == pytest.approx(9.0)


def test_forecast_units_sum_gender_predictions(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    pd.DataFrame(
        {
            "category": ["Clothing", "Clothing", "Beauty"],
            "gender": ["Female", "Male", "Female"],
            "lstm": [7.5, 6.0, 3.0],
        }
    ).to_csv(outputs / "predictions.csv", index=False)
    config = {
        "paths": {"outputs": str(outputs)},
        "data": {"categories": ["Beauty", "Clothing", "Electronics"]},
    }

    assert load_forecast_units(config) == {"Beauty": 3.0, "Clothing": 13.5}


def test_environment_is_reproducible_with_same_seed() -> None:
    first = DynamicPricingEnv([ProductPricingData("Beauty", 50, 10, -1.2)])
    second = DynamicPricingEnv([ProductPricingData("Beauty", 50, 10, -1.2)])
    first.reset(seed=123)
    second.reset(seed=123)

    transitions_first = [first.step(2) for _ in range(4)]
    transitions_second = [second.step(2) for _ in range(4)]

    for left, right in zip(transitions_first, transitions_second, strict=True):
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]
