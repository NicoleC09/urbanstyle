"""Evaluación comparable de agentes RL y políticas estáticas de precios."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.urbanstyle.task_c_pricing.data import (
    load_forecast_units,
    load_product_data,
    make_environment,
)
from src.urbanstyle.task_c_pricing.policies import (
    STATIC_POLICIES,
    theoretical_price_multiplier,
)
from src.urbanstyle.utils.config import PROJECT_ROOT, load_config, resolve_path
from stable_baselines3 import DQN, PPO

LOGGER = logging.getLogger("urbanstyle.task_c.evaluate")
MODEL_TYPES = {"PPO": PPO, "DQN": DQN}


def evaluate_pricing(
    config: dict,
    *,
    episodes: int | None = None,
    seed: int | None = None,
    model_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]]]:
    """Evalúa margen e ingreso con productos, segmentos y semillas compartidos."""
    products = load_product_data(config)
    episodes = int(episodes or config["task_c"]["evaluation_episodes"])
    seed = int(config["project"]["seed"] if seed is None else seed)
    model_dir = model_dir or resolve_path(config["paths"]["models"])
    loaded_models: dict[str, Any] = {}
    for algorithm, model_type in MODEL_TYPES.items():
        path = model_dir / algorithm.lower()
        if path.with_suffix(".zip").is_file():
            loaded_models[algorithm] = model_type.load(str(path), device="cpu")
        else:
            LOGGER.warning("Se omite %s: falta %s", algorithm, path.with_suffix(".zip"))

    policies: dict[str, Any] = dict(STATIC_POLICIES)
    policies.update({f"{name} agente": model for name, model in loaded_models.items()})
    margins = {name: [] for name in policies}
    revenues = {name: [] for name in policies}
    environment = make_environment(config, products, forecast_units=load_forecast_units(config))
    for episode in range(episodes):
        product = products[episode % len(products)]
        segment = episode % len(environment.segment_multipliers)
        episode_seed = seed + episode
        for name, policy in policies.items():
            observation, _ = environment.reset(
                seed=episode_seed,
                options={"product": product.name, "segment": segment},
            )
            total_margin = 0.0
            total_revenue = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                if name.endswith("agente"):
                    action, _ = policy.predict(observation, deterministic=True)
                    action = int(action)
                else:
                    action = int(policy(observation, environment))
                observation, _, terminated, truncated, info = environment.step(action)
                total_margin += float(info["profit"])
                total_revenue += float(info["revenue"])
            margins[name].append(total_margin)
            revenues[name].append(total_revenue)
    environment.close()

    rows = []
    for name in policies:
        rows.append(
            {
                "policy": name,
                "episodes": episodes,
                "mean_margin": float(np.mean(margins[name])),
                "std_margin": float(np.std(margins[name], ddof=1)) if episodes > 1 else 0.0,
                "mean_revenue": float(np.mean(revenues[name])),
                "std_revenue": float(np.std(revenues[name], ddof=1)) if episodes > 1 else 0.0,
                "median_revenue": float(np.median(revenues[name])),
                "min_revenue": float(np.min(revenues[name])),
                "max_revenue": float(np.max(revenues[name])),
            }
        )
    return pd.DataFrame(rows), {"margin": margins, "revenue": revenues}


def save_evaluation(
    summary: pd.DataFrame,
    episode_results: dict[str, dict[str, list[float]]],
    config: dict,
    seed: int,
) -> dict[str, Path]:
    """Escribe margen/ingreso en CSV/JSON y un gráfico de margen con desviación."""
    output_dir = resolve_path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pricing_evaluation.csv"
    json_path = output_dir / "pricing_evaluation.json"
    plot_path = output_dir / "pricing_evaluation.png"
    summary.to_csv(csv_path, index=False)
    payload = {
        "seed": seed,
        "reward_objective": "gross_margin",
        "unit_cost_fraction": float(config["pricing"]["unit_cost_fraction"]),
        "elasticity_assumptions": {
            product.name: {"elasticity": product.elasticity, "method": product.elasticity_method}
            for product in load_product_data(config)
        },
        "theoretical_price_multipliers": {
            product.name: float(
                theoretical_price_multiplier(
                    product.elasticity,
                    config["pricing"]["unit_cost_fraction"],
                    config["pricing"]["price_min_multiplier"],
                    config["pricing"]["price_max_multiplier"],
                )
            )
            for product in load_product_data(config)
        },
        "results": summary.to_dict(orient="records"),
        "episode_results": episode_results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    colors = ["#2F6BFF" if "agente" not in name else "#4ADE9B" for name in summary["policy"]]
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(
        summary["policy"],
        summary["mean_margin"],
        yerr=summary["std_margin"],
        capsize=5,
        color=colors,
    )
    axis.set(
        title="Margen por episodio: políticas de precio",
        ylabel="Margen medio ± desviación",
        xlabel="Política",
    )
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return {"csv": csv_path, "json": json_path, "plot": plot_path}


def evaluate_from_project(
    *, episodes: int | None = None, seed: int | None = None, config_path: str | Path | None = None
) -> dict[str, Path]:
    """Evalúa modelos encontrados y políticas estáticas usando config.yaml."""
    config = load_config(config_path or PROJECT_ROOT / "config.yaml")
    actual_seed = int(config["project"]["seed"] if seed is None else seed)
    summary, episode_results = evaluate_pricing(config, episodes=episodes, seed=actual_seed)
    return save_evaluation(summary, episode_results, config, actual_seed)
