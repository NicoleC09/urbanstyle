"""Entrenamiento independiente de PPO y DQN para precios dinámicos."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.urbanstyle.task_c_pricing.data import (
    load_forecast_units,
    load_product_data,
    make_environment,
)
from src.urbanstyle.utils.config import PROJECT_ROOT, load_config, resolve_path
from src.urbanstyle.utils.seed import seed_everything
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

LOGGER = logging.getLogger("urbanstyle.task_c.train")


def _build_agent(algorithm: str, environment, config: dict, seed: int, device: str):
    task = config["task_c"]
    common: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": environment,
        "learning_rate": float(task["learning_rate"]),
        "gamma": float(task["gamma"]),
        "seed": seed,
        "device": device,
        "verbose": 0,
    }
    if algorithm == "PPO":
        return PPO(
            **common,
            n_steps=int(task["ppo_n_steps"]),
            batch_size=int(task["ppo_batch_size"]),
        )
    if algorithm == "DQN":
        return DQN(
            **common,
            buffer_size=int(task["dqn_buffer_size"]),
            learning_starts=int(task["dqn_learning_starts"]),
            batch_size=int(task["dqn_batch_size"]),
            train_freq=int(task["dqn_train_freq"]),
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
        )
    raise ValueError(f"Algoritmo no soportado: {algorithm}")


def train_agents(
    config: dict,
    *,
    algorithms: tuple[str, ...] = ("PPO", "DQN"),
    total_timesteps: int | None = None,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, Path]:
    """Entrena los agentes para maximizar margen y guarda modelos/curvas."""
    products = load_product_data(config)
    task = config["task_c"]
    seed = int(config["project"]["seed"] if seed is None else seed)
    total_timesteps = int(task["total_timesteps"] if total_timesteps is None else total_timesteps)
    device = device or str(task.get("device", "auto"))
    model_dir = resolve_path(config["paths"]["models"])
    output_dir = resolve_path(config["paths"]["outputs"])
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    for index, algorithm in enumerate(algorithms):
        run_seed = seed + index
        seed_everything(run_seed)
        environment = Monitor(
            make_environment(config, products, forecast_units=load_forecast_units(config))
        )
        agent = _build_agent(algorithm, environment, config, run_seed, device)
        LOGGER.info(
            "Entrenando %s durante %d pasos, seed=%d, device=%s",
            algorithm,
            total_timesteps,
            run_seed,
            device,
        )
        agent.learn(total_timesteps=total_timesteps, progress_bar=False)
        model_path = model_dir / algorithm.lower()
        agent.save(str(model_path))
        artifacts[algorithm] = model_path.with_suffix(".zip")
        rewards = [float(value) for value in environment.get_episode_rewards()]
        curve_path = output_dir / f"{algorithm.lower()}_learning_curve.png"
        figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        if rewards:
            axis.plot(
                np.arange(1, len(rewards) + 1),
                rewards,
                color="#B8F36B",
                alpha=0.35,
                label="Episodio",
            )
            window = min(10, len(rewards))
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
                axis.plot(
                    np.arange(window, len(rewards) + 1),
                    smoothed,
                    color="#F28C78",
                    label=f"Media móvil ({window})",
                )
            axis.legend(frameon=False)
        axis.set(
            title=f"{algorithm} · margen por episodio",
            xlabel="Episodio",
            ylabel="Margen acumulado",
        )
        axis.grid(alpha=0.2)
        figure.savefig(curve_path, dpi=150)
        plt.close(figure)
        summary_path = output_dir / f"{algorithm.lower()}_training.json"
        summary_path.write_text(
            json.dumps(
                {
                    "algorithm": algorithm,
                    "seed": run_seed,
                    "total_timesteps": total_timesteps,
                    "episodes": len(rewards),
                    "mean_last_10_rewards": float(np.mean(rewards[-10:])) if rewards else None,
                    "objective": "gross_margin",
                    "unit_cost_fraction": float(config["pricing"]["unit_cost_fraction"]),
                    "device": device,
                    "elasticity_methods": {
                        product.name: product.elasticity_method for product in products
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        artifacts[f"{algorithm}_curve"] = curve_path
        artifacts[f"{algorithm}_summary"] = summary_path
        environment.close()
    return artifacts


def train_from_project(
    *,
    algorithms: tuple[str, ...] = ("PPO", "DQN"),
    total_timesteps: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    """Carga la configuración del proyecto y entrena agentes."""
    return train_agents(
        load_config(config_path or PROJECT_ROOT / "config.yaml"),
        algorithms=algorithms,
        total_timesteps=total_timesteps,
        seed=seed,
        device=device,
    )
