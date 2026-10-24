"""Compara algoritmos de ruta y layouts con simulaciones repetidas."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.urbanstyle.task_a_pathfinding.layouts import available_layouts
from src.urbanstyle.task_a_pathfinding.simulation import simulate_customers
from src.urbanstyle.utils.config import load_config, resolve_path
from src.urbanstyle.utils.logging import configure_logging

LOGGER = logging.getLogger("urbanstyle.layout_experiment")


def load_category_proportions(config: dict) -> dict[str, float]:
    """Lee proporciones preparadas o calcula el reparto desde el CSV original."""
    calibration_path = resolve_path(config["paths"]["processed_data"]) / "calibration.json"
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        proportions = calibration.get("category_sales_share")
        if isinstance(proportions, dict) and proportions:
            return {
                category: float(proportions.get(category, 0.0))
                for category in config["data"]["categories"]
            }

    raw_path = resolve_path(config["paths"]["raw_data"])
    if not raw_path.is_file():
        raise FileNotFoundError(
            "No se encontró el CSV para ponderar las visitas. Colócalo en "
            f"{raw_path} y ejecuta python scripts/prepare_data.py."
        )
    frame = pd.read_csv(raw_path)
    required = {"Product Category", "Total Amount"}
    if not required.issubset(frame.columns):
        raise ValueError("El CSV necesita las columnas Product Category y Total Amount.")
    totals = frame.groupby("Product Category")["Total Amount"].sum()
    grand_total = float(totals.sum())
    if grand_total <= 0:
        raise ValueError("El CSV no contiene ventas positivas para ponderar las visitas.")
    return {
        category: float(totals.get(category, 0.0) / grand_total)
        for category in config["data"]["categories"]
    }


def run_experiment(runs: int, customers: int, seed: int = 42) -> pd.DataFrame:
    """Ejecuta N simulaciones por algoritmo/layout y agrega métricas."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    proportions = load_category_proportions(config)
    task_config = config["task_a"]
    layouts = available_layouts(task_config["grid_width"], task_config["grid_height"])
    records: list[dict[str, float | int | str]] = []
    for layout_index, layout in enumerate(layouts):
        for algorithm in ("BFS", "A*"):
            results = [
                simulate_customers(
                    layout,
                    algorithm,
                    customers,
                    proportions,
                    seed=seed + run_index + layout_index * runs,
                )
                for run_index in range(runs)
            ]
            records.append(
                {
                    "layout": layout.name,
                    "algorithm": algorithm,
                    "runs": runs,
                    "customers_per_run": customers,
                    "max_congestion_mean": sum(result.max_congestion for result in results) / runs,
                    "mean_path_length": sum(result.mean_path_length for result in results) / runs,
                    "dwell_steps_mean": sum(result.dwell_steps for result in results) / runs,
                    "search_time_ms_mean": sum(result.mean_search_ms for result in results) / runs,
                    "explored_nodes_mean": sum(result.explored_nodes for result in results) / runs,
                }
            )
    return pd.DataFrame.from_records(records)


def save_chart(summary: pd.DataFrame, output_path: Path) -> None:
    """Guarda una figura comparando congestión y permanencia por layout."""
    layouts = list(summary["layout"].drop_duplicates())
    algorithms = ("BFS", "A*")
    positions = list(range(len(layouts)))
    bar_width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {"BFS": "#F28C78", "A*": "#B8F36B"}
    for index, algorithm in enumerate(algorithms):
        subset = summary.loc[summary["algorithm"] == algorithm].set_index("layout").loc[layouts]
        offsets = [position + (index - 0.5) * bar_width for position in positions]
        axes[0].bar(
            offsets,
            subset["max_congestion_mean"],
            bar_width,
            label=algorithm,
            color=colors[algorithm],
        )
        axes[1].bar(
            offsets, subset["dwell_steps_mean"], bar_width, label=algorithm, color=colors[algorithm]
        )
    axes[0].set_title("Congestión máxima media")
    axes[0].set_ylabel("Visitas en la celda más transitada")
    axes[1].set_title("Permanencia media")
    axes[1].set_ylabel("Pasos por cliente")
    for axis in axes:
        axis.set_xticks(positions, layouts)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    figure.suptitle("UrbanStyle · Comparativa de distribución")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    """Ejecuta y exporta el experimento configurado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, help="Simulaciones por layout y algoritmo")
    parser.add_argument("--customers", type=int, default=24, help="Clientes por simulación")
    parser.add_argument("--seed", type=int, help="Semilla inicial")
    args = parser.parse_args()
    configure_logging()
    config = load_config(PROJECT_ROOT / "config.yaml")
    runs = args.runs or int(config["task_a"]["experiment_runs"])
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    if runs < 1 or args.customers < 1:
        parser.error("--runs y --customers deben ser mayores que cero.")

    try:
        summary = run_experiment(runs, args.customers, seed)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        LOGGER.error("No se pudo ejecutar el experimento: %s", error)
        return 1

    output_dir = resolve_path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "layout_experiment.csv"
    chart_path = output_dir / "layout_experiment.png"
    summary.to_csv(csv_path, index=False)
    save_chart(summary, chart_path)
    LOGGER.info("Tabla guardada en %s", csv_path)
    LOGGER.info("Gráfico guardado en %s", chart_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
