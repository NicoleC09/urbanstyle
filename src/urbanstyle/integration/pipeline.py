"""Escenario punta a punta UrbanStyle: pronóstico, precio, tráfico y campaña."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from src.urbanstyle.task_a_pathfinding.layouts import create_layout
from src.urbanstyle.task_a_pathfinding.simulation import simulate_customers
from src.urbanstyle.task_c_pricing.data import load_product_data, make_environment
from src.urbanstyle.task_d_gan.generate import (
    FashionGenerator,
    generator_checkpoint,
    load_fashion_classifier,
    save_generated_grid,
)
from src.urbanstyle.task_d_gan.poster import agent_offer_copy, create_campaign_poster
from src.urbanstyle.task_d_gan.train import FASHION_CLASSES
from src.urbanstyle.utils.config import PROJECT_ROOT, load_config, resolve_path
from stable_baselines3 import DQN, PPO

LOGGER = logging.getLogger("urbanstyle.integration")
AGENT_TYPES = {"PPO": PPO, "DQN": DQN}


@dataclass(frozen=True)
class ScenarioResult:
    """Resultados resumidos y rutas de artefactos del escenario integrado."""

    category: str
    forecast_units: float
    pricing_algorithm: str
    action: str
    price: float
    units_sold: int
    revenue: float
    discount: int
    layout: str
    pathfinding_algorithm: str
    customers: int
    max_congestion: int
    congestion_cell: tuple[int, int]
    dwell_steps: float
    fashion_class: str
    proxy_note: str
    poster_path: str
    generated_grid_path: str
    seed: int
    traffic_heatmap: tuple[tuple[int, int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convierte el resultado en un objeto JSON serializable."""
        result = asdict(self)
        result["congestion_cell"] = list(self.congestion_cell)
        return result


def select_demand_category(forecasts: dict[str, float]) -> tuple[str, float]:
    """Elige la categoría con mayor pronóstico, desempate alfabético estable."""
    if not forecasts:
        raise ValueError("No hay pronósticos cargados para ejecutar el escenario.")
    category = min(forecasts, key=lambda name: (-float(forecasts[name]), name))
    return category, float(forecasts[category])


def suggested_discount(price_index: float, minimum: int = 5, maximum: int = 50) -> int:
    """Deriva un descuento promocional acotado a partir del precio del agente."""
    discount = round((1.0 - float(price_index)) * 100)
    return min(maximum, max(minimum, discount))


def fashion_proxy_class(category: str, config: dict) -> int:
    """Obtiene clase Fashion-MNIST proxy para una categoría comercial."""
    mapping = config["integration"]["gan_class_by_category"]
    if category not in mapping:
        raise ValueError(f"No hay clase visual configurada para {category}.")
    class_id = int(mapping[category])
    if not 0 <= class_id < len(FASHION_CLASSES):
        raise ValueError(f"La clase Fashion-MNIST de {category} no es válida: {class_id}.")
    return class_id


def _load_forecasts(config: dict) -> dict[str, float]:
    """Carga pronósticos LSTM; el escenario no oculta la ausencia de Task B."""
    path = resolve_path(config["paths"]["outputs"]) / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta {path}; ejecuta python scripts/train_lstm.py antes del escenario."
        )
    frame = pd.read_csv(path)
    required = {"category", "gender", "lstm"}
    if not required.issubset(frame.columns):
        raise ValueError("predictions.csv debe incluir category, gender y lstm.")
    forecast = frame.groupby("category")["lstm"].sum().to_dict()
    return {category: float(value) for category, value in forecast.items()}


def _load_sales_shares(config: dict) -> dict[str, float]:
    """Carga proporciones de venta reales generadas por prepare_data.py."""
    path = resolve_path(config["paths"]["processed_data"]) / "calibration.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta {path}; ejecuta python scripts/prepare_data.py antes del escenario."
        )
    calibration = json.loads(path.read_text(encoding="utf-8"))
    shares = calibration.get("category_sales_share", {})
    if not shares:
        raise ValueError("calibration.json no contiene proporciones de venta por categoría.")
    return {category: float(shares.get(category, 0.0)) for category in config["data"]["categories"]}


def _agent_action(
    config: dict,
    category: str,
    forecast_units: float,
    algorithm: str,
    seed: int,
) -> tuple[dict[str, Any], float]:
    """Carga agente entrenado, decide una acción y simula su primer paso."""
    if algorithm not in AGENT_TYPES:
        raise ValueError(f"Agente RL desconocido: {algorithm}")
    model_path = resolve_path(config["paths"]["models"]) / algorithm.lower()
    if not model_path.with_suffix(".zip").is_file():
        raise FileNotFoundError(
            f"Falta {model_path.with_suffix('.zip')}; ejecuta python scripts/train_rl.py "
            f"--algorithm {algorithm}."
        )
    products = load_product_data(config)
    product = next((item for item in products if item.name == category), None)
    if product is None:
        raise ValueError(f"No hay calibración de precios para {category}.")
    environment = make_environment(config, products, forecast_units={category: forecast_units})
    segment = int(config["integration"]["segment_index"])
    observation, _ = environment.reset(
        seed=seed,
        options={"product": category, "segment": segment, "initial_price": 1.0},
    )
    model = AGENT_TYPES[algorithm].load(str(model_path), device="cpu")
    action_array, _ = model.predict(observation, deterministic=True)
    action_id = int(action_array)
    _, _, _, _, info = environment.step(action_id)
    action_name = ("Bajar", "Mantener", "Subir")[action_id]
    environment.close()
    return info | {"action_name": action_name}, info["price_index"]


def run_scenario(
    config: dict,
    *,
    seed: int | None = None,
    pricing_algorithm: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> ScenarioResult:
    """Ejecuta las etapas B→C→A→D y guarda el resumen del escenario."""
    seed = int(config["project"]["seed"] if seed is None else seed)
    integration = config["integration"]
    pricing_algorithm = pricing_algorithm or str(integration["pricing_algorithm"])

    forecasts = _load_forecasts(config)
    category, forecast_units = select_demand_category(forecasts)
    if progress_callback:
        progress_callback(1)
    pricing_info, price_index = _agent_action(
        config, category, forecast_units, pricing_algorithm, seed
    )
    discount = suggested_discount(price_index)
    if progress_callback:
        progress_callback(2)

    shares = _load_sales_shares(config)
    layout = create_layout(
        str(integration["layout"]),
        int(config["task_a"]["grid_width"]),
        int(config["task_a"]["grid_height"]),
    )
    customers = int(integration["customers"])
    pathfinding_algorithm = str(integration["pathfinding_algorithm"])
    traffic_result = simulate_customers(
        layout,
        pathfinding_algorithm,
        customers,
        shares,
        seed=seed,
    )
    congestion_cell, max_congestion = traffic_result.top_congested[0]
    if progress_callback:
        progress_callback(3)

    class_id = fashion_proxy_class(category, config)
    generator = FashionGenerator(generator_checkpoint(config, "cgan"), device="cpu")
    images = generator.generate(class_id, count=4, seed=seed)
    output_dir = resolve_path(config["paths"]["outputs"]) / "integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = save_generated_grid(images, output_dir / "scenario_campaign.png", columns=2)
    classifier_path = resolve_path(config["paths"]["models"]) / "fmnist_classifier.pt"
    classifier = load_fashion_classifier(classifier_path) if classifier_path.is_file() else None
    candidates = generator.generate(
        class_id,
        count=64,
        seed=seed,
        truncation=float(config["task_d"].get("truncation", 0.8)),
    )
    offer_copy, offer_discount = agent_offer_copy(
        int(pricing_info["action"]),
        float(pricing_info["price"]) / max(price_index, 1e-9),
        float(pricing_info["price"]),
    )
    discount = offer_discount
    poster_path = create_campaign_poster(
        candidates,
        product_name=FASHION_CLASSES[class_id],
        class_id=class_id,
        destination=output_dir / "scenario_poster.png",
        classifier=classifier,
        hero_image=images[0],
        template="Medianoche",
        style="Neón",
        discount_percent=offer_discount,
        offer_copy=offer_copy,
        seed=seed,
    )
    if progress_callback:
        progress_callback(4)
    proxy_note = (
        f"Imagen ilustrativa Fashion-MNIST: {FASHION_CLASSES[class_id]}; "
        "no representa una fotografía real del producto."
    )

    result = ScenarioResult(
        category=category,
        forecast_units=forecast_units,
        pricing_algorithm=pricing_algorithm,
        action=str(pricing_info["action_name"]),
        price=float(pricing_info["price"]),
        units_sold=int(pricing_info["units_sold"]),
        revenue=float(pricing_info["revenue"]),
        discount=discount,
        layout=layout.name,
        pathfinding_algorithm=pathfinding_algorithm,
        customers=customers,
        max_congestion=max_congestion,
        congestion_cell=congestion_cell,
        dwell_steps=float(traffic_result.dwell_steps),
        fashion_class=FASHION_CLASSES[class_id],
        proxy_note=proxy_note,
        poster_path=str(poster_path),
        generated_grid_path=str(grid_path),
        seed=seed,
        traffic_heatmap=tuple(
            (position[0], position[1], int(value))
            for position, value in sorted(traffic_result.traffic.items())
        ),
    )
    summary_path = output_dir / "scenario_summary.json"
    summary_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Escenario integrado guardado en %s", summary_path)
    return result


def run_from_project(
    *,
    seed: int | None = None,
    pricing_algorithm: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> ScenarioResult:
    """Carga la configuración del proyecto y ejecuta el escenario completo."""
    return run_scenario(
        load_config(PROJECT_ROOT / "config.yaml"),
        seed=seed,
        pricing_algorithm=pricing_algorithm,
        progress_callback=progress_callback,
    )
