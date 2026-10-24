"""Ejecuta las evaluaciones B y C sobre checkpoints existentes."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.task_b_forecasting.train import evaluate_saved_forecaster
from src.urbanstyle.task_c_pricing.evaluate import evaluate_from_project
from src.urbanstyle.utils.config import load_config
from src.urbanstyle.utils.logging import configure_logging


def main() -> int:
    """Evalúa modelos ya entrenados; no ejecuta entrenamiento."""
    configure_logging()
    logger = logging.getLogger("urbanstyle.evaluate_all")
    config = load_config(PROJECT_ROOT / "config.yaml")
    try:
        for name, path in evaluate_saved_forecaster(config).items():
            logger.info("Task B %s: %s", name, path)
        for name, path in evaluate_from_project().items():
            logger.info("Task C %s: %s", name, path)
    except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as error:
        logger.error("No se pudieron completar las evaluaciones: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
