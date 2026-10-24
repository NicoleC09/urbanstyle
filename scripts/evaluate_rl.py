"""CLI para evaluar agentes entrenados frente a políticas estáticas."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.task_c_pricing.evaluate import evaluate_from_project
from src.urbanstyle.utils.logging import configure_logging


def main() -> int:
    """Procesa argumentos y guarda la comparativa de margen e ingreso."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, help="Episodios de evaluación; por defecto 100")
    parser.add_argument("--seed", type=int, help="Semilla para condiciones comparables")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes debe ser positivo.")
    configure_logging()
    try:
        artifacts = evaluate_from_project(episodes=args.episodes, seed=args.seed)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        logging.getLogger("urbanstyle.evaluate_rl").error("No se pudo evaluar: %s", error)
        return 1
    for name, path in artifacts.items():
        logging.getLogger("urbanstyle.evaluate_rl").info("%s: %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
