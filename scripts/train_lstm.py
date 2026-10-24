"""CLI para entrenar el LSTM semanal fuera de la aplicación."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.task_b_forecasting.train import train_from_project
from src.urbanstyle.utils.logging import configure_logging


def main() -> int:
    """Entrena el modelo con configuración del proyecto y semilla fija."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, help="Sobrescribe épocas máximas configuradas")
    parser.add_argument("--seed", type=int, help="Sobrescribe la semilla global")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs debe ser positivo.")
    configure_logging()
    try:
        artifacts = train_from_project(max_epochs=args.epochs, seed=args.seed)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        logging.getLogger("urbanstyle.train_lstm").error("No se pudo entrenar: %s", error)
        return 1
    for name, path in artifacts.items():
        logging.getLogger("urbanstyle.train_lstm").info("%s: %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
