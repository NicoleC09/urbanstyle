"""CLI para reevaluar el checkpoint LSTM existente sin entrenarlo."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.task_b_forecasting.train import evaluate_saved_forecaster
from src.urbanstyle.utils.config import load_config
from src.urbanstyle.utils.logging import configure_logging


def main() -> int:
    """Evalúa el modelo guardado con el split test cronológico."""
    configure_logging()
    try:
        artifacts = evaluate_saved_forecaster(load_config(PROJECT_ROOT / "config.yaml"))
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        logging.getLogger("urbanstyle.evaluate_lstm").error("No se pudo evaluar: %s", error)
        return 1
    for name, path in artifacts.items():
        logging.getLogger("urbanstyle.evaluate_lstm").info("%s: %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
