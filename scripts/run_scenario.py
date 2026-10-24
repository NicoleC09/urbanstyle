"""Ejecuta el escenario integrado UrbanStyle sin abrir la interfaz Pygame."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.integration.pipeline import run_from_project
from src.urbanstyle.utils.logging import configure_logging


def main() -> int:
    """Ejecuta B→C→A→D con semilla y agente opcionales."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Sobrescribe la semilla del escenario")
    parser.add_argument("--algorithm", choices=("PPO", "DQN"), help="Agente de precios")
    args = parser.parse_args()
    configure_logging()
    try:
        result = run_from_project(seed=args.seed, pricing_algorithm=args.algorithm)
    except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as error:
        logging.getLogger("urbanstyle.run_scenario").error(
            "No se pudo ejecutar el escenario: %s", error
        )
        return 1
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
