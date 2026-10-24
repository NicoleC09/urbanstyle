"""CLI para entrenar PPO y/o DQN fuera de la aplicación gráfica."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.task_c_pricing.train import train_from_project
from src.urbanstyle.utils.logging import configure_logging


def main() -> int:
    """Procesa argumentos y ejecuta entrenamiento standalone."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("PPO", "DQN", "both"), default="both")
    parser.add_argument(
        "--timesteps", type=int, help="Sobrescribe pasos de entrenamiento configurados"
    )
    parser.add_argument("--seed", type=int, help="Sobrescribe la semilla global")
    parser.add_argument("--device", help="Dispositivo SB3: auto, cpu, cuda o mps")
    args = parser.parse_args()
    if args.timesteps is not None and args.timesteps < 1:
        parser.error("--timesteps debe ser positivo.")
    configure_logging()
    algorithms = ("PPO", "DQN") if args.algorithm == "both" else (args.algorithm,)
    try:
        artifacts = train_from_project(
            algorithms=algorithms,
            total_timesteps=args.timesteps,
            seed=args.seed,
            device=args.device,
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as error:
        logging.getLogger("urbanstyle.train_rl").error("No se pudo entrenar: %s", error)
        return 1
    for name, path in artifacts.items():
        logging.getLogger("urbanstyle.train_rl").info("%s: %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
