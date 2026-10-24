"""CLI para entrenar DCGAN/cGAN sin bloquear la aplicación Pygame."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.task_d_gan.train import train_from_project
from src.urbanstyle.utils.logging import configure_logging


def main() -> int:
    """Ejecuta uno o ambos entrenamientos desde la línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("dcgan", "cgan", "both"), default="cgan")
    parser.add_argument("--epochs", type=int, help="Sobrescribe épocas de config.yaml")
    parser.add_argument("--samples", type=int, help="Limita imágenes de entrenamiento")
    parser.add_argument("--seed", type=int, help="Sobrescribe la semilla global")
    parser.add_argument("--loss", choices=("bce", "hinge"), default="bce")
    parser.add_argument("--spectral-norm", action="store_true")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs debe ser positivo.")
    if args.samples is not None and args.samples < 2:
        parser.error("--samples debe ser al menos 2.")
    configure_logging()
    kinds = ("dcgan", "cgan") if args.type == "both" else (args.type,)
    try:
        for gan_type in kinds:
            artifacts = train_from_project(
                gan_type=gan_type,
                epochs=args.epochs,
                max_train_samples=args.samples,
                seed=args.seed,
                loss_type=args.loss,
                spectral_norm=args.spectral_norm,
            )
            for name, path in artifacts.items():
                logging.getLogger("urbanstyle.train_gan").info("%s %s: %s", gan_type, name, path)
    except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as error:
        logging.getLogger("urbanstyle.train_gan").error("No se pudo entrenar GAN: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
