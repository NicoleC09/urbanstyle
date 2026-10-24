"""CLI para generar una galería o póster usando un generador ya entrenado."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.task_d_gan.generate import (
    FashionGenerator,
    create_poster,
    generator_checkpoint,
    save_generated_grid,
)
from src.urbanstyle.task_d_gan.train import FASHION_CLASSES
from src.urbanstyle.utils.config import load_config, resolve_path
from src.urbanstyle.utils.logging import configure_logging


def main() -> int:
    """Genera imágenes deterministas y opcionalmente compone un póster."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=("dcgan", "cgan"), default="cgan")
    parser.add_argument("--class-id", type=int, default=7, choices=range(10))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--discount", type=int, default=20)
    parser.add_argument("--poster", action="store_true")
    parser.add_argument("--output", type=Path, help="Ruta destino PNG")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count debe ser positivo.")
    configure_logging()
    config = load_config(PROJECT_ROOT / "config.yaml")
    try:
        generator = FashionGenerator(generator_checkpoint(config, args.type))
        images = generator.generate(args.class_id, args.count, args.seed)
        output_path = (
            args.output
            if args.output
            else resolve_path(config["paths"]["outputs"])
            / f"gan/generated_{args.type}_{args.class_id}_{args.seed}.png"
        )
        if args.poster:
            destination = create_poster(
                images,
                category=FASHION_CLASSES[args.class_id],
                discount=args.discount,
                destination=output_path,
            )
        else:
            destination = save_generated_grid(images, output_path)
    except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as error:
        logging.getLogger("urbanstyle.generate_poster").error(
            "No se pudo generar la imagen: %s", error
        )
        return 1
    logging.getLogger("urbanstyle.generate_poster").info("Imagen guardada en %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
