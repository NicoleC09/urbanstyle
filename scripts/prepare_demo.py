"""Verifica los artefactos de demo y prepara un póster reproducible."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.urbanstyle.task_d_gan.generate import FashionGenerator, load_fashion_classifier
from src.urbanstyle.task_d_gan.poster import create_campaign_poster, rank_generated_images
from src.urbanstyle.utils.config import load_config, resolve_path
from src.urbanstyle.utils.seed import seed_everything


def prepare_demo() -> tuple[bool, list[tuple[bool, str]]]:
    """Comprueba datos/modelos/reportes y crea el póster determinista de demo."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    paths = {
        "CSV fuente": resolve_path(config["paths"]["raw_data"]),
        "Series semanales": resolve_path(config["paths"]["processed_data"]) / "weekly_sales.csv",
        "Calibración": resolve_path(config["paths"]["processed_data"]) / "calibration.json",
        "Forecast para demo": resolve_path(config["paths"]["outputs"]) / "predictions.csv",
        "Métricas forecast": resolve_path(config["paths"]["outputs"]) / "forecast_metrics.csv",
        "Evaluación pricing": resolve_path(config["paths"]["outputs"]) / "pricing_evaluation.csv",
        "LSTM": resolve_path(config["paths"]["models"]) / "lstm.pt",
        "PPO": resolve_path(config["paths"]["models"]) / "ppo.zip",
        "DQN": resolve_path(config["paths"]["models"]) / "dqn.zip",
        "Generador cGAN": resolve_path(config["paths"]["models"]) / "cgan_generator.pt",
        "Clasificador Fashion-MNIST": resolve_path(config["paths"]["models"])
        / "fmnist_classifier.pt",
        "Métricas cGAN": resolve_path(config["paths"]["outputs"]) / "gan" / "metrics.json",
        "Capturas de respaldo": PROJECT_ROOT / "docs" / "captura_inicio.png",
    }
    results = [(path.is_file(), f"{label}: {path}") for label, path in paths.items()]
    if not all(ok for ok, _ in results):
        return False, results

    try:
        generator = FashionGenerator(paths["Generador cGAN"], device="cpu")
        classifier = load_fashion_classifier(paths["Clasificador Fashion-MNIST"], device="cpu")
        class_id = int(config["integration"]["gan_class_by_category"]["Clothing"])
        truncation = float(config["task_d"]["truncation"])
        candidates = generator.generate(class_id, 64, seed, truncation=truncation)
        ranked = rank_generated_images(candidates, class_id, classifier)
        selected = candidates[ranked[:3]]
        output_dir = resolve_path(config["paths"]["outputs"]) / "demo"
        poster_path = output_dir / f"poster_vestido_seed_{seed}.png"
        create_campaign_poster(
            selected,
            product_name="Vestido",
            class_id=class_id,
            destination=poster_path,
            hero_image=candidates[ranked[0]],
            template="Medianoche",
            style="Neón",
            discount_percent=20,
            offer_copy="-20%",
            seed=seed,
        )
        results.append((poster_path.is_file(), f"Póster determinista: {poster_path}"))
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        results.append((False, f"Póster demo: {error}"))

    return all(ok for ok, _ in results), results


def main() -> int:
    try:
        ready, results = prepare_demo()
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"✗ Configuración de demo: {error}")
        return 1
    for passed, message in results:
        print(f"{'✓' if passed else '✗'} {message}")
    print("Semáforo de demo: LISTA" if ready else "Semáforo de demo: REVISAR")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
