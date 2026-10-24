"""Entrena una CNN Fashion-MNIST para evaluar clases de imágenes generadas."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.urbanstyle.task_d_gan.classifier import FashionClassifier
from src.urbanstyle.task_d_gan.train import FASHION_CLASSES
from src.urbanstyle.utils.config import load_config, resolve_path
from src.urbanstyle.utils.seed import seed_everything
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

LOGGER = logging.getLogger("urbanstyle.task_d.classifier")


@torch.inference_mode()
def evaluate(
    model: FashionClassifier, loader: DataLoader, device: torch.device
) -> tuple[float, list[float]]:
    """Evalúa accuracy global y por clase en un loader sin augmentación."""
    model.eval()
    correct = torch.zeros(len(FASHION_CLASSES), dtype=torch.long)
    totals = torch.zeros_like(correct)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        for class_id in range(len(FASHION_CLASSES)):
            mask = labels == class_id
            totals[class_id] += mask.sum().cpu()
            correct[class_id] += ((predictions == labels) & mask).sum().cpu()
    per_class = [
        float(correct[index] / max(1, int(totals[index]))) for index in range(len(FASHION_CLASSES))
    ]
    accuracy = float(correct.sum() / totals.sum().clamp_min(1))
    return accuracy, per_class


def train_classifier(
    config: dict,
    *,
    epochs: int = 20,
    max_train_samples: int = 60000,
    seed: int | None = None,
    download: bool = True,
) -> dict:
    """Entrena y guarda el clasificador; detiene tras seis epochs sin mejora."""
    seed = int(config["project"]["seed"] if seed is None else seed)
    seed_everything(seed)
    torch.manual_seed(seed)
    dataset_path = resolve_path(config["paths"]["processed_data"]) / "fashion_mnist"
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = datasets.FashionMNIST(
        str(dataset_path), train=True, download=download, transform=train_transform
    )
    test_set = datasets.FashionMNIST(
        str(dataset_path), train=False, download=download, transform=eval_transform
    )
    sample_count = min(max_train_samples, len(train_set))
    train_subset = torch.utils.data.Subset(train_set, range(sample_count))
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    workers = int(config["task_d"].get("num_workers", 0))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_subset,
        batch_size=int(config["task_d"]["classifier_batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    model = FashionClassifier(classes=10, channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best_accuracy = 0.0
    best_state = None
    best_per_class = [0.0] * 10
    stale_epochs = 0
    history = []
    model_path = resolve_path(config["paths"]["models"]) / "fmnist_classifier.pt"
    output_dir = resolve_path(config["paths"]["outputs"]) / "gan"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = (
                images.to(device, non_blocking=pin_memory),
                labels.to(device, non_blocking=pin_memory),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        accuracy, per_class = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, seen),
            "test_accuracy": accuracy,
            "per_class_accuracy": per_class,
        }
        history.append(row)
        LOGGER.info(
            "Clasificador época %d/%d: train_loss=%.4f test_accuracy=%.2f%%",
            epoch + 1,
            epochs,
            row["train_loss"],
            accuracy * 100,
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_per_class = per_class
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
            torch.save(
                {"state_dict": best_state, "classes": 10, "channels": 32, "seed": seed},
                model_path,
            )
        else:
            stale_epochs += 1
        if best_accuracy >= 0.91 or stale_epochs >= 6:
            break

    if best_state is None:
        raise RuntimeError("El clasificador no produjo checkpoint.")
    report = {
        "seed": seed,
        "device": str(device),
        "training_samples": sample_count,
        "epochs_completed": len(history),
        "best_test_accuracy": best_accuracy,
        "target_accuracy": 0.91,
        "target_reached": best_accuracy >= 0.91,
        "per_class_accuracy": best_per_class,
        "history": history,
    }
    report_path = output_dir / "classifier_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if best_accuracy < 0.91:
        LOGGER.warning("Accuracy objetivo 91%% no alcanzado: %.2f%%", best_accuracy * 100)
    return {"model": model_path, "metrics": report_path, **report}


def main() -> int:
    """CLI de entrenamiento con opciones de reproducibilidad."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 2:
        parser.error("--epochs debe ser positivo y --samples al menos 2.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        result = train_classifier(
            load_config(PROJECT_ROOT / "config.yaml"),
            epochs=args.epochs,
            max_train_samples=args.samples,
            seed=args.seed,
        )
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("No se pudo entrenar el clasificador: %s", error)
        return 1
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result.items()
        if key not in {"history", "per_class_accuracy"}
    }
    print(json.dumps(printable, indent=2))
    return 0 if result["target_reached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
