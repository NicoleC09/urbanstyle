"""Entrenamiento standalone DCGAN/cGAN en Fashion-MNIST."""

from __future__ import annotations

import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from src.urbanstyle.task_d_gan.classifier import FashionClassifier, conditional_accuracy
from src.urbanstyle.task_d_gan.generate import load_fashion_classifier, sample_z
from src.urbanstyle.task_d_gan.models import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    DCDiscriminator,
    DCGenerator,
    initialize_dcgan,
)
from src.urbanstyle.utils.config import PROJECT_ROOT, load_config, resolve_path
from src.urbanstyle.utils.seed import seed_everything
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

LOGGER = logging.getLogger("urbanstyle.task_d.train")
FASHION_CLASSES = (
    "Camiseta",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
)


def _save_sample_grid(images: torch.Tensor, destination: Path, columns: int = 8) -> None:
    """Guarda imágenes normalizadas en [-1,1] como una grilla PNG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image((images.detach().cpu().clamp(-1, 1) + 1) / 2, destination, nrow=columns, padding=2)


@torch.no_grad()
def _update_ema(ema: nn.Module, generator: nn.Module, decay: float) -> None:
    """Actualiza parámetros EMA y sincroniza buffers de normalización."""
    source_parameters = dict(generator.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.lerp_(source_parameters[name].detach(), 1.0 - decay)
    source_buffers = dict(generator.named_buffers())
    for name, buffer in ema.named_buffers():
        if name in source_buffers:
            buffer.copy_(source_buffers[name])


@torch.inference_mode()
def _score_conditional_samples(
    generator: ConditionalGenerator,
    classifier: FashionClassifier,
    fixed_noise: torch.Tensor,
    fixed_labels: torch.Tensor,
) -> tuple[float, float, list[dict[str, float | int]]]:
    """Mide accuracy y confianza condicional en una grilla fija equilibrada."""
    generator.eval()
    classifier.eval()
    images = generator(fixed_noise, fixed_labels)
    accuracy, confidence, probabilities = conditional_accuracy(classifier, images, fixed_labels)
    per_class: list[dict[str, float | int]] = []
    for class_id in range(generator.classes):
        mask = fixed_labels == class_id
        if mask.any():
            per_class.append(
                {
                    "class_id": class_id,
                    "accuracy": float(
                        (probabilities[mask].argmax(dim=1) == fixed_labels[mask]).float().mean()
                    ),
                    "confidence": float(probabilities[mask, class_id].mean()),
                }
            )
    return accuracy, confidence, per_class


def _save_generator_checkpoint(
    path: Path,
    generator: nn.Module,
    *,
    gan_type: str,
    latent_size: int,
    channels: int,
    classes: int,
    embedding_size: int,
    seed: int,
    epoch: int,
    sample_count: int,
    ema_decay: float,
    accuracy: float | None,
    loss_type: str,
) -> None:
    """Guarda el formato compatible con FashionGenerator de inferencia."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gan_type": gan_type,
            "generator_state_dict": generator.state_dict(),
            "model_config": {
                "latent_size": latent_size,
                "channels": channels,
                "classes": classes,
                "embedding_size": embedding_size,
            },
            "seed": seed,
            "epochs": epoch,
            "training_samples": sample_count,
            "class_names": FASHION_CLASSES,
            "ema_decay": ema_decay,
            "best_conditional_accuracy": accuracy,
            "loss_type": loss_type,
        },
        path,
    )


def train_gan(
    config: dict,
    *,
    gan_type: str = "cgan",
    epochs: int | None = None,
    max_train_samples: int | None = None,
    seed: int | None = None,
    download: bool = True,
    loss_type: str = "bce",
    spectral_norm: bool = False,
) -> dict[str, Path]:
    """Entrena un GAN; cGAN usa smoothing, ruido decreciente y checkpoint EMA."""
    if gan_type not in {"dcgan", "cgan"}:
        raise ValueError("gan_type debe ser 'dcgan' o 'cgan'.")
    if loss_type not in {"bce", "hinge"}:
        raise ValueError("loss_type debe ser 'bce' o 'hinge'.")
    task = config["task_d"]
    seed = int(config["project"]["seed"] if seed is None else seed)
    epochs = int(task["epochs"] if epochs is None else epochs)
    max_train_samples = int(
        task["max_train_samples"] if max_train_samples is None else max_train_samples
    )
    if epochs < 1 or max_train_samples < 2:
        raise ValueError("Épocas debe ser positivo y se requieren al menos dos imágenes.")
    if gan_type == "dcgan" and loss_type == "hinge":
        raise ValueError("El plan B hinge/spectral-norm solo está habilitado para cGAN.")

    seed_everything(seed)
    torch.manual_seed(seed)
    dataset_path = resolve_path(config["paths"]["processed_data"]) / "fashion_mnist"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.FashionMNIST(
        root=str(dataset_path), train=True, download=download, transform=transform
    )
    sample_count = min(max_train_samples, len(dataset))
    batch_size = min(int(task["batch_size"]), sample_count)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    workers = int(task.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
        drop_last=sample_count % batch_size == 1,
    )

    latent_size = int(task["latent_size"])
    channels = int(task["generator_channels"])
    d_channels = int(task["discriminator_channels"])
    classes = int(task["classes"])
    embedding_size = int(task["embedding_size"])
    if gan_type == "cgan":
        generator: nn.Module = ConditionalGenerator(
            latent_size, classes, embedding_size, channels
        ).to(device)
        discriminator: nn.Module = ConditionalDiscriminator(
            classes, d_channels, spectral_norm=spectral_norm
        ).to(device)
    else:
        generator = DCGenerator(latent_size, channels).to(device)
        discriminator = DCDiscriminator(d_channels).to(device)
    generator.apply(initialize_dcgan)
    discriminator.apply(initialize_dcgan)
    ema_generator = deepcopy(generator).eval().requires_grad_(False)

    classifier: FashionClassifier | None = None
    fixed_labels: torch.Tensor | None = None
    if gan_type == "cgan":
        classifier_path = resolve_path(config["paths"]["models"]) / "fmnist_classifier.pt"
        classifier = load_fashion_classifier(classifier_path, device=device)
        class_repeats = (int(task["sample_count"]) + classes - 1) // classes
        fixed_labels = torch.arange(classes, device=device).repeat(class_repeats)[
            : int(task["sample_count"])
        ]

    if loss_type == "hinge":
        generator_lr = float(task["generator_learning_rate_hinge"])
        discriminator_lr = float(task["discriminator_learning_rate_hinge"])
    else:
        generator_lr = float(task["learning_rate"])
        discriminator_lr = float(task["learning_rate"])
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=generator_lr, betas=(float(task["beta1"]), 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=discriminator_lr, betas=(float(task["beta1"]), 0.999)
    )
    bce = nn.BCEWithLogitsLoss()
    fixed_noise = sample_z(
        int(task["sample_count"]), seed, float(task["truncation"]), latent_size, device
    )
    output_dir = resolve_path(config["paths"]["outputs"]) / "gan"
    type_dir = output_dir / gan_type
    model_dir = resolve_path(config["paths"]["models"])
    type_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    metrics_history: list[dict[str, Any]] = []
    snapshot_epochs = {int(value) for value in task["snapshot_epochs"]}
    ema_updates = 0
    best_accuracy = -1.0
    best_epoch = 0
    checkpoint_path = model_dir / f"{gan_type}_generator.pt"
    if gan_type == "dcgan":
        checkpoint_path = model_dir / "dcgan_generator.pt"
    ema_decay = float(task["ema_decay"])

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        total_generator_loss = 0.0
        total_discriminator_loss = 0.0
        batch_count = 0
        instance_noise_std = float(task["instance_noise_std"]) * (
            float(task["instance_noise_decay"]) ** (epoch - 1)
        )
        for real_images, real_labels in loader:
            real_images = real_images.to(device, non_blocking=pin_memory)
            real_labels = real_labels.to(device, non_blocking=pin_memory)
            current_batch = len(real_images)
            noise = torch.randn(current_batch, latent_size, device=device)
            fake_labels = torch.randint(0, classes, (current_batch,), device=device)
            real_logits_targets = torch.full(
                (current_batch,), float(task["label_smoothing"]), device=device
            )
            fake_targets = torch.zeros(current_batch, device=device)

            discriminator_optimizer.zero_grad(set_to_none=True)
            if gan_type == "cgan":
                fake_images = generator(noise, fake_labels)
                real_input = real_images + torch.randn_like(real_images) * instance_noise_std
                fake_input = (
                    fake_images.detach() + torch.randn_like(fake_images) * instance_noise_std
                )
                real_logits = discriminator(real_input, real_labels)
                fake_logits = discriminator(fake_input, fake_labels)
            else:
                fake_images = generator(noise)
                real_logits = discriminator(real_images)
                fake_logits = discriminator(fake_images.detach())
            if loss_type == "hinge":
                discriminator_loss = (
                    torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()
                )
            else:
                discriminator_loss = bce(real_logits, real_logits_targets) + bce(
                    fake_logits, fake_targets
                )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            generated_logits = (
                discriminator(fake_images, fake_labels)
                if gan_type == "cgan"
                else discriminator(fake_images)
            )
            generator_loss = (
                -generated_logits.mean()
                if loss_type == "hinge"
                else bce(generated_logits, torch.ones_like(generated_logits))
            )
            generator_loss.backward()
            generator_optimizer.step()
            if gan_type == "cgan":
                ema_updates += 1
                decay = min(ema_decay, (1 + ema_updates) / (10 + ema_updates))
                _update_ema(ema_generator, generator, decay)
            total_generator_loss += float(generator_loss.detach())
            total_discriminator_loss += float(discriminator_loss.detach())
            batch_count += 1

        history.append(
            {
                "epoch": epoch,
                "generator_loss": total_generator_loss / max(1, batch_count),
                "discriminator_loss": total_discriminator_loss / max(1, batch_count),
            }
        )
        LOGGER.info(
            "%s epoch %d/%d: G=%.4f D=%.4f",
            gan_type,
            epoch,
            epochs,
            history[-1]["generator_loss"],
            history[-1]["discriminator_loss"],
        )

        if gan_type == "cgan":
            assert classifier is not None and fixed_labels is not None
            accuracy, confidence, class_metrics = _score_conditional_samples(
                ema_generator, classifier, fixed_noise, fixed_labels
            )
            metrics_history.append(
                {
                    "epoch": epoch,
                    "conditional_accuracy": accuracy,
                    "mean_confidence": confidence,
                    "per_class": class_metrics,
                }
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                _save_generator_checkpoint(
                    checkpoint_path,
                    ema_generator,
                    gan_type=gan_type,
                    latent_size=latent_size,
                    channels=channels,
                    classes=classes,
                    embedding_size=embedding_size,
                    seed=seed,
                    epoch=epoch,
                    sample_count=sample_count,
                    ema_decay=ema_decay,
                    accuracy=best_accuracy,
                    loss_type=loss_type,
                )
            if epoch % int(task["checkpoint_interval"]) == 0 or epoch == epochs:
                _save_generator_checkpoint(
                    model_dir / f"cgan_generator_epoch_{epoch:03d}.pt",
                    ema_generator,
                    gan_type=gan_type,
                    latent_size=latent_size,
                    channels=channels,
                    classes=classes,
                    embedding_size=embedding_size,
                    seed=seed,
                    epoch=epoch,
                    sample_count=sample_count,
                    ema_decay=ema_decay,
                    accuracy=accuracy,
                    loss_type=loss_type,
                )
        elif epoch % int(task["checkpoint_interval"]) == 0 or epoch == epochs:
            _save_generator_checkpoint(
                model_dir / f"dcgan_generator_epoch_{epoch:03d}.pt",
                generator,
                gan_type=gan_type,
                latent_size=latent_size,
                channels=channels,
                classes=classes,
                embedding_size=embedding_size,
                seed=seed,
                epoch=epoch,
                sample_count=sample_count,
                ema_decay=ema_decay,
                accuracy=None,
                loss_type=loss_type,
            )

        should_snapshot = epoch in snapshot_epochs or epoch == epochs
        if gan_type == "cgan" and should_snapshot:
            ema_generator.eval()
            with torch.inference_mode():
                samples = ema_generator(fixed_noise, fixed_labels)
            snapshot_path = output_dir / f"samples_epoch_{epoch:02d}.png"
            _save_sample_grid(samples, snapshot_path, columns=8)
            shutil.copy2(snapshot_path, type_dir / f"epoch_{epoch:03d}.png")
        elif gan_type == "dcgan" and should_snapshot:
            generator.eval()
            with torch.inference_mode():
                samples = generator(fixed_noise)
            snapshot_path = type_dir / f"epoch_{epoch:03d}.png"
            _save_sample_grid(samples, snapshot_path, columns=8)

    history_path = type_dir / "losses.csv"
    import pandas as pd

    pd.DataFrame(history).to_csv(history_path, index=False)
    (output_dir / "losses.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if gan_type == "cgan":
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "classifier_checkpoint": str(model_dir / "fmnist_classifier.pt"),
                    "best_conditional_accuracy": best_accuracy,
                    "best_epoch": best_epoch,
                    "classes": FASHION_CLASSES,
                    "epochs": metrics_history,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    loss_plot = output_dir / "losses.png"
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    axis.plot(
        [row["epoch"] for row in history],
        [row["generator_loss"] for row in history],
        label="Generador",
        color="#B8F36B",
    )
    axis.plot(
        [row["epoch"] for row in history],
        [row["discriminator_loss"] for row in history],
        label="Discriminador",
        color="#F28C78",
    )
    axis.set(
        title=f"{gan_type.upper()} · pérdidas por época", xlabel="Época", ylabel=loss_type.upper()
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(loss_plot, dpi=150)
    plt.close(figure)
    summary_path = type_dir / "training.json"
    summary_path.write_text(
        json.dumps(
            {
                "gan_type": gan_type,
                "seed": seed,
                "epochs": epochs,
                "training_samples": sample_count,
                "device": str(device),
                "loss_type": loss_type,
                "spectral_norm": spectral_norm,
                "last_generator_loss": history[-1]["generator_loss"],
                "last_discriminator_loss": history[-1]["discriminator_loss"],
                "best_conditional_accuracy": best_accuracy if gan_type == "cgan" else None,
                "best_epoch": best_epoch if gan_type == "cgan" else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    last_samples = output_dir / f"samples_epoch_{epochs:02d}.png"
    if not last_samples.is_file():
        last_samples = type_dir / f"epoch_{epochs:03d}.png"
    return {
        "model": checkpoint_path,
        "samples": last_samples,
        "losses": history_path,
        "loss_json": output_dir / "losses.json",
        "metrics": output_dir / "metrics.json",
        "loss_plot": loss_plot,
        "summary": summary_path,
    }


def train_from_project(
    *,
    gan_type: str = "cgan",
    epochs: int | None = None,
    max_train_samples: int | None = None,
    seed: int | None = None,
    config_path: str | Path | None = None,
    loss_type: str = "bce",
    spectral_norm: bool = False,
) -> dict[str, Path]:
    """Carga config.yaml y ejecuta el trainer del GAN seleccionado."""
    return train_gan(
        load_config(config_path or PROJECT_ROOT / "config.yaml"),
        gan_type=gan_type,
        epochs=epochs,
        max_train_samples=max_train_samples,
        seed=seed,
        loss_type=loss_type,
        spectral_norm=spectral_norm,
    )
