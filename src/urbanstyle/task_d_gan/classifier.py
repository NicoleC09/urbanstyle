"""Clasificador CNN compacto para puntuar muestras Fashion-MNIST."""

from __future__ import annotations

import torch
from torch import nn


class FashionClassifier(nn.Module):
    """CNN de diez clases sobre imágenes normalizadas de 28×28."""

    def __init__(self, classes: int = 10, channels: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Devuelve logits de clase para un batch de imágenes [N,1,28,28]."""
        return self.classifier(self.features(images))


@torch.inference_mode()
def conditional_accuracy(
    classifier: FashionClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, torch.Tensor]:
    """Mide exactitud condicional y confianza media para clases solicitadas."""
    logits = classifier(images)
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    accuracy = (predictions == labels).float().mean().item()
    requested_confidence = probabilities.gather(1, labels[:, None]).mean().item()
    return accuracy, requested_confidence, probabilities
