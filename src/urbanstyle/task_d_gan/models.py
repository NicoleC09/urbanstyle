"""Arquitecturas DCGAN y cGAN para Fashion-MNIST de 28×28."""

from __future__ import annotations

import torch
from torch import nn


class DCGenerator(nn.Module):
    """Genera imágenes monocromas 28×28 en el rango [-1, 1]."""

    def __init__(self, latent_size: int = 64, channels: int = 32):
        super().__init__()
        self.latent_size = latent_size
        self.network = nn.Sequential(
            nn.Linear(latent_size, channels * 4 * 7 * 7),
            nn.BatchNorm1d(channels * 4 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (channels * 4, 7, 7)),
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Convierte ruido gaussiano [N, latent_size] a imágenes [N,1,28,28]."""
        return self.network(noise)


class DCDiscriminator(nn.Module):
    """Clasificador DCGAN que devuelve logits de real/falso."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(channels * 2 * 7 * 7, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Devuelve un logit por imagen."""
        return self.network(images).squeeze(-1)


class ConditionalGenerator(nn.Module):
    """Generador convolucional condicional de 28×28 con embedding de etiqueta."""

    def __init__(
        self,
        latent_size: int = 100,
        classes: int = 10,
        embedding_size: int = 32,
        channels: int = 128,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.classes = classes
        self.embedding = nn.Embedding(classes, embedding_size)
        self.project = nn.Sequential(
            nn.Linear(latent_size + embedding_size, channels * 7 * 7),
            nn.BatchNorm1d(channels * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Unflatten(1, (channels, 7, 7)),
            nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels // 2, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Genera imágenes condicionadas por clase Fashion-MNIST."""
        if noise.ndim != 2 or noise.shape[1] != self.latent_size:
            raise ValueError(f"El ruido debe tener forma [N, {self.latent_size}].")
        if labels.ndim != 1 or labels.shape[0] != noise.shape[0]:
            raise ValueError("Debe recibirse una etiqueta por vector de ruido.")
        conditioned = torch.cat((noise, self.embedding(labels)), dim=1)
        return self.upsample(self.project(conditioned))


class ConditionalDiscriminator(nn.Module):
    """Discriminador cGAN que recibe imagen y mapa espacial de clase."""

    def __init__(
        self,
        classes: int = 10,
        channels: int = 64,
        *,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.classes = classes
        conv_1 = nn.Conv2d(1 + classes, channels, 4, 2, 1)
        conv_2 = nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False)
        linear = nn.Linear(channels * 2 * 7 * 7, 1)
        if spectral_norm:
            conv_1 = nn.utils.spectral_norm(conv_1)
            conv_2 = nn.utils.spectral_norm(conv_2)
            linear = nn.utils.spectral_norm(linear)
        self.network = nn.Sequential(
            conv_1,
            nn.LeakyReLU(0.2, inplace=True),
            conv_2,
            nn.BatchNorm2d(channels * 2) if not spectral_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            nn.Flatten(),
            linear,
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Devuelve logits tras añadir un canal one-hot espacial por clase."""
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("Las imágenes deben tener forma [N, 1, 28, 28].")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError("Debe recibirse una etiqueta por imagen.")
        class_map = torch.nn.functional.one_hot(labels, self.classes).to(images.dtype)
        class_map = class_map[:, :, None, None].expand(-1, -1, images.shape[2], images.shape[3])
        return self.network(torch.cat((images, class_map), dim=1)).squeeze(-1)


def initialize_dcgan(module: nn.Module) -> None:
    """Inicialización convencional DCGAN para convoluciones y batch norm."""
    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
