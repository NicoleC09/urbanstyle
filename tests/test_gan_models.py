"""Pruebas de formas y rangos para modelos generativos Task D."""

import torch
from src.urbanstyle.task_d_gan.models import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    DCDiscriminator,
    DCGenerator,
)


def test_dcgan_generator_and_discriminator_shapes_and_range() -> None:
    generator = DCGenerator(latent_size=32, channels=8)
    discriminator = DCDiscriminator(channels=8)
    images = generator(torch.randn(4, 32))

    assert images.shape == (4, 1, 28, 28)
    assert images.min() >= -1 and images.max() <= 1
    assert discriminator(images).shape == (4,)


def test_conditional_gan_shapes_and_class_conditioning() -> None:
    generator = ConditionalGenerator(latent_size=32, classes=10, embedding_size=8, channels=16)
    discriminator = ConditionalDiscriminator(classes=10, channels=8)
    labels = torch.tensor([0, 2, 5, 9])
    images = generator(torch.randn(4, 32), labels)

    assert images.shape == (4, 1, 28, 28)
    assert images.min() >= -1 and images.max() <= 1
    assert discriminator(images, labels).shape == (4,)
