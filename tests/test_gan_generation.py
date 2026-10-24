"""Pruebas de generación determinista y del póster promocional."""

import torch
from PIL import Image
from src.urbanstyle.task_d_gan.generate import FashionGenerator, sample_z
from src.urbanstyle.task_d_gan.models import ConditionalGenerator
from src.urbanstyle.task_d_gan.poster import (
    TEMPLATES,
    _product_rgba,
    agent_offer_copy,
    campaign_offer_copy,
    create_campaign_poster,
)


def test_conditional_generation_is_deterministic(tmp_path) -> None:
    model_config = {"latent_size": 12, "channels": 4, "classes": 10, "embedding_size": 5}
    generator = ConditionalGenerator(12, 10, 5, 4)
    checkpoint_path = tmp_path / "cgan_generator.pt"
    torch.save(
        {
            "gan_type": "cgan",
            "generator_state_dict": generator.state_dict(),
            "model_config": model_config,
        },
        checkpoint_path,
    )
    loaded = FashionGenerator(checkpoint_path, device="cpu")

    first = loaded.generate(class_id=7, count=3, seed=1234)
    second = loaded.generate(class_id=7, count=3, seed=1234)

    assert first.shape == (3, 1, 28, 28)
    assert first.min() >= -1 and first.max() <= 1
    torch.testing.assert_close(first, second)


def test_campaign_poster_has_exact_4_5_size_for_every_template(tmp_path) -> None:
    images = torch.zeros((3, 1, 28, 28))

    for template in TEMPLATES:
        destination = create_campaign_poster(
            images,
            product_name="Zapatilla",
            class_id=7,
            discount_percent=20,
            template=template,
            destination=tmp_path / f"poster_{template}.png",
        )
        with Image.open(destination) as image:
            assert image.format == "PNG"
            assert image.size == (1080, 1350)


def test_truncated_noise_is_seeded_and_bounded() -> None:
    first = sample_z(32, seed=99, truncation=0.8, latent_size=100)
    second = sample_z(32, seed=99, truncation=0.8, latent_size=100)

    assert first.shape == (32, 100)
    assert first.abs().max() <= 0.8
    torch.testing.assert_close(first, second)


def test_campaign_copy_uses_actual_price_decision_and_rounding() -> None:
    assert agent_offer_copy(action=0, reference_price=100, agent_price=90) == ("-10%", 10)
    assert agent_offer_copy(action=0, reference_price=100, agent_price=96) == (
        "Precio especial",
        5,
    )
    assert agent_offer_copy(action=1, reference_price=100, agent_price=100) == (
        "Nueva colección",
        0,
    )
    assert agent_offer_copy(action=2, reference_price=100, agent_price=105) == (
        "Edición premium",
        0,
    )
    assert campaign_offer_copy(20) == ("-20%", 20)


def test_product_mask_removes_black_canvas_background() -> None:
    product = _product_rgba(
        torch.full((1, 28, 28), -1.0),
        ((12, 25, 74), (47, 107, 255), (255, 255, 255)),
        168,
        "Neón",
    )

    assert product.getchannel("A").getpixel((0, 0)) == 0
