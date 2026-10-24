"""Pruebas de forma y carga del clasificador Fashion-MNIST."""

import torch
from src.urbanstyle.task_d_gan.classifier import FashionClassifier


def test_classifier_returns_ten_class_logits() -> None:
    classifier = FashionClassifier(classes=10, channels=8)
    logits = classifier(torch.randn(4, 1, 28, 28))

    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_classifier_checkpoint_roundtrip(tmp_path) -> None:
    model = FashionClassifier(classes=10, channels=8)
    path = tmp_path / "fmnist_classifier.pt"
    torch.save({"state_dict": model.state_dict(), "classes": 10, "channels": 8}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    loaded = FashionClassifier(checkpoint["classes"], checkpoint["channels"])
    loaded.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loaded.eval()

    torch.testing.assert_close(model(torch.zeros(1, 1, 28, 28)), loaded(torch.zeros(1, 1, 28, 28)))
