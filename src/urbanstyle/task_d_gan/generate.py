"""Carga de generador entrenado, generación reproducible y póster promocional."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from src.urbanstyle.task_d_gan.classifier import FashionClassifier
from src.urbanstyle.task_d_gan.models import ConditionalGenerator, DCGenerator
from src.urbanstyle.utils.config import resolve_path
from torchvision.utils import save_image

LOGGER = logging.getLogger("urbanstyle.task_d.generate")


def sample_z(
    count: int,
    seed: int,
    truncation: float = 0.8,
    latent_size: int = 100,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Muestrea ruido reproducible y trunca sus valores al intervalo [-ψ, ψ]."""
    if count < 1 or latent_size < 1:
        raise ValueError("count y latent_size deben ser positivos.")
    if truncation <= 0:
        raise ValueError("truncation debe ser mayor que cero.")
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device).manual_seed(int(seed))
    noise = torch.randn(count, latent_size, generator=generator, device=target_device)
    return noise.clamp(-truncation, truncation)


def load_fashion_classifier(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> FashionClassifier:
    """Carga un clasificador entrenado y lo fija en modo inferencia."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta el clasificador {path}; ejecuta python scripts/train_classifier.py."
        )
    target_device = torch.device(device)
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    classifier = FashionClassifier(
        classes=int(checkpoint.get("classes", 10)),
        channels=int(checkpoint.get("channels", 32)),
    )
    classifier.load_state_dict(checkpoint["state_dict"])
    return classifier.to(target_device).eval()


class FashionGenerator:
    """Adaptador de checkpoint que expone generación determinista sencilla."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Falta el generador {path}; ejecuta python scripts/train_gan.py --type cgan."
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.gan_type = self.checkpoint["gan_type"]
        model_config = self.checkpoint["model_config"]
        if self.gan_type == "cgan":
            self.generator = ConditionalGenerator(
                model_config["latent_size"],
                model_config["classes"],
                model_config["embedding_size"],
                model_config["channels"],
            )
        elif self.gan_type == "dcgan":
            self.generator = DCGenerator(model_config["latent_size"], model_config["channels"])
        else:
            raise ValueError(f"Tipo GAN desconocido en checkpoint: {self.gan_type}")
        self.generator.load_state_dict(self.checkpoint["generator_state_dict"])
        self.generator.to(self.device).eval()
        self.latent_size = int(model_config["latent_size"])

    @torch.inference_mode()
    def generate(
        self, class_id: int, count: int = 1, seed: int = 42, truncation: float = 0.8
    ) -> torch.Tensor:
        """Genera count imágenes [-1,1] idénticas para la misma semilla/clase."""
        classes = int(self.checkpoint["model_config"]["classes"])
        if not 0 <= class_id < classes:
            raise ValueError(f"La clase debe estar entre 0 y {classes - 1}.")
        if count < 1:
            raise ValueError("La cantidad de imágenes debe ser positiva.")
        noise = sample_z(
            count, seed, truncation=truncation, latent_size=self.latent_size, device=self.device
        )
        if self.gan_type == "cgan":
            labels = torch.full((count,), class_id, dtype=torch.long, device=self.device)
            return self.generator(noise, labels).cpu()
        return self.generator(noise).cpu()


def save_generated_grid(images: torch.Tensor, destination: str | Path, columns: int = 4) -> Path:
    """Guarda una galería PNG a partir de imágenes normalizadas en [-1,1]."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image((images.clamp(-1, 1) + 1) / 2, path, nrow=max(1, columns), padding=3)
    return path


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Busca una fuente sans serif instalada y usa la fuente PIL como fallback."""
    candidates = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_poster(
    images: torch.Tensor,
    *,
    category: str,
    discount: int,
    destination: str | Path,
    brand: str = "UrbanStyle",
) -> Path:
    """Compone imágenes generadas en una pieza promocional PNG."""
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("Se espera un tensor N×1×H×W de imágenes Fashion-MNIST.")
    if not 1 <= discount <= 90:
        raise ValueError("El descuento debe estar entre 1% y 90%.")
    count = int(images.shape[0])
    columns = min(3, max(1, count))
    rows = (count + columns - 1) // columns
    tile_size = 196
    margin = 44
    header_height = 210
    image_width = columns * tile_size + (columns - 1) * 16
    image_height = rows * tile_size + (rows - 1) * 16
    canvas = Image.new(
        "RGB", (image_width + margin * 2, header_height + image_height + 62), "#101512"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 28), brand.upper(), fill="#B8F36B", font=_font(22, bold=True))
    draw.text((margin, 66), f"-{discount}%", fill="#F2F2E9", font=_font(64, bold=True))
    draw.text(
        (margin + 225, 93), f"EN {category.upper()}", fill="#F28C78", font=_font(24, bold=True)
    )
    draw.line((margin, 176, canvas.width - margin, 176), fill="#354139", width=2)
    for index, tensor in enumerate(images):
        gray = ((tensor[0].detach().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).numpy()
        tile = Image.fromarray(gray, mode="L").resize(
            (tile_size, tile_size), Image.Resampling.LANCZOS
        )
        rgb_tile = Image.new("RGB", (tile_size, tile_size), "#222C25")
        rgb_tile.paste(tile.convert("RGB"), (0, 0))
        x = margin + (index % columns) * (tile_size + 16)
        y = header_height + (index // columns) * (tile_size + 16)
        canvas.paste(rgb_tile, (x, y))
    draw.text(
        (margin, canvas.height - 34),
        "NUEVA TEMPORADA  ·  URBAN STYLE, EVERY DAY",
        fill="#A8B1A8",
        font=_font(14),
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return path


def generator_checkpoint(config: dict, gan_type: str = "cgan") -> Path:
    """Resuelve la ruta estándar del generador solicitado."""
    if gan_type not in {"cgan", "dcgan"}:
        raise ValueError("El tipo debe ser cgan o dcgan.")
    return resolve_path(config["paths"]["models"]) / f"{gan_type}_generator.pt"
