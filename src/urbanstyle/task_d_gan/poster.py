"""Composición de pósteres UrbanStyle desde muestras Fashion-MNIST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from src.urbanstyle.task_d_gan.classifier import FashionClassifier

POSTER_SIZE = (1080, 1350)
TEMPLATES = ("Medianoche", "Hielo", "Eléctrico")

_PALETTES = {
    "Medianoche": {
        "top": (5, 9, 20),
        "bottom": (18, 48, 107),
        "text": (244, 247, 252),
        "accent": (47, 107, 255),
        "muted": (143, 160, 191),
        "pill": (18, 29, 51),
        "product": ((12, 25, 74), (47, 107, 255), (255, 255, 255)),
    },
    "Hielo": {
        "top": (239, 247, 255),
        "bottom": (207, 229, 255),
        "text": (9, 24, 55),
        "accent": (47, 107, 255),
        "muted": (63, 86, 125),
        "pill": (220, 235, 255),
        "product": ((10, 35, 95), (47, 107, 255), (255, 255, 255)),
    },
    "Eléctrico": {
        "top": (24, 79, 255),
        "bottom": (7, 33, 128),
        "text": (255, 255, 255),
        "accent": (124, 196, 255),
        "muted": (211, 229, 255),
        "pill": (22, 62, 183),
        "product": ((22, 46, 138), (173, 211, 255), (255, 255, 255)),
    },
}


def rank_generated_images(
    images: torch.Tensor,
    class_id: int,
    classifier: FashionClassifier | None,
) -> list[int]:
    """Ordena índices por probabilidad de la clase solicitada, mejor primero."""
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("Las candidatas deben tener forma N×1×28×28.")
    if images.shape[0] == 0:
        raise ValueError("Se requiere al menos una candidata.")
    if classifier is None:
        return list(range(len(images)))
    classifier.eval()
    device = next(classifier.parameters()).device
    with torch.inference_mode():
        probabilities = classifier(images.to(device)).softmax(dim=1)[:, class_id]
    return torch.argsort(probabilities, descending=True).cpu().tolist()


def campaign_offer_copy(discount_percent: float) -> tuple[str, int]:
    """Redondea el descuento a múltiplos de cinco y evita ofertas menores al 10 %."""
    discount = max(0, int(round(float(discount_percent) / 5) * 5))
    if discount >= 10:
        return f"-{discount}%", discount
    if discount > 0:
        return "Precio especial", discount
    return "Nueva colección", 0


def agent_offer_copy(action: int, reference_price: float, agent_price: float) -> tuple[str, int]:
    """Crea copy coherente con acción y diferencia real entre los precios."""
    if action == 1:
        return "Nueva colección", 0
    if action > 1:
        return "Edición premium", 0
    if reference_price <= 0:
        return "Precio especial", 0
    discount = max(0.0, (reference_price - agent_price) / reference_price * 100)
    return campaign_offer_copy(discount)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    root = Path(__file__).resolve().parents[3]
    inter = root / "assets" / "fonts" / "Inter.ttf"
    windows_font = Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    for path in (inter, windows_font):
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _background(template: str, seed: int) -> Image.Image:
    palette = _PALETTES[template]
    width, height = POSTER_SIZE
    top = np.asarray(palette["top"], dtype=np.float32)
    bottom = np.asarray(palette["bottom"], dtype=np.float32)
    ratio = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    pixels = top[None, None, :] * (1 - ratio) + bottom[None, None, :] * ratio
    pixels = np.broadcast_to(pixels, (height, width, 3)).copy()
    rng = np.random.default_rng(seed)
    pixels += rng.normal(0, 8.0, (height, width, 1))
    image = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)), "RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    accent = (*palette["accent"], 36)
    for radius in (210, 330, 455):
        bounds = (width // 2 - radius, 610 - radius, width // 2 + radius, 610 + radius)
        draw.arc(bounds, 205, 335, fill=accent, width=2)
    grid_color = (*palette["text"], 12)
    for x in range(48, width, 72):
        draw.line((x, 230, x, 1090), fill=grid_color, width=1)
    for y in range(260, 1090, 72):
        draw.line((48, y, width - 48, y), fill=grid_color, width=1)
    return image


def _product_rgba(
    tensor: torch.Tensor,
    colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    size: int,
    style: str,
    rotation: float = 0,
) -> Image.Image:
    grayscale = ((tensor.detach().cpu().squeeze().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).numpy()
    luminance = grayscale.astype(np.float32) / 255.0
    alpha = np.power(np.clip((luminance - 0.08) / 0.92, 0, 1), 0.8)
    stops = np.asarray(colors, dtype=np.float32)
    low_to_mid = np.clip(luminance * 2, 0, 1)[..., None]
    mid_to_high = np.clip((luminance - 0.5) * 2, 0, 1)[..., None]
    rgb = np.where(
        (luminance <= 0.5)[..., None],
        stops[0] * (1 - low_to_mid) + stops[1] * low_to_mid,
        stops[1] * (1 - mid_to_high) + stops[2] * mid_to_high,
    )
    rgba = Image.fromarray(
        np.uint8(np.clip(np.concatenate((rgb, (alpha * 255)[..., None]), axis=2), 0, 255)),
        "RGBA",
    )
    resampling = Image.Resampling.NEAREST if style == "Pixel" else Image.Resampling.LANCZOS
    rgba = rgba.resize((size, size), resampling)
    if style == "Neón":
        rgb_image = rgba.convert("RGB").filter(
            ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3)
        )
        rgb_image.putalpha(rgba.getchannel("A"))
        rgba = rgb_image
    else:
        pixels = np.asarray(rgba).copy()
        yy, xx = np.indices(pixels.shape[:2])
        dither = ((xx + yy) % 4 == 0) & (pixels[:, :, 3] > 12)
        pixels[dither, :3] = np.clip(pixels[dither, :3] + 28, 0, 255)
        pixels[yy % 5 == 0, 3] = (pixels[yy % 5 == 0, 3] * 0.88).astype(np.uint8)
        rgba = Image.fromarray(pixels, "RGBA")
    if rotation:
        rgba = rgba.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    return rgba


def _composite_product(
    canvas: Image.Image,
    tensor: torch.Tensor,
    center: tuple[int, int],
    size: int,
    colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    style: str,
    *,
    rotation: float = 0,
    opacity: float = 1.0,
    hero: bool = False,
) -> None:
    product = _product_rgba(tensor, colors, size, style, rotation)
    alpha = product.getchannel("A")
    if opacity < 1:
        alpha = alpha.point(lambda value: round(value * opacity))
        product.putalpha(alpha)
    x = center[0] - product.width // 2
    y = center[1] - product.height // 2
    shadow_mask = alpha.filter(ImageFilter.GaussianBlur(max(8, size // 18)))
    shadow = Image.new("RGBA", product.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_mask.point(lambda value: round(value * 0.32)))
    canvas.alpha_composite(shadow, (x + size // 22, y + size // 18))
    if hero:
        glow = Image.new("RGBA", product.size, (47, 107, 255, 0))
        glow.putalpha(
            alpha.filter(ImageFilter.GaussianBlur(max(10, size // 12))).point(
                lambda value: round(value * 0.34)
            )
        )
        canvas.alpha_composite(glow, (x, y))
    canvas.alpha_composite(product, (x, y))
    if hero:
        reflection = ImageOps.flip(product)
        fade = reflection.getchannel("A")
        fade = fade.point(lambda value: round(value * 0.12))
        fade = fade.resize((fade.width, max(1, fade.height // 3)), Image.Resampling.BILINEAR)
        reflection = reflection.crop((0, 0, reflection.width, fade.height))
        reflection.putalpha(fade)
        canvas.alpha_composite(reflection, (x, y + size // 2 + 34))


def create_campaign_poster(
    images: torch.Tensor,
    *,
    product_name: str,
    class_id: int,
    destination: str | Path,
    classifier: FashionClassifier | None = None,
    hero_image: torch.Tensor | None = None,
    template: str = "Medianoche",
    style: str = "Neón",
    discount_percent: int = 20,
    offer_copy: str | None = None,
    seed: int = 42,
    brand: str = "URBANSTYLE",
) -> Path:
    """Genera un póster 4:5 con productos recortados y oferta publicitaria."""
    if template not in TEMPLATES:
        raise ValueError(f"Plantilla desconocida: {template}.")
    if style not in {"Neón", "Pixel"}:
        raise ValueError("El estilo debe ser Neón o Pixel.")
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("Se espera un tensor N×1×28×28 Fashion-MNIST.")
    if images.shape[0] == 0:
        raise ValueError("Se requiere al menos una imagen para el póster.")
    order = rank_generated_images(images, class_id, classifier)
    chosen = [images[index] for index in order[:3]]
    if hero_image is not None:
        chosen = [
            hero_image,
            *[images[index] for index in order if not torch.equal(images[index], hero_image)][:2],
        ]
    while len(chosen) < 3:
        chosen.append(chosen[-1])
    palette = _PALETTES[template]
    canvas = _background(template, seed).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = POSTER_SIZE
    text = palette["text"]
    accent = palette["accent"]
    pill_fill = (*palette["pill"], 238)
    draw.text((74, 58), brand, fill=text, font=_font(38, bold=True), stroke_width=0)
    draw.rounded_rectangle((760, 58, 1006, 112), radius=27, fill=pill_fill)
    draw.text((783, 74), "NUEVA COLECCIÓN", fill=text, font=_font(17, bold=True))
    draw.text((74, 176), product_name.upper(), fill=palette["muted"], font=_font(30, bold=True))

    copy = offer_copy
    if copy is None:
        copy = (
            f"-{discount_percent}%"
            if discount_percent >= 10
            else "Precio especial"
            if discount_percent > 0
            else "Nueva colección"
        )
    if copy.startswith("-") and copy.endswith("%"):
        offer_font = _font(230, bold=True)
    else:
        offer_font = _font(94, bold=True)
    draw.text((68, 224), copy, fill=text, font=offer_font, stroke_width=0)

    _composite_product(
        canvas,
        chosen[0],
        (width // 2, 760),
        570,
        palette["product"],
        style,
        hero=True,
    )
    _composite_product(
        canvas,
        chosen[1],
        (242, 812),
        270,
        palette["product"],
        style,
        rotation=-9,
        opacity=0.78,
    )
    _composite_product(
        canvas,
        chosen[2],
        (838, 828),
        270,
        palette["product"],
        style,
        rotation=8,
        opacity=0.78,
    )

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rounded_rectangle((74, 1092, 1006, 1180), radius=42, fill=(*accent, 255))
    draw.text((365, 1114), "Compra ahora  →", fill=(255, 255, 255), font=_font(33, bold=True))
    footer = "Solo esta semana · Hasta agotar existencias"
    draw.text((74, 1243), footer, fill=palette["muted"], font=_font(21))

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(path, format="PNG", optimize=True)
    return path
