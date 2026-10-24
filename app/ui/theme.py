"""Tokens de diseño UrbanStyle y primitivas Pygame antialias."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pygame


@dataclass(frozen=True)
class Theme:
    """Paleta azul/negro/blanco con alias para las pestañas existentes."""

    background: pygame.Color
    panel: pygame.Color
    elevated: pygame.Color
    border: pygame.Color
    primary: pygame.Color
    hover: pygame.Color
    sky: pygame.Color
    deep: pygame.Color
    text: pygame.Color
    muted: pygame.Color
    disabled: pygame.Color
    error: pygame.Color
    success: pygame.Color
    category_beauty: pygame.Color
    category_clothing: pygame.Color
    category_electronics: pygame.Color

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Theme:
        """Construye el theme desde tokens centralizados en config.yaml."""
        palette = config["app"]["palette"]
        return cls(**{name: pygame.Color(value) for name, value in palette.items()})

    @property
    def surface(self) -> pygame.Color:
        """Alias legacy para el fondo de panel."""
        return self.panel

    @property
    def surface_alt(self) -> pygame.Color:
        """Alias legacy para tarjetas elevadas."""
        return self.elevated

    @property
    def foreground(self) -> pygame.Color:
        """Alias legacy para el texto principal."""
        return self.text

    @property
    def accent(self) -> pygame.Color:
        """Alias legacy para acciones primarias."""
        return self.primary

    @property
    def accent_dark(self) -> pygame.Color:
        """Alias legacy para azul profundo."""
        return self.deep

    @property
    def coral(self) -> pygame.Color:
        """Alias legacy para estado de error."""
        return self.error

    @property
    def gold(self) -> pygame.Color:
        """Alias legacy de alerta atenuada."""
        return self.muted

    @property
    def line(self) -> pygame.Color:
        """Alias legacy para bordes."""
        return self.border

    @property
    def white(self) -> pygame.Color:
        """Token blanco de la paleta."""
        return self.category_electronics


@lru_cache(maxsize=64)
def _panel_surface(
    size: tuple[int, int],
    color: tuple[int, int, int, int],
    radius: int,
    border_color: tuple[int, int, int, int] | None,
    gradient: bool,
    shadow: bool,
) -> pygame.Surface:
    """Rasteriza y conserva paneles repetidos para evitar trabajo por frame."""
    shadow_offset = 5 if shadow else 0
    result = pygame.Surface((size[0], size[1] + shadow_offset), pygame.SRCALPHA)
    if shadow:
        pygame.draw.rect(
            result,
            (0, 0, 0, 78),
            pygame.Rect(0, shadow_offset, size[0], size[1]),
            border_radius=max(0, radius + 2),
        )
    fill = pygame.Surface(size, pygame.SRCALPHA)
    top = pygame.Color(color)
    if gradient:
        bottom = top.lerp(pygame.Color("#1F2E4D"), 0.16)
        for y in range(size[1]):
            blend = y / max(1, size[1] - 1)
            pygame.draw.line(fill, top.lerp(bottom, blend), (0, y), (size[0], y))
        mask = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
        fill.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    else:
        pygame.draw.rect(fill, top, fill.get_rect(), border_radius=radius)
    if border_color is not None:
        pygame.draw.rect(
            fill,
            pygame.Color(border_color),
            fill.get_rect(),
            width=1,
            border_radius=radius,
        )
    result.blit(fill, (0, 0))
    return result


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: pygame.Color,
    radius: int = 16,
    shadow: bool = False,
    border_color: pygame.Color | None = None,
    gradient: bool = True,
) -> None:
    """Dibuja tarjeta redondeada con borde fino, gradiente y sombra suave."""
    if rect.width <= 0 or rect.height <= 0:
        return
    rgba = tuple(pygame.Color(color))
    border = tuple(pygame.Color(border_color)) if border_color is not None else None
    panel = _panel_surface(rect.size, rgba, radius, border, gradient, shadow)
    surface.blit(panel, rect)


def draw_vector_icon(
    surface: pygame.Surface,
    name: str,
    center: tuple[int, int],
    color: pygame.Color,
    size: int = 18,
    width: int = 2,
) -> None:
    """Dibuja iconos básicos con líneas/polígonos; no depende de glifos."""
    x, y = center
    half = size // 2
    if name == "play":
        pygame.draw.polygon(
            surface,
            color,
            ((x - half // 2, y - half), (x - half // 2, y + half), (x + half, y)),
        )
    elif name == "pause":
        pygame.draw.rect(surface, color, (x - half, y - half, width + 1, size))
        pygame.draw.rect(surface, color, (x + 2, y - half, width + 1, size))
    elif name == "reset":
        pygame.draw.arc(surface, color, (x - half, y - half, size, size), 0.5, 5.5, width)
        pygame.draw.polygon(
            surface, color, ((x + half, y - half), (x + half, y - 2), (x + 2, y - half))
        )
    elif name == "download":
        pygame.draw.line(surface, color, (x, y - half), (x, y + 2), width)
        pygame.draw.lines(
            surface, color, False, ((x - 5, y - 2), (x, y + 3), (x + 5, y - 2)), width
        )
        pygame.draw.line(surface, color, (x - half, y + half), (x + half, y + half), width)
    elif name == "info":
        pygame.draw.circle(surface, color, center, half, width)
        pygame.draw.circle(surface, color, (x, y - 5), 1)
        pygame.draw.line(surface, color, (x, y - 1), (x, y + 5), width)
    elif name == "dice":
        pygame.draw.rect(surface, color, (x - half, y - half, size, size), width, border_radius=3)
        for dx, dy in ((-4, -4), (4, -4), (0, 0), (-4, 4), (4, 4)):
            pygame.draw.circle(surface, color, (x + dx, y + dy), 1)
    elif name == "check":
        pygame.draw.lines(
            surface,
            color,
            False,
            ((x - half, y), (x - 2, y + half // 2), (x + half, y - half // 2)),
            width,
        )
    elif name == "door":
        pygame.draw.rect(
            surface,
            color,
            pygame.Rect(x - half + 2, y - half, size - 4, size),
            width,
            border_radius=2,
        )
        pygame.draw.circle(surface, color, (x + half - 5, y), max(1, width))
    elif name == "register":
        pygame.draw.rect(
            surface,
            color,
            pygame.Rect(x - half, y - 3, size, half + 3),
            width,
            border_radius=2,
        )
        pygame.draw.line(surface, color, (x - half + 2, y + 3), (x + half - 2, y + 3), width)
        pygame.draw.rect(surface, color, pygame.Rect(x - 3, y - half, 7, 5), border_radius=1)
    elif name in {"previous", "next"}:
        direction = -1 if name == "previous" else 1
        pygame.draw.lines(
            surface,
            color,
            False,
            (
                (x - direction * 3, y - half),
                (x + direction * 3, y),
                (x - direction * 3, y + half),
            ),
            width,
        )


def load_fonts(config: dict[str, Any], project_root: str | Path) -> dict[str, pygame.font.Font]:
    """Carga cuatro niveles Inter desde assets/fonts, con fallback del sistema."""
    sizes = config["app"].get("typography", {})
    font_path = Path(project_root) / "assets" / "fonts" / "Inter.ttf"
    font_source = str(font_path) if font_path.is_file() else None
    dimensions = {
        "title": int(sizes.get("title", 32)),
        "subtitle": int(sizes.get("subtitle", 22)),
        "body": int(sizes.get("body", 16)),
        "label": int(sizes.get("label", 12)),
    }
    if font_source:
        brand_font = pygame.font.Font(font_source, dimensions["subtitle"])
        brand_font.set_bold(True)
        tab_font = pygame.font.Font(font_source, dimensions["body"])
        tab_font.set_bold(True)
        return {
            "title": pygame.font.Font(font_source, dimensions["title"]),
            "subtitle": pygame.font.Font(font_source, dimensions["subtitle"]),
            "body": pygame.font.Font(font_source, dimensions["body"]),
            "label": pygame.font.Font(font_source, dimensions["label"]),
            "small": pygame.font.Font(font_source, dimensions["label"]),
            "tab": tab_font,
            "brand": brand_font,
        }
    return {
        "title": pygame.font.SysFont("segoeui", dimensions["title"], bold=True),
        "subtitle": pygame.font.SysFont("segoeui", dimensions["subtitle"], bold=True),
        "body": pygame.font.SysFont("segoeui", dimensions["body"]),
        "label": pygame.font.SysFont("segoeui", dimensions["label"]),
        "small": pygame.font.SysFont("segoeui", dimensions["label"]),
        "tab": pygame.font.SysFont("segoeui", dimensions["body"], bold=True),
        "brand": pygame.font.SysFont("segoeui", dimensions["subtitle"], bold=True),
    }


def contrast_ratio(foreground: pygame.Color, background: pygame.Color) -> float:
    """Calcula la relación de contraste WCAG para dos colores opacos."""

    def luminance(color: pygame.Color) -> float:
        channels = []
        for component in (color.r, color.g, color.b):
            value = component / 255.0
            channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    lighter = max(luminance(foreground), luminance(background))
    darker = min(luminance(foreground), luminance(background))
    return (lighter + 0.05) / (darker + 0.05)
