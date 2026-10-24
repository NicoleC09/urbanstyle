"""Layouts predefinidos y escalables de la tienda UrbanStyle."""

from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid, Position


@dataclass(frozen=True)
class StoreLayout:
    """Distribución de una tienda, con entrada, caja y secciones transitables."""

    name: str
    grid: Grid
    entrance: Position
    checkout: Position
    exit: Position
    sections: dict[str, Position]


def _scale(position: Position, width: int, height: int) -> Position:
    """Escala una coordenada definida sobre la grilla de referencia 12x8."""
    return (
        round(position[0] * (width - 1) / 11),
        round(position[1] * (height - 1) / 7),
    )


def create_layout(name: str, width: int = 12, height: int = 8) -> StoreLayout:
    """Crea un layout clásico, abierto o en serpentina."""
    if width < 8 or height < 6:
        raise ValueError("El layout mínimo permitido es de 8x6.")
    if name not in {"Clásico", "Abierto", "Serpentina"}:
        raise ValueError(f"Layout desconocido: {name}")

    obstacles: set[Position] = set()
    if name == "Clásico":
        obstacles = {
            _scale((x, y), width, height)
            for x in (3, 4, 7, 8)
            for y in range(1, 7)
            if (x, y) not in {(3, 4), (7, 2), (8, 5)}
        }
    elif name == "Serpentina":
        obstacles = {
            _scale((x, y), width, height)
            for x in (2, 4, 6, 8, 10)
            for y in range(1, 7)
            if (x + y) % 4 != 0
        }
    else:
        obstacles = {
            _scale((x, y), width, height)
            for x, y in ((3, 2), (3, 3), (3, 4), (7, 2), (7, 3), (7, 4))
        }

    entrance = _scale((0, 3), width, height)
    checkout = _scale((10, 3), width, height)
    exit_position = _scale((11, 3), width, height)
    section_points = {
        "Beauty": _scale((2, 1), width, height),
        "Clothing": _scale((5, 6), width, height),
        "Electronics": _scale((9, 1), width, height),
    }
    protected = {entrance, checkout, exit_position, *section_points.values()}
    obstacles.difference_update(protected)
    return StoreLayout(
        name,
        Grid(width, height, frozenset(obstacles)),
        entrance,
        checkout,
        exit_position,
        section_points,
    )


def available_layouts(width: int = 12, height: int = 8) -> tuple[StoreLayout, ...]:
    """Devuelve los tres layouts listos para seleccionar."""
    return tuple(
        create_layout(name, width, height) for name in ("Clásico", "Abierto", "Serpentina")
    )
