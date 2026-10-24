"""Representación de grillas transitables para la simulación de tienda."""

from __future__ import annotations

from dataclasses import dataclass, field

Position = tuple[int, int]


@dataclass(frozen=True)
class Grid:
    """Grilla rectangular con celdas bloqueadas y movimiento ortogonal."""

    width: int
    height: int
    blocked: frozenset[Position] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("La grilla debe tener dimensiones positivas.")
        outside = [cell for cell in self.blocked if not self.in_bounds(cell)]
        if outside:
            raise ValueError(f"Hay obstáculos fuera de la grilla: {outside}")

    def in_bounds(self, position: Position) -> bool:
        """Comprueba que una coordenada pertenezca a la grilla."""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, position: Position) -> bool:
        """Indica si la coordenada está dentro y no es un obstáculo."""
        return self.in_bounds(position) and position not in self.blocked

    def neighbors(self, position: Position) -> tuple[Position, ...]:
        """Devuelve vecinos transitables en orden estable: arriba, derecha, abajo, izquierda."""
        x, y = position
        candidates = ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y))
        return tuple(cell for cell in candidates if self.is_walkable(cell))
