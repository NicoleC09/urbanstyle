"""Resultados y métricas de los algoritmos de pathfinding."""

from __future__ import annotations

from dataclasses import dataclass

from .grid import Position


@dataclass(frozen=True)
class SearchResult:
    """Camino y coste de una búsqueda sobre la grilla."""

    path: tuple[Position, ...]
    explored_nodes: int
    elapsed_ms: float

    @property
    def path_length(self) -> int:
        """Cantidad de movimientos, excluyendo la celda inicial."""
        return max(0, len(self.path) - 1)
