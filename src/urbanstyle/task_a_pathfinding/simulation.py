"""Simulación reproducible de recorridos y congestión de clientes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from .grid import Position
from .layouts import StoreLayout
from .metrics import SearchResult
from .search import astar, bfs

Algorithm = Literal["BFS", "A*"]


@dataclass(frozen=True)
class CustomerRoute:
    """Ruta acumulada de un cliente y sus secciones visitadas."""

    positions: tuple[Position, ...]
    visited_categories: tuple[str, ...]


@dataclass(frozen=True)
class SimulationResult:
    """Resumen de rutas, tráfico y coste de búsqueda de una simulación."""

    algorithm: Algorithm
    customers: tuple[CustomerRoute, ...]
    traffic: dict[Position, int]
    searches: tuple[SearchResult, ...]
    dwell_steps: float

    @property
    def explored_nodes(self) -> int:
        """Nodos expandidos sumados en todos los tramos de clientes."""
        return sum(result.explored_nodes for result in self.searches)

    @property
    def mean_path_length(self) -> float:
        """Longitud media de un tramo entre puntos de interés."""
        return (
            sum(result.path_length for result in self.searches) / len(self.searches)
            if self.searches
            else 0.0
        )

    @property
    def mean_search_ms(self) -> float:
        """Tiempo medio de búsqueda medido en milisegundos."""
        return (
            sum(result.elapsed_ms for result in self.searches) / len(self.searches)
            if self.searches
            else 0.0
        )

    @property
    def max_congestion(self) -> int:
        """Máximo número de visitas acumuladas en una celda."""
        return max(self.traffic.values(), default=0)

    @property
    def top_congested(self) -> tuple[tuple[Position, int], ...]:
        """Las cinco celdas con más tráfico, ordenadas de forma estable."""
        return tuple(
            sorted(self.traffic.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))[:5]
        )


def _weighted_categories(
    rng: random.Random, proportions: dict[str, float], count: int
) -> tuple[str, ...]:
    remaining = {key: max(0.0, float(value)) for key, value in proportions.items()}
    selected: list[str] = []
    while remaining and len(selected) < count:
        total = sum(remaining.values())
        if total <= 0:
            choice = rng.choice(sorted(remaining))
        else:
            threshold = rng.random() * total
            choice = next(
                category
                for category, weight in remaining.items()
                if (threshold := threshold - weight) < 0
            )
        selected.append(choice)
        del remaining[choice]
    return tuple(selected)


def simulate_customers(
    layout: StoreLayout,
    algorithm: Algorithm,
    customer_count: int,
    category_proportions: dict[str, float],
    seed: int = 42,
) -> SimulationResult:
    """Simula clientes que recorren secciones ponderadas, caja y salida."""
    if customer_count < 1:
        raise ValueError("El número de clientes debe ser al menos 1.")
    if algorithm not in {"BFS", "A*"}:
        raise ValueError(f"Algoritmo desconocido: {algorithm}")
    missing = set(layout.sections) - set(category_proportions)
    if missing:
        raise ValueError("Faltan proporciones para: " + ", ".join(sorted(missing)))
    search = bfs if algorithm == "BFS" else astar
    rng = random.Random(seed)
    customers: list[CustomerRoute] = []
    traffic: dict[Position, int] = {}
    searches: list[SearchResult] = []
    total_steps = 0

    for _ in range(customer_count):
        visit_count = rng.randint(1, min(3, len(layout.sections)))
        categories = _weighted_categories(rng, category_proportions, visit_count)
        destinations = [layout.sections[category] for category in categories]
        destinations.extend((layout.checkout, layout.exit))
        current = layout.entrance
        route: list[Position] = [current]
        for destination in destinations:
            result = search(layout.grid, current, destination)
            searches.append(result)
            if not result.path:
                raise RuntimeError(
                    f"No hay camino en el layout {layout.name}: {current} -> {destination}."
                )
            route.extend(result.path[1:])
            current = destination
        for position in route:
            traffic[position] = traffic.get(position, 0) + 1
        total_steps += len(route) - 1
        customers.append(CustomerRoute(tuple(route), categories))

    return SimulationResult(
        algorithm=algorithm,
        customers=tuple(customers),
        traffic=traffic,
        searches=tuple(searches),
        dwell_steps=total_steps / customer_count,
    )
