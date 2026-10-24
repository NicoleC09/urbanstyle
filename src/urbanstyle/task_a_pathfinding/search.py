"""Implementaciones desde cero de BFS y A* con heurística Manhattan."""

from __future__ import annotations

import heapq
import time
from collections import deque
from itertools import count

from .grid import Grid, Position
from .metrics import SearchResult


def _validate_endpoints(grid: Grid, start: Position, goal: Position) -> None:
    if not grid.is_walkable(start):
        raise ValueError(f"El origen no es transitable: {start}")
    if not grid.is_walkable(goal):
        raise ValueError(f"El destino no es transitable: {goal}")


def _reconstruct_path(
    parents: dict[Position, Position], start: Position, goal: Position
) -> tuple[Position, ...]:
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    return tuple(path)


def bfs(grid: Grid, start: Position, goal: Position) -> SearchResult:
    """Encuentra un camino mínimo con búsqueda en anchura."""
    _validate_endpoints(grid, start, goal)
    started = time.perf_counter()
    frontier = deque([start])
    parents: dict[Position, Position] = {}
    discovered = {start}
    explored = 0

    while frontier:
        current = frontier.popleft()
        explored += 1
        if current == goal:
            return SearchResult(
                _reconstruct_path(parents, start, goal),
                explored,
                (time.perf_counter() - started) * 1000,
            )
        for neighbor in grid.neighbors(current):
            if neighbor not in discovered:
                discovered.add(neighbor)
                parents[neighbor] = current
                frontier.append(neighbor)

    return SearchResult((), explored, (time.perf_counter() - started) * 1000)


def manhattan(first: Position, second: Position) -> int:
    """Distancia Manhattan para movimientos ortogonales."""
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def astar(grid: Grid, start: Position, goal: Position) -> SearchResult:
    """Encuentra un camino mínimo con A* y heurística Manhattan."""
    _validate_endpoints(grid, start, goal)
    started = time.perf_counter()
    sequence = count()
    frontier: list[tuple[int, int, int, Position]] = []
    heapq.heappush(frontier, (manhattan(start, goal), 0, next(sequence), start))
    parents: dict[Position, Position] = {}
    costs = {start: 0}
    explored = 0

    while frontier:
        _, cost, _, current = heapq.heappop(frontier)
        if cost != costs.get(current):
            continue
        explored += 1
        if current == goal:
            return SearchResult(
                _reconstruct_path(parents, start, goal),
                explored,
                (time.perf_counter() - started) * 1000,
            )
        for neighbor in grid.neighbors(current):
            next_cost = cost + 1
            if next_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = next_cost
                parents[neighbor] = current
                priority = next_cost + manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, next_cost, next(sequence), neighbor))

    return SearchResult((), explored, (time.perf_counter() - started) * 1000)
