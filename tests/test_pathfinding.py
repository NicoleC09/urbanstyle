"""Pruebas de optimalidad y validez de BFS/A*."""

import pytest
from src.urbanstyle.task_a_pathfinding.grid import Grid
from src.urbanstyle.task_a_pathfinding.layouts import available_layouts
from src.urbanstyle.task_a_pathfinding.search import astar, bfs
from src.urbanstyle.task_a_pathfinding.simulation import simulate_customers


@pytest.fixture
def store_grid() -> Grid:
    return Grid(
        12,
        8,
        frozenset(
            (x, y)
            for x in (3, 4, 7, 8)
            for y in range(1, 7)
            if (x, y) not in {(3, 4), (7, 2), (8, 5)}
        ),
    )


def assert_valid_path(grid: Grid, path: tuple[tuple[int, int], ...]) -> None:
    assert path
    assert all(grid.is_walkable(cell) for cell in path)
    assert all(
        abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        for first, second in zip(path, path[1:], strict=False)
    )


def test_bfs_and_astar_find_optimal_valid_paths(store_grid: Grid) -> None:
    start, goal = (0, 0), (11, 7)

    breadth_first = bfs(store_grid, start, goal)
    a_star = astar(store_grid, start, goal)

    assert_valid_path(store_grid, breadth_first.path)
    assert_valid_path(store_grid, a_star.path)
    assert breadth_first.path[0] == a_star.path[0] == start
    assert breadth_first.path[-1] == a_star.path[-1] == goal
    assert breadth_first.path_length == a_star.path_length
    assert a_star.explored_nodes <= breadth_first.explored_nodes


def test_search_reports_no_path_when_goal_is_enclosed() -> None:
    grid = Grid(3, 3, frozenset({(1, 0), (0, 1), (2, 1), (1, 2)}))

    assert bfs(grid, (0, 0), (2, 2)).path == ()
    assert astar(grid, (0, 0), (2, 2)).path == ()


def test_search_rejects_blocked_endpoint() -> None:
    grid = Grid(3, 3, frozenset({(1, 1)}))

    with pytest.raises(ValueError, match="destino"):
        astar(grid, (0, 0), (1, 1))


@pytest.mark.parametrize("layout", available_layouts(), ids=lambda layout: layout.name)
def test_predefined_layouts_are_navigable(layout) -> None:
    result = simulate_customers(
        layout,
        "A*",
        customer_count=6,
        category_proportions={"Beauty": 0.3, "Clothing": 0.4, "Electronics": 0.3},
        seed=7,
    )

    assert len(result.customers) == 6
    assert result.max_congestion > 0
    assert result.dwell_steps > 0


def test_simulation_is_reproducible_and_has_category_visits() -> None:
    layout = available_layouts()[0]
    proportions = {"Beauty": 0.7, "Clothing": 0.2, "Electronics": 0.1}

    first = simulate_customers(layout, "BFS", 30, proportions, seed=99)
    second = simulate_customers(layout, "BFS", 30, proportions, seed=99)

    assert first.traffic == second.traffic
    assert [customer.visited_categories for customer in first.customers] == [
        customer.visited_categories for customer in second.customers
    ]
    assert sum("Beauty" in customer.visited_categories for customer in first.customers) > sum(
        "Electronics" in customer.visited_categories for customer in first.customers
    )
