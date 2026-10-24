"""Regresiones de bounds para los layouts principales de la aplicación."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest
from app.tabs import TABS
from app.tabs.home_tab import HomeTab
from app.tabs.task_a_tab import TaskATab
from app.tabs.task_b_tab import TaskBTab
from app.tabs.task_c_tab import TaskCTab
from app.tabs.task_d_tab import TaskDTab
from app.ui.theme import Theme, contrast_ratio
from app.ui.widgets import Button, Chip, Dropdown, SegmentedControl, Slider, Tabs, Toggle
from src.urbanstyle.task_a_pathfinding.layouts import available_layouts
from src.urbanstyle.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def pygame_headless():
    pygame.init()
    pygame.display.set_mode((1280, 720))
    yield
    pygame.quit()


def _inside(parent: pygame.Rect, children: list[pygame.Rect]) -> None:
    for child in children:
        assert parent.contains(child), f"{child} desborda {parent}"


def _make_home() -> HomeTab:
    tab = HomeTab.__new__(HomeTab)
    tab.button = Button(pygame.Rect(0, 0, 240, 44), "Ejecutar")
    tab.download_button = Button(pygame.Rect(0, 0, 180, 34), "Descargar")
    tab.loading = False
    return tab


def _make_task_a() -> TaskATab:
    tab = TaskATab.__new__(TaskATab)
    tab.layouts = available_layouts(12, 8)
    tab.layout_index = 0
    tab.algorithm_control = SegmentedControl(pygame.Rect(0, 0, 184, 38), ("BFS", "A*"))
    tab.layout_thumb_rects = [pygame.Rect(0, 0, 80, 54) for _ in tab.layouts]
    tab.customer_slider = Slider(pygame.Rect(0, 0, 112, 4), 4, 60, 18)
    tab.speed_slider = Slider(pygame.Rect(0, 0, 112, 4), 2, 18, 8)
    tab.heatmap_toggle = Toggle(pygame.Rect(0, 0, 42, 24), True)
    tab.control_rect = pygame.Rect(0, 0, 0, 0)
    tab.grid_rect = pygame.Rect(0, 0, 0, 0)
    tab.metrics_rect = pygame.Rect(0, 0, 0, 0)
    tab.buttons = {
        "run": Button(pygame.Rect(0, 0, 58, 34), "Iniciar"),
        "pause": Button(pygame.Rect(0, 0, 58, 34), "Pausa"),
        "reset": Button(pygame.Rect(0, 0, 58, 34), "Reset"),
    }
    return tab


def _make_task_b() -> TaskBTab:
    tab = TaskBTab.__new__(TaskBTab)
    tab.series_labels = [
        "Beauty · F",
        "Beauty · M",
        "Clothing · F",
        "Clothing · M",
        "Electronics · F",
        "Electronics · M",
    ]
    tab.series_chips = [
        Chip(pygame.Rect(0, 0, 60, 28), label, index == 0)
        for index, label in enumerate(tab.series_labels)
    ]
    tab.context_windows = [4, 8, 12]
    tab.context_control = SegmentedControl(pygame.Rect(0, 0, 150, 28), ("4", "8", "12"), 1)
    tab.baseline_toggle = Toggle(pygame.Rect(0, 0, 42, 24), True)
    tab.zoom_toggle = Toggle(pygame.Rect(0, 0, 42, 24), False)
    tab.embedding_info_rect = pygame.Rect(0, 0, 16, 16)
    tab.buttons = {
        "export": Button(pygame.Rect(0, 0, 160, 36), "Exportar CSV"),
    }
    tab.chart_rect = pygame.Rect(0, 0, 0, 0)
    tab.metrics_rect = pygame.Rect(0, 0, 0, 0)
    tab.control_rect = pygame.Rect(0, 0, 0, 0)
    return tab


def _make_task_c() -> TaskCTab:
    tab = TaskCTab.__new__(TaskCTab)
    tab.category_control = SegmentedControl(
        pygame.Rect(0, 0, 212, 38), ("Beauty", "Clothing", "Electronics")
    )
    tab.algorithm_control = SegmentedControl(pygame.Rect(0, 0, 212, 38), ("PPO", "DQN"))
    tab.price_slider = Slider(pygame.Rect(0, 0, 175, 4), 0.6, 4.0, 1.0)
    tab.speed_slider = Slider(pygame.Rect(0, 0, 175, 4), 0.25, 2.0, 1.0)
    tab.chart_metric_control = SegmentedControl(
        pygame.Rect(0, 0, 156, 28), ("Margen", "Ingreso")
    )
    tab.buttons = {
        "run": Button(pygame.Rect(0, 0, 130, 38), "Ejecutar episodio"),
        "pause": Button(pygame.Rect(0, 0, 100, 38), "Pausar"),
        "reset": Button(pygame.Rect(0, 0, 100, 38), "Reiniciar"),
    }
    tab.control_rect = pygame.Rect(0, 0, 0, 0)
    tab.chart_rect = pygame.Rect(0, 0, 0, 0)
    tab.metrics_rect = pygame.Rect(0, 0, 0, 0)
    return tab


def _make_task_d() -> TaskDTab:
    tab = TaskDTab.__new__(TaskDTab)
    tab.category_control = SegmentedControl(
        pygame.Rect(0, 0, 280, 34), ("Beauty", "Clothing", "Electronics")
    )
    tab.classes = [
        "Camiseta",
        "Pantalón",
        "Suéter",
        "Vestido",
        "Abrigo",
        "Sandalia",
        "Camisa",
        "Zapatilla",
        "Bolso",
        "Botín",
    ]
    tab.class_chips = [Chip(pygame.Rect(0, 0, 90, 34), name) for name in tab.classes]
    tab.seed_rect = pygame.Rect(0, 0, 200, 30)
    tab.dice_button = Button(pygame.Rect(0, 0, 38, 30), "Dado")
    tab.count_slider = Slider(pygame.Rect(0, 0, 150, 4), 1, 9, 4)
    tab.variety_slider = Slider(pygame.Rect(0, 0, 150, 4), 0.2, 1.2, 0.8)
    tab.discount_slider = Slider(pygame.Rect(0, 0, 150, 4), 5, 70, 20)
    tab.template_control = SegmentedControl(pygame.Rect(0, 0, 150, 30), ("Noche", "Hielo", "Eléc."))
    tab.style_control = SegmentedControl(pygame.Rect(0, 0, 100, 30), ("Neón", "Pixel"))
    tab.buttons = {
        key: Button(pygame.Rect(0, 0, 155, 38), key)
        for key in ("generate", "random", "poster", "download", "large", "save_grid")
    }
    tab.history_buttons = [
        Button(pygame.Rect(0, 0, 90, 30), label) for label in ("Muestras", "Curvas", "Calidad")
    ]
    tab.epoch_slider = Slider(pygame.Rect(0, 0, 240, 4), 1, 80, 80)
    tab.control_rect = pygame.Rect(0, 0, 0, 0)
    tab.gallery_rect = pygame.Rect(0, 0, 0, 0)
    tab.preview_rect = pygame.Rect(0, 0, 0, 0)
    return tab


@pytest.mark.parametrize("size", [(1280, 720), (1366, 768), (1920, 1080)])
def test_tab_layouts_and_controls_stay_inside_window(size: tuple[int, int]) -> None:
    width, height = size
    window = pygame.Rect(0, 0, width, height)
    content = pygame.Rect(36, 174, width - 72, height - 210)

    tabs = Tabs([tab.label for tab in TABS])
    tabs.layout(pygame.Rect(36, 86, width - 72, 54))
    _inside(window, tabs.rects)

    home = _make_home()
    home.layout(content)
    _inside(
        window,
        [
            home.scenario_rect,
            home.steps_rect,
            home.poster_rect,
            home.button.rect,
            home.download_button.rect,
            home.stepper_rect,
            *home.kpi_rects,
            home.sparkline_rect,
            home.heatmap_rect,
        ],
    )
    _inside(content, [home.scenario_rect, home.steps_rect, home.poster_rect])
    _inside(home.scenario_rect, [home.button.rect])
    _inside(home.poster_rect, [home.download_button.rect])
    _inside(
        home.steps_rect,
        [home.stepper_rect, *home.kpi_rects, home.sparkline_rect, home.heatmap_rect],
    )

    task_a = _make_task_a()
    task_a.layout_controls(content)
    _inside(
        window,
        [
            task_a.control_rect,
            task_a.grid_rect,
            task_a.metrics_rect,
            task_a.algorithm_control.rect,
            *task_a.layout_thumb_rects,
            task_a.customer_slider.rect,
            task_a.speed_slider.rect,
            task_a.heatmap_toggle.rect,
            *(button.rect for button in task_a.buttons.values()),
        ],
    )
    _inside(content, [task_a.control_rect, task_a.grid_rect, task_a.metrics_rect])
    _inside(
        task_a.control_rect,
        [task_a.algorithm_control.rect, *task_a.layout_thumb_rects, task_a.heatmap_toggle.rect],
    )
    _inside(task_a.control_rect, [button.rect for button in task_a.buttons.values()])

    task_b = _make_task_b()
    task_b.layout_controls(content)
    _inside(
        window,
        [
            task_b.control_rect,
            task_b.chart_rect,
            task_b.metrics_rect,
            *(chip.rect for chip in task_b.series_chips),
            task_b.context_control.rect,
            task_b.baseline_toggle.rect,
            task_b.zoom_toggle.rect,
            task_b.embedding_info_rect,
            *(button.rect for button in task_b.buttons.values()),
        ],
    )
    _inside(content, [task_b.control_rect, task_b.chart_rect, task_b.metrics_rect])
    _inside(
        task_b.control_rect,
        [
            *(chip.rect for chip in task_b.series_chips),
            task_b.context_control.rect,
            task_b.baseline_toggle.rect,
            task_b.zoom_toggle.rect,
            task_b.embedding_info_rect,
            *[b.rect for b in task_b.buttons.values()],
        ],
    )

    task_c = _make_task_c()
    task_c.layout_controls(content)
    _inside(
        window,
        [
            task_c.control_rect,
            task_c.chart_rect,
            task_c.metrics_rect,
            task_c.category_control.rect,
            task_c.algorithm_control.rect,
            task_c.price_slider.rect,
            task_c.speed_slider.rect,
            task_c.chart_metric_control.rect,
            *(button.rect for button in task_c.buttons.values()),
        ],
    )
    _inside(content, [task_c.control_rect, task_c.chart_rect, task_c.metrics_rect])
    _inside(
        task_c.control_rect,
        [
            task_c.category_control.rect,
            task_c.algorithm_control.rect,
            task_c.price_slider.rect,
            task_c.speed_slider.rect,
            *(button.rect for button in task_c.buttons.values()),
        ],
    )
    _inside(task_c.chart_rect, [task_c.chart_metric_control.rect])

    task_d = _make_task_d()
    task_d.layout_controls(content)
    _inside(
        window,
        [
            task_d.control_rect,
            task_d.preview_rect,
            task_d.gallery_rect,
            task_d.category_control.rect,
            *(chip.rect for chip in task_d.class_chips),
            task_d.seed_rect,
            task_d.dice_button.rect,
            task_d.count_slider.rect,
            task_d.variety_slider.rect,
            task_d.discount_slider.rect,
            task_d.template_control.rect,
            task_d.style_control.rect,
            *(button.rect for button in task_d.buttons.values()),
            *(button.rect for button in task_d.history_buttons),
            task_d.epoch_slider.rect,
        ],
    )
    _inside(content, [task_d.control_rect, task_d.preview_rect, task_d.gallery_rect])
    _inside(
        task_d.control_rect,
        [
            task_d.category_control.rect,
            *(chip.rect for chip in task_d.class_chips),
            task_d.seed_rect,
            task_d.dice_button.rect,
            task_d.count_slider.rect,
            task_d.variety_slider.rect,
            task_d.discount_slider.rect,
            task_d.template_control.rect,
            task_d.style_control.rect,
            task_d.buttons["poster"].rect,
        ],
    )
    _inside(
        task_d.preview_rect,
        [task_d.buttons[key].rect for key in ("generate", "random", "download", "large")],
    )
    _inside(
        task_d.gallery_rect,
        [*(button.rect for button in task_d.history_buttons), task_d.epoch_slider.rect],
    )


def test_long_dropdown_scrolls_and_selects_visible_rows() -> None:
    dropdown = Dropdown(
        pygame.Rect(20, 20, 160, 38), tuple(f"Clase {index}" for index in range(10))
    )
    surface = pygame.Surface((300, 400))
    theme = Theme.from_config(load_config(PROJECT_ROOT / "config.yaml"))
    font = pygame.font.Font(str(PROJECT_ROOT / "assets/fonts/Inter.ttf"), 12)
    dropdown.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(40, 30)))
    dropdown.draw(surface, font, theme)
    pointer = (40, dropdown.rect.bottom + 20)
    pygame.mouse.set_pos(pointer)
    dropdown.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=-2, pos=pointer))

    assert dropdown.scroll_offset == 2
    selected_row = dropdown._option_rect(3)
    dropdown.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONUP,
            button=1,
            pos=selected_row.center,
        )
    )
    assert dropdown.selected == 3
    assert not dropdown.open


def test_primary_text_tokens_meet_wcag_aa_contrast() -> None:
    pygame.init()
    theme = Theme.from_config(load_config(PROJECT_ROOT / "config.yaml"))

    for background in (theme.background, theme.panel, theme.elevated):
        assert contrast_ratio(theme.text, background) >= 4.5
        assert contrast_ratio(theme.muted, background) >= 4.5
