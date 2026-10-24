"""Pruebas de selección de series y ventanas en la pestaña B."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from app.tabs.task_b_tab import TaskBTab
from src.urbanstyle.utils.config import PROJECT_ROOT, load_config


def test_series_chips_and_context_windows_filter_loaded_artifacts() -> None:
    pygame.init()
    pygame.display.set_mode((1280, 720))
    tab = TaskBTab(load_config(), PROJECT_ROOT)
    tab.layout_controls(pygame.Rect(36, 174, 1208, 510))

    assert len(tab.series_chips) == 6
    assert tab.selected_window == 8
    assert tab._comparison_summary() == (4, 6)

    tab.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONUP,
            button=1,
            pos=tab.series_chips[5].rect.center,
        )
    )
    assert tab.selected_series == "Electronics · M"
    _, predictions, _ = tab._series_frames()
    assert not predictions.empty
    assert set(predictions["gender"]) == {"Male"}
    assert set(predictions["context_weeks"]) == {8}

    tab.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=tab.baseline_toggle.rect.center)
    )
    assert not tab.show_baseline
    tab.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=tab.zoom_toggle.rect.center)
    )
    assert tab.zoom_test

    tab.handle_event(
        pygame.event.Event(
            pygame.MOUSEBUTTONUP,
            button=1,
            pos=tab.context_control.rects[2].center,
        )
    )
    assert tab.selected_window == 12
    _, predictions, _ = tab._series_frames()
    assert set(predictions["context_weeks"]) == {12}
    pygame.quit()
