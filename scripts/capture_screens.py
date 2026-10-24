"""Captura las cinco pestañas en resoluciones de rúbrica y mide el render de frames."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import render_frame
from app.tabs import TABS
from app.tabs.home_tab import HomeTab
from app.tabs.task_a_tab import TaskATab
from app.tabs.task_b_tab import TaskBTab
from app.tabs.task_c_tab import TaskCTab
from app.tabs.task_d_tab import TaskDTab
from app.ui.theme import Theme, load_fonts
from app.ui.widgets import Tabs
from src.urbanstyle.utils.config import load_config

RESOLUTIONS = ((1280, 720), (1366, 768), (1920, 1080))


def _build_tabs(config: dict):
    tabs = Tabs([tab.label for tab in TABS])
    home = HomeTab(config, PROJECT_ROOT)
    task_a = TaskATab(config, PROJECT_ROOT)
    task_b = TaskBTab(config, PROJECT_ROOT)
    task_c = TaskCTab(config, PROJECT_ROOT)
    task_d = TaskDTab(config, PROJECT_ROOT)
    return tabs, home, task_a, task_b, task_c, task_d


def _render(screen, config, fonts, theme, app_tabs, active: int) -> None:
    app_tabs[0].active = active
    render_frame(
        screen,
        config,
        fonts,
        theme,
        *app_tabs,
        delta_seconds=1 / 60,
    )


def capture_screens(frames: int = 30) -> tuple[list[dict], list[Path]]:
    """Guarda capturas PNG y retorna mediciones de render por pestaña/resolución."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    output_dir = PROJECT_ROOT / "outputs" / "screens"
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = PROJECT_ROOT / "docs"
    pygame.init()
    fonts = load_fonts(config, PROJECT_ROOT)
    theme = Theme.from_config(config)
    app_tabs = _build_tabs(config)
    names = ("inicio", "task_a", "task_b", "task_c", "task_d")
    measurements = []
    capture_paths = []
    try:
        for width, height in RESOLUTIONS:
            screen = pygame.display.set_mode((width, height))
            resolution_dir = output_dir / f"{width}x{height}"
            resolution_dir.mkdir(parents=True, exist_ok=True)
            for index, name in enumerate(names):
                _render(screen, config, fonts, theme, app_tabs, index)
                path = resolution_dir / f"{name}.png"
                pygame.image.save(screen, path)
                capture_paths.append(path)
                if width == 1280 and height == 720:
                    docs_path = docs_dir / f"captura_{name}.png"
                    pygame.image.save(screen, docs_path)

                for _ in range(3):
                    _render(screen, config, fonts, theme, app_tabs, index)
                durations = []
                for _ in range(frames):
                    started = time.perf_counter()
                    _render(screen, config, fonts, theme, app_tabs, index)
                    pygame.display.flip()
                    durations.append((time.perf_counter() - started) * 1000)
                ordered = sorted(durations)
                p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
                mean_ms = statistics.fmean(durations)
                measurements.append(
                    {
                        "resolution": f"{width}x{height}",
                        "tab": name,
                        "frames": frames,
                        "mean_ms": round(mean_ms, 3),
                        "p95_ms": round(p95, 3),
                        "p95_render_fps": round(1000 / p95, 1) if p95 else 0.0,
                        "within_60fps_budget": p95 <= 1000 / 60,
                    }
                )
    finally:
        app_tabs[1].close()
        app_tabs[5].close()
        pygame.quit()

    report_path = output_dir / "performance.csv"
    with report_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=measurements[0].keys())
        writer.writeheader()
        writer.writerows(measurements)
    capture_paths.append(report_path)
    return measurements, capture_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30, help="Frames medidos por pestaña")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames debe ser positivo.")
    try:
        measurements, paths = capture_screens(args.frames)
    except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError, pygame.error) as error:
        print(f"No se pudieron capturar las pantallas: {error}", file=sys.stderr)
        return 1
    for row in measurements:
        print(
            f"{row['resolution']} {row['tab']}: p95={row['p95_ms']:.2f} ms "
            f"({row['p95_render_fps']:.1f} FPS)"
        )
    print(f"Capturas y mediciones: {PROJECT_ROOT / 'outputs' / 'screens'}")
    print(f"README captures: {PROJECT_ROOT / 'docs'}")
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())