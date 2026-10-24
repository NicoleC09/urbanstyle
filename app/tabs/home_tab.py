"""Dashboard Inicio para ejecutar y resumir el escenario UrbanStyle."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pygame
from PIL import Image
from src.urbanstyle.integration.pipeline import ScenarioResult, run_scenario
from src.urbanstyle.utils.config import resolve_path

from app.ui.layout import Track, WeightedLayout
from app.ui.theme import Theme, draw_panel, draw_vector_icon
from app.ui.widgets import Button, ProgressBar, Toast, Tooltip

LOGGER = logging.getLogger("urbanstyle.app.home")
STEP_LABELS = ("Demanda", "Precio", "Tienda", "Campaña")


class HomeTab:
    """Panel de escenario con progreso, KPIs y campaña final."""

    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.result: ScenarioResult | None = None
        self.poster_path: Path | None = None
        self.error: str | None = None
        self.loading = False
        self.progress_stage = 0
        self.progress_visual = 0.0
        self.poster_opacity = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="urbanstyle-scenario")
        self._future: Future[ScenarioResult] | None = None
        self.button = Button(
            pygame.Rect(0, 0, 220, 48),
            "Ejecutar escenario UrbanStyle",
            self.run_scenario,
            kind="primary",
            icon="play",
        )
        self.download_button = Button(
            pygame.Rect(0, 0, 180, 34), "Descargar PNG", self.download_poster, icon="download"
        )
        self.info_rect = pygame.Rect(0, 0, 18, 18)
        self.info_tooltip = Tooltip(
            self.info_rect,
            "La imagen generada es una ilustración Fashion-MNIST, no una foto del producto.",
        )
        self.toast: Toast | None = None
        self.status = "Listo para ejecutar"
        self.scenario_rect = pygame.Rect(0, 0, 0, 0)
        self.steps_rect = pygame.Rect(0, 0, 0, 0)
        self.poster_rect = pygame.Rect(0, 0, 0, 0)
        self.stepper_rect = pygame.Rect(0, 0, 0, 0)
        self.kpi_rects: list[pygame.Rect] = []
        self.sparkline_rect = pygame.Rect(0, 0, 0, 0)
        self.heatmap_rect = pygame.Rect(0, 0, 0, 0)
        self._sparkline: list[float] = []
        self._poster_surface: pygame.Surface | None = None
        self._poster_cache_path: Path | None = None
        self._poster_cache_size: tuple[int, int] | None = None
        self._animated_values: dict[str, float] = {}
        self._load_last_result()

    def _load_last_result(self) -> None:
        """Recupera el último resumen guardado sin volver a ejecutar modelos."""
        summary_path = (
            resolve_path(self.config["paths"]["outputs"]) / "integration" / "scenario_summary.json"
        )
        if not summary_path.is_file():
            return
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            data["congestion_cell"] = tuple(data["congestion_cell"])
            data["traffic_heatmap"] = tuple(tuple(cell) for cell in data.get("traffic_heatmap", ()))
            self.result = ScenarioResult(**data)
            self.poster_path = Path(self.result.poster_path)
            self.status = "Último escenario completado"
            self.progress_stage = 4
            self.progress_visual = 4.0
            self._load_sparkline(self.result.category)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            LOGGER.warning("No se pudo cargar el último escenario: %s", error)

    def layout(self, content_rect: pygame.Rect) -> None:
        """Distribuye tarjetas, stepper y zona de visualizaciones."""
        viewport = pygame.Rect(
            content_rect.x + 20,
            content_rect.y + 96,
            content_rect.width - 40,
            max(0, content_rect.height - 112),
        )
        self.scenario_rect, self.steps_rect, self.poster_rect = WeightedLayout(gap=14).columns(
            viewport, [Track(0.9, 250), Track(1.5, 420), Track(0.9, 250)]
        )
        self.button.rect = pygame.Rect(
            self.scenario_rect.x + 16,
            self.scenario_rect.y + 68,
            self.scenario_rect.width - 32,
            48,
        )
        self.button.enabled = not self.loading
        self.button.label = (
            "Ejecutando escenario…" if self.loading else "Ejecutar escenario UrbanStyle"
        )
        self.stepper_rect = pygame.Rect(
            self.steps_rect.x + 14, self.steps_rect.y + 14, self.steps_rect.width - 28, 62
        )
        inner = self.steps_rect.inflate(-24, -92)
        gap = 8
        card_width = (inner.width - gap) // 2
        card_height = min(54, max(44, (inner.height - 3 * 6 - 70) // 4))
        self.kpi_rects = [
            pygame.Rect(
                inner.x + (index % 2) * (card_width + gap),
                inner.y + (index // 2) * (card_height + 6),
                card_width,
                card_height,
            )
            for index in range(7)
        ]
        chart_y = inner.y + 4 * (card_height + 6)
        chart_height = max(34, self.steps_rect.bottom - 14 - chart_y)
        self.sparkline_rect = pygame.Rect(inner.x, chart_y, inner.width * 2 // 3 - 4, chart_height)
        self.heatmap_rect = pygame.Rect(
            self.sparkline_rect.right + 8,
            chart_y,
            max(42, inner.right - self.sparkline_rect.right - 8),
            chart_height,
        )
        self.info_rect = pygame.Rect(self.poster_rect.right - 36, self.poster_rect.y + 14, 18, 18)
        self.download_button.rect = pygame.Rect(
            self.poster_rect.x + 16,
            self.poster_rect.bottom - 42,
            self.poster_rect.width - 32,
            34,
        )

    def _set_progress(self, stage: int) -> None:
        self.progress_stage = max(self.progress_stage, min(4, int(stage)))

    def _run_in_worker(self) -> ScenarioResult:
        return run_scenario(self.config, progress_callback=self._set_progress)

    def run_scenario(self) -> None:
        """Envía inferencia y simulaciones al worker, nunca entrena modelos."""
        if self.loading:
            return
        self.loading = True
        self.error = None
        self.progress_stage = 0
        self.progress_visual = 0.0
        self.poster_opacity = 0
        self.status = "Preparando demanda…"
        self.button.enabled = False
        self._future = self._executor.submit(self._run_in_worker)

    def update(self, delta_seconds: float = 1 / 60) -> None:
        """Actualiza animaciones y recoge el worker sin bloquear el bucle Pygame."""
        self.progress_visual += (self.progress_stage - self.progress_visual) * min(
            1.0, delta_seconds * 5
        )
        if self.loading:
            self.status = (
                "Preparando demanda…"
                if self.progress_stage == 0
                else "Demanda lista · evaluando precio…"
                if self.progress_stage == 1
                else "Precio listo · simulando tienda…"
                if self.progress_stage == 2
                else "Tienda lista · creando campaña…"
            )
        if self.result:
            self.poster_opacity = min(255, self.poster_opacity + round(520 * delta_seconds))
            self._animate_values(delta_seconds)
        if not self._future or not self._future.done():
            return
        try:
            self.result = self._future.result()
            self.poster_path = Path(self.result.poster_path)
            self._poster_surface = None
            self.progress_stage = 4
            self.status = "Escenario completado"
            self._load_sparkline(self.result.category)
            self._animated_values.clear()
        except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as error:
            self.error = str(error)
            self.status = "No se pudo completar el escenario"
            LOGGER.exception("Falló el escenario integrado")
        finally:
            self.loading = False
            self.button.enabled = True
            self._future = None

    def _animate_values(self, delta_seconds: float) -> None:
        if self.result is None:
            return
        targets = {
            "Pronóstico": self.result.forecast_units,
            "Precio del agente": self.result.price,
            "Descuento": float(self.result.discount),
            "Ingreso": self.result.revenue,
            "Congestión": float(self.result.max_congestion),
            "Permanencia": self.result.dwell_steps,
        }
        for key, target in targets.items():
            current = self._animated_values.get(key, 0.0)
            self._animated_values[key] = min(
                float(target), current + max(0.01, abs(float(target)) * delta_seconds * 2.6)
            )

    def _load_sparkline(self, category: str) -> None:
        path = resolve_path(self.config["paths"]["processed_data"]) / "weekly_sales.csv"
        try:
            with path.open(encoding="utf-8", newline="") as source:
                rows = csv.DictReader(source)
                records = [row for row in rows if row.get("Product Category") == category]
            by_week: dict[str, float] = {}
            for row in records:
                by_week[row.get("week", "")] = by_week.get(row.get("week", ""), 0.0) + float(
                    row.get("units", 0.0)
                )
            self._sparkline = [value for _, value in sorted(by_week.items())][-16:]
        except (OSError, ValueError, TypeError):
            self._sparkline = []

    def download_poster(self) -> None:
        if not self.result or not self.poster_path or not self.poster_path.is_file():
            self.toast = Toast("Ejecuta el escenario para crear la campaña")
            return
        destination_dir = resolve_path(self.config["paths"]["outputs"]) / "posters"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "urbanstyle_scenario.png"
        try:
            shutil.copy2(self.poster_path, destination)
            self.toast = Toast(f"PNG guardado: {destination.name}")
        except OSError as error:
            LOGGER.exception("No se pudo guardar el póster: %s", error)
            self.toast = Toast("No se pudo descargar el póster")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _draw_label(surface, text, position, font, color) -> None:
        surface.blit(font.render(text, True, color), position)

    def handle_event(self, event: pygame.event.Event) -> None:
        self.button.handle_event(event)
        self.download_button.handle_event(event)

    def _draw_stepper(self, surface, fonts, theme: Theme) -> None:
        rect = self.stepper_rect
        node_y = rect.y + 16
        left = rect.x + 28
        right = rect.right - 28
        centers = [(round(left + index * (right - left) / 3), node_y) for index in range(4)]
        progress = min(4.0, max(0.0, self.progress_visual))
        for index in range(3):
            pygame.draw.line(
                surface,
                theme.primary if progress >= index + 1 else theme.border,
                centers[index],
                centers[index + 1],
                3,
            )
        for index, (label, center) in enumerate(zip(STEP_LABELS, centers, strict=True)):
            complete = progress >= index + 1 or (self.result is not None and index < 4)
            active = self.loading and int(progress) == index
            color = theme.success if complete else theme.primary if active else theme.border
            pygame.draw.circle(surface, color, center, 12)
            if complete:
                draw_vector_icon(surface, "check", center, theme.text, 14, 2)
            else:
                number = fonts["label"].render(str(index + 1), True, theme.text)
                surface.blit(number, number.get_rect(center=center))
            label_color = theme.text if complete or active else theme.muted
            label_surface = fonts["label"].render(label, True, label_color)
            surface.blit(label_surface, label_surface.get_rect(center=(center[0], node_y + 28)))
        ProgressBar(pygame.Rect(rect.x + 4, rect.bottom - 3, rect.width - 8, 3), progress / 4).draw(
            surface, theme
        )

    def _kpi_values(self) -> list[tuple[str, str, float | None, str]]:
        if self.result is None:
            return [
                (label, "—", None, trend)
                for label, trend in (
                    ("Categoría", ""),
                    ("Pronóstico", ""),
                    ("Precio del agente", ""),
                    ("Descuento", ""),
                    ("Ingreso", ""),
                    ("Congestión", ""),
                    ("Permanencia", ""),
                )
            ]
        result = self.result
        trend = "↑" if result.action == "Subir" else "↓" if result.action == "Bajar" else "→"
        return [
            ("Categoría", result.category, None, ""),
            (
                "Pronóstico",
                f"{self._animated_values.get('Pronóstico', 0.0):.1f} u",
                self._animated_values.get("Pronóstico", 0.0),
                "↑",
            ),
            (
                "Precio del agente",
                f"{self._animated_values.get('Precio del agente', 0.0):.2f}",
                self._animated_values.get("Precio del agente", 0.0),
                trend,
            ),
            (
                "Descuento",
                f"{round(self._animated_values.get('Descuento', 0.0))}%",
                self._animated_values.get("Descuento", 0.0),
                "↓" if result.discount else "→",
            ),
            (
                "Ingreso",
                f"{self._animated_values.get('Ingreso', 0.0):,.0f}",
                self._animated_values.get("Ingreso", 0.0),
                "↑",
            ),
            (
                "Congestión",
                f"{round(self._animated_values.get('Congestión', 0.0))}",
                self._animated_values.get("Congestión", 0.0),
                "↑",
            ),
            (
                "Permanencia",
                f"{self._animated_values.get('Permanencia', 0.0):.1f} pasos",
                self._animated_values.get("Permanencia", 0.0),
                "→",
            ),
        ]

    def _draw_kpis(self, surface, fonts, theme: Theme) -> None:
        values = self._kpi_values()
        for rect, (label, value, _, trend) in zip(self.kpi_rects, values, strict=False):
            draw_panel(surface, rect, theme.panel, radius=8, border_color=theme.border)
            self._draw_label(
                surface,
                label.upper(),
                (rect.x + 10, rect.y + 6),
                fonts["label"],
                theme.muted,
            )
            value_font = fonts["body"] if len(value) > 12 else fonts["subtitle"]
            rendered = value_font.render(value, True, theme.text)
            surface.blit(rendered, (rect.x + 10, rect.bottom - rendered.get_height() - 4))
            if trend:
                center = (rect.right - 14, rect.y + 15)
                if trend == "→":
                    pygame.draw.line(
                        surface,
                        theme.sky,
                        (center[0] - 5, center[1]),
                        (center[0] + 5, center[1]),
                        2,
                    )
                    pygame.draw.lines(
                        surface,
                        theme.sky,
                        False,
                        (
                            (center[0] + 1, center[1] - 4),
                            (center[0] + 5, center[1]),
                            (center[0] + 1, center[1] + 4),
                        ),
                        2,
                    )
                else:
                    direction = -1 if trend == "↑" else 1
                    y = center[1]
                    pygame.draw.polygon(
                        surface,
                        theme.success if trend == "↑" else theme.primary,
                        (
                            (center[0] - 5, y + direction * 3),
                            (center[0] + 5, y + direction * 3),
                            (center[0], y - direction * 4),
                        ),
                    )

    def _draw_sparkline(self, surface, fonts, theme: Theme) -> None:
        rect = self.sparkline_rect
        draw_panel(surface, rect, theme.panel, radius=8, border_color=theme.border)
        self._draw_label(surface, "DEMANDA", (rect.x + 7, rect.y + 4), fonts["label"], theme.muted)
        values = self._sparkline
        if len(values) < 2:
            self._draw_label(
                surface,
                "—",
                (rect.x + 8, rect.centery - 5),
                fonts["label"],
                theme.muted,
            )
            return
        graph = pygame.Rect(rect.x + 8, rect.y + 20, rect.width - 16, max(8, rect.height - 26))
        low, high = min(values), max(values)
        scale = max(1e-6, high - low)
        points = [
            (
                graph.x + round(index * graph.width / (len(values) - 1)),
                graph.bottom - round((value - low) * graph.height / scale),
            )
            for index, value in enumerate(values)
        ]
        pygame.draw.lines(surface, theme.primary, False, points, 2)
        pygame.draw.circle(surface, theme.sky, points[-1], 3)

    def _draw_heatmap(self, surface, fonts, theme: Theme) -> None:
        rect = self.heatmap_rect
        draw_panel(surface, rect, theme.panel, radius=8, border_color=theme.border)
        self._draw_label(surface, "TIENDA", (rect.x + 6, rect.y + 4), fonts["label"], theme.muted)
        columns, rows = 6, 4
        data = self.result.traffic_heatmap if self.result else ()
        cells: dict[tuple[int, int], int] = {}
        for x, y, value in data:
            bin_position = (min(columns - 1, x * columns // 12), min(rows - 1, y * rows // 8))
            cells[bin_position] = cells.get(bin_position, 0) + value
        max_value = max(cells.values(), default=1)
        fallback_hotspot = None
        if self.result and not cells:
            fallback_hotspot = (
                min(columns - 1, max(0, self.result.congestion_cell[0] // 2)),
                min(rows - 1, max(0, self.result.congestion_cell[1] // 2)),
            )
        area = pygame.Rect(rect.x + 7, rect.y + 20, rect.width - 14, rect.height - 26)
        cell_width = max(2, (area.width - 5 * 2) // columns)
        cell_height = max(2, (area.height - 3 * 2) // rows)
        for row in range(rows):
            for column in range(columns):
                ratio = cells.get((column, row), 0) / max_value if data else 0
                if fallback_hotspot == (column, row):
                    ratio = 1.0
                color = theme.panel.lerp(theme.primary, min(1.0, ratio))
                pygame.draw.rect(
                    surface,
                    color,
                    (
                        area.x + column * (cell_width + 2),
                        area.y + row * (cell_height + 2),
                        cell_width,
                        cell_height,
                    ),
                    border_radius=2,
                )

    def _draw_poster(self, surface, fonts, theme: Theme) -> None:
        rect = self.poster_rect
        info_button = Button(self.info_rect, "", icon="info", kind="ghost")
        info_button.draw(surface, fonts["label"], theme)
        self.info_tooltip.rect = self.info_rect.copy()
        self.info_tooltip.draw(surface, fonts["label"], theme)
        if self.poster_path and self.poster_path.is_file():
            image_rect = pygame.Rect(rect.x + 16, rect.y + 44, rect.width - 32, rect.height - 96)
            if (
                self._poster_cache_path != self.poster_path
                or self._poster_cache_size != image_rect.size
                or self._poster_surface is None
            ):
                try:
                    with Image.open(self.poster_path) as source:
                        image = source.convert("RGB")
                        image.thumbnail(image_rect.size, Image.Resampling.LANCZOS)
                        self._poster_surface = pygame.image.fromstring(
                            image.tobytes(), image.size, "RGB"
                        )
                    self._poster_cache_path = self.poster_path
                    self._poster_cache_size = image_rect.size
                except (OSError, pygame.error) as error:
                    LOGGER.warning("No se pudo preparar el póster %s: %s", self.poster_path, error)
                    self._poster_surface = None
            if self._poster_surface:
                poster = self._poster_surface.copy()
                poster.set_alpha(self.poster_opacity)
                surface.blit(poster, poster.get_rect(center=image_rect.center))
        else:
            self._draw_label(
                surface,
                "CAMPAÑA",
                (rect.x + 16, rect.y + 18),
                fonts["label"],
                theme.muted,
            )
            self._draw_label(
                surface,
                "El póster aparecerá aquí" if self.result is None else "Campaña no disponible",
                (rect.x + 16, rect.y + 52),
                fonts["body"],
                theme.muted,
            )
        self.download_button.enabled = bool(self.poster_path and self.poster_path.is_file())
        self.download_button.draw(surface, fonts["label"], theme)

    def draw(self, surface: pygame.Surface, content_rect: pygame.Rect, fonts, theme: Theme) -> None:
        self.layout(content_rect)
        self._draw_label(
            surface,
            "URBANSTYLE  /  ESCENARIO",
            (content_rect.x + 28, content_rect.y + 18),
            fonts["label"],
            theme.sky,
        )
        self._draw_label(
            surface,
            "De la demanda a la campaña",
            (content_rect.x + 28, content_rect.y + 38),
            fonts["title"],
            theme.text,
        )
        for rect in (self.scenario_rect, self.steps_rect, self.poster_rect):
            draw_panel(surface, rect, theme.elevated, radius=10, border_color=theme.border)

        left_x = self.scenario_rect.x + 16
        self._draw_label(
            surface,
            "ESCENARIO",
            (left_x, self.scenario_rect.y + 16),
            fonts["label"],
            theme.muted,
        )
        self.button.draw(surface, fonts["body"], theme)
        status = self.error or self.status
        self._draw_label(
            surface,
            status[:42],
            (left_x, self.button.rect.bottom + 16),
            fonts["label"],
            theme.error if self.error else theme.sky,
        )
        if self.result:
            self._draw_label(
                surface,
                f"{self.result.action} · {self.result.pricing_algorithm}",
                (left_x, self.button.rect.bottom + 42),
                fonts["body"],
                theme.text,
            )
            self._draw_label(
                surface,
                f"{self.result.layout} · {self.result.customers} clientes",
                (left_x, self.button.rect.bottom + 68),
                fonts["label"],
                theme.muted,
            )
        else:
            self._draw_label(
                surface,
                "Modelos existentes · sin entrenar",
                (left_x, self.button.rect.bottom + 46),
                fonts["label"],
                theme.muted,
            )
        if self.loading:
            self._draw_stepper(surface, fonts, theme)
        else:
            self._draw_stepper(surface, fonts, theme)

        self._draw_kpis(surface, fonts, theme)
        self._draw_sparkline(surface, fonts, theme)
        self._draw_heatmap(surface, fonts, theme)
        self._draw_poster(surface, fonts, theme)
        if self.toast:
            self.toast.draw(surface, fonts["label"], theme)
