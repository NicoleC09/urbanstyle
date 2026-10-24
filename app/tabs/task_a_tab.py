"""Vista interactiva para pathfinding y simulación de tienda."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pygame
from src.urbanstyle.task_a_pathfinding.layouts import StoreLayout, available_layouts
from src.urbanstyle.task_a_pathfinding.simulation import SimulationResult, simulate_customers
from src.urbanstyle.utils.seed import seed_everything

from app.ui.layout import Track, WeightedLayout
from app.ui.theme import Theme, draw_panel, draw_vector_icon
from app.ui.widgets import Button, SegmentedControl, Slider, Toggle

LOGGER = logging.getLogger("urbanstyle.app.task_a")


class TaskATab:
    """Controles y visualizaciones de la simulación de recorridos."""

    def __init__(self, config: dict, project_root: Path):
        task_config = config["task_a"]
        self.config = config
        self.project_root = project_root
        self.layouts = available_layouts(task_config["grid_width"], task_config["grid_height"])
        self.layout_index = 0
        self.algorithm_index = 1
        self.algorithms = ("BFS", "A*")
        self.algorithm_control = SegmentedControl(
            pygame.Rect(0, 0, 184, 38), self.algorithms, selected=self.algorithm_index
        )
        self.layout_thumb_rects: list[pygame.Rect] = []
        self.customer_slider = Slider(
            pygame.Rect(0, 0, 160, 4),
            task_config["customers_min"],
            task_config["customers_max"],
            task_config["default_customers"],
        )
        self.speed_slider = Slider(
            pygame.Rect(0, 0, 160, 4),
            task_config["speed_min"],
            task_config["speed_max"],
            task_config["default_speed"],
        )
        self.heatmap_enabled = True
        self.heatmap_toggle = Toggle(pygame.Rect(0, 0, 42, 24), True)
        self.running = False
        self.paused = False
        self.elapsed_seconds = 0.0
        self.seed = int(config["project"]["seed"])
        self.proportions, self.data_warning = self._load_proportions(config, project_root)
        self.selected_result: SimulationResult | None = None
        self.comparison: dict[str, SimulationResult] = {}
        self.customer_progress: list[float] = []
        self.selected_customer_index: int | None = None
        self.control_rect = pygame.Rect(0, 0, 0, 0)
        self.grid_rect = pygame.Rect(0, 0, 0, 0)
        self.metrics_rect = pygame.Rect(0, 0, 0, 0)
        self.cell_size = 1
        self.buttons = {
            "run": Button(pygame.Rect(0, 0, 58, 34), "Iniciar", self.start, kind="primary"),
            "pause": Button(pygame.Rect(0, 0, 58, 34), "Pausa", self.toggle_pause),
            "reset": Button(pygame.Rect(0, 0, 58, 34), "Reset", self.reset, kind="ghost"),
        }
        self._set_button_states()

    @staticmethod
    def _load_proportions(config: dict, project_root: Path) -> tuple[dict[str, float], str | None]:
        categories = config["data"]["categories"]
        processed_dir = Path(config["paths"]["processed_data"])
        raw_path = Path(config["paths"]["raw_data"])
        if not processed_dir.is_absolute():
            processed_dir = project_root / processed_dir
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path
        processed_path = processed_dir / "calibration.json"
        try:
            if processed_path.is_file():
                calibration = json.loads(processed_path.read_text(encoding="utf-8"))
                values = calibration.get("category_sales_share", {})
                if values:
                    return {
                        category: float(values.get(category, 0.0)) for category in categories
                    }, None
            if raw_path.is_file():
                frame = pd.read_csv(raw_path)
                if not {"Product Category", "Total Amount"}.issubset(frame.columns):
                    raise ValueError("El CSV no contiene Product Category y Total Amount.")
                sales = frame.groupby("Product Category")["Total Amount"].sum()
                total = float(sales.sum())
                if total <= 0:
                    raise ValueError("El CSV no contiene ventas positivas.")
                return {
                    category: float(sales.get(category, 0.0) / total) for category in categories
                }, None
        except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as error:
            LOGGER.warning("No se pudieron cargar proporciones de venta: %s", error)
            return {}, str(error)
        return (
            {},
            f"Falta el CSV en {config['paths']['raw_data']}. Ejecuta la preparación de datos.",
        )

    @property
    def layout(self) -> StoreLayout:
        """Layout activo."""
        return self.layouts[self.layout_index]

    @property
    def algorithm(self) -> str:
        """Algoritmo que se animará en la grilla."""
        return self.algorithms[self.algorithm_index]

    def _set_button_states(self) -> None:
        self.buttons["pause"].label = "Reanudar" if self.paused else "Pausar"
        self.buttons["run"].enabled = bool(self.proportions)
        self.buttons["pause"].enabled = self.running

    def layout_controls(self, content_rect: pygame.Rect) -> None:
        """Coloca los paneles y controles según el tamaño de la ventana."""
        inner_y = content_rect.y + 132
        panel_rect = pygame.Rect(
            content_rect.x + 22,
            inner_y,
            content_rect.width - 44,
            max(0, content_rect.height - 154),
        )
        self.control_rect, self.grid_rect, self.metrics_rect = WeightedLayout(gap=16).columns(
            panel_rect,
            [Track(1, 212), Track(2, 420), Track(1, 238)],
        )

        controls_x = self.control_rect.x + 14
        self.algorithm_control.rect = pygame.Rect(
            controls_x, self.control_rect.y + 44, self.control_rect.width - 28, 34
        )
        thumb_gap = 6
        thumb_width = (self.control_rect.width - 28 - 2 * thumb_gap) // 3
        self.layout_thumb_rects = [
            pygame.Rect(
                controls_x + index * (thumb_width + thumb_gap),
                self.control_rect.y + 99,
                thumb_width,
                54,
            )
            for index in range(len(self.layouts))
        ]
        self.customer_slider.rect = pygame.Rect(controls_x, self.control_rect.y + 196, 112, 4)
        self.speed_slider.rect = pygame.Rect(controls_x, self.control_rect.y + 252, 112, 4)
        self.heatmap_toggle.rect.topleft = (self.control_rect.right - 58, self.control_rect.y + 288)
        button_y = self.control_rect.bottom - 42
        for index, key in enumerate(("run", "pause", "reset")):
            self.buttons[key].rect = pygame.Rect(
                controls_x + index * 63,
                button_y,
                min(58, self.control_rect.width - 28 - index * 63),
                34,
            )
        grid_area_width = self.grid_rect.width - 32
        grid_area_height = self.grid_rect.height - 94
        self.cell_size = max(
            1,
            min(
                64,
                grid_area_width // self.layout.grid.width,
                grid_area_height // self.layout.grid.height,
            ),
        )
        grid_pixel_width = self.cell_size * self.layout.grid.width
        grid_pixel_height = self.cell_size * self.layout.grid.height
        self.grid_origin = (
            self.grid_rect.centerx - grid_pixel_width // 2,
            self.grid_rect.y + 42 + max(0, (grid_area_height - grid_pixel_height) // 2),
        )

    def handle_event(self, event: pygame.event.Event) -> None:
        """Enruta eventos a controles cuando la pestaña está activa."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.toggle_pause()
            elif event.key == pygame.K_r:
                self.reset()
        if self.algorithm_control.handle_event(event):
            self.cycle_algorithm(self.algorithm_control.selected)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for index, rect in enumerate(self.layout_thumb_rects):
                if rect.collidepoint(event.pos) and index != self.layout_index:
                    self.layout_index = index
                    self.reset()
                    break
            self._select_customer_at(event.pos)
        if self.heatmap_toggle.handle_event(event):
            self.heatmap_enabled = self.heatmap_toggle.value
        self.customer_slider.handle_event(event)
        self.speed_slider.handle_event(event)
        for button in self.buttons.values():
            button.handle_event(event)

    def start(self) -> None:
        """Calcula la simulación comparativa y comienza la animación."""
        if not self.proportions:
            return
        seed_everything(self.seed)
        count = round(self.customer_slider.value)
        self.comparison = {
            algorithm: simulate_customers(
                self.layout,
                algorithm,
                count,
                self.proportions,
                seed=self.seed,
            )
            for algorithm in self.algorithms
        }
        self.selected_result = self.comparison[self.algorithm]
        self.customer_progress = [0.0] * len(self.selected_result.customers)
        self.elapsed_seconds = 0.0
        self.running = True
        self.paused = False
        self._set_button_states()

    def toggle_pause(self) -> None:
        """Pausa o reanuda el avance de clientes."""
        if self.running:
            self.paused = not self.paused
            self._set_button_states()

    def reset(self) -> None:
        """Detiene y limpia la simulación actual."""
        self.running = False
        self.paused = False
        self.elapsed_seconds = 0.0
        self.selected_result = None
        self.comparison = {}
        self.customer_progress = []
        self.selected_customer_index = None
        self._set_button_states()

    def cycle_algorithm(self, selected: int | None = None) -> None:
        """Alterna entre BFS y A*, o fija la selección del control."""
        self.algorithm_index = (
            (self.algorithm_index + 1) % len(self.algorithms) if selected is None else selected
        )
        self.algorithm_control.selected = self.algorithm_index
        self.selected_result = self.comparison.get(self.algorithm)
        if self.running and self.selected_result:
            self.customer_progress = [0.0] * len(self.selected_result.customers)
        self._set_button_states()

    def cycle_layout(self) -> None:
        """Alterna layout y limpia cualquier simulación anterior."""
        self.layout_index = (self.layout_index + 1) % len(self.layouts)
        self.reset()

    def toggle_heatmap(self) -> None:
        """Activa o desactiva la capa de calor."""
        self.heatmap_enabled = not self.heatmap_enabled
        self.heatmap_toggle.value = self.heatmap_enabled

    def _select_customer_at(self, point: tuple[int, int]) -> None:
        if not self.selected_result:
            return
        for index in range(len(self.selected_result.customers)):
            center = self._customer_center(index)
            if (center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2 <= max(
                12, self.cell_size // 2
            ) ** 2:
                self.selected_customer_index = (
                    None if self.selected_customer_index == index else index
                )
                return

    def _customer_center(self, index: int) -> tuple[int, int]:
        customer = self.selected_result.customers[index]
        progress = self.customer_progress[index]
        lower = int(progress)
        fraction = progress - lower
        route = customer.positions
        first = route[min(lower, len(route) - 1)]
        second = route[min(lower + 1, len(route) - 1)]
        return (
            round(
                self.grid_origin[0]
                + (first[0] + (second[0] - first[0]) * fraction + 0.5) * self.cell_size
            ),
            round(
                self.grid_origin[1]
                + (first[1] + (second[1] - first[1]) * fraction + 0.5) * self.cell_size
            ),
        )

    def update(self, delta_seconds: float) -> None:
        """Interpola la posición de cada cliente a la velocidad configurada."""
        if not self.running or self.paused or not self.selected_result:
            return
        self.elapsed_seconds += delta_seconds
        self.customer_progress = [
            min(
                float(len(customer.positions) - 1),
                progress + self.speed_slider.value * delta_seconds,
            )
            for customer, progress in zip(
                self.selected_result.customers, self.customer_progress, strict=True
            )
        ]
        if all(
            progress >= len(customer.positions) - 1
            for customer, progress in zip(
                self.selected_result.customers, self.customer_progress, strict=True
            )
        ):
            self.running = False

    def _draw_label(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: pygame.Color,
    ) -> None:
        surface.blit(font.render(text, True, color), position)

    def draw(
        self,
        surface: pygame.Surface,
        content_rect: pygame.Rect,
        fonts: dict[str, pygame.font.Font],
        theme: Theme,
    ) -> None:
        """Dibuja controles, mapa, heatmap, clientes y resultados comparados."""
        self.layout_controls(content_rect)
        self._draw_label(
            surface,
            "RECORRIDOS DE CLIENTES",
            (content_rect.x + 32, content_rect.y + 30),
            fonts["small"],
            theme.accent,
        )
        self._draw_label(
            surface,
            "BFS y A* sobre el mismo espacio comercial",
            (content_rect.x + 32, content_rect.y + 55),
            fonts["title"],
            theme.foreground,
        )
        for panel in (self.control_rect, self.grid_rect, self.metrics_rect):
            draw_panel(surface, panel, theme.elevated, radius=10, border_color=theme.border)
        controls_x = self.control_rect.x + 14
        self._draw_label(
            surface,
            "RECORRIDO",
            (controls_x, self.control_rect.y + 14),
            fonts["small"],
            theme.sky,
        )
        self.algorithm_control.draw(surface, fonts["small"], theme)
        self._draw_label(
            surface,
            "LAYOUT",
            (controls_x, self.control_rect.y + 82),
            fonts["small"],
            theme.muted,
        )
        self._draw_layout_thumbnails(surface, fonts, theme)
        for button in self.buttons.values():
            button.draw(surface, fonts["small"], theme)
        self._draw_label(
            surface,
            f"CLIENTES  {round(self.customer_slider.value)}",
            (controls_x, self.control_rect.y + 171),
            fonts["small"],
            theme.muted,
        )
        self.customer_slider.draw(surface, fonts["small"], theme, label="")
        self._draw_label(
            surface,
            f"VELOCIDAD  {self.speed_slider.value:.0f}×",
            (controls_x, self.control_rect.y + 227),
            fonts["small"],
            theme.muted,
        )
        self.speed_slider.draw(surface, fonts["small"], theme, label="")
        self._draw_label(
            surface,
            "HEATMAP",
            (controls_x, self.control_rect.y + 293),
            fonts["small"],
            theme.muted,
        )
        self.heatmap_toggle.draw(surface, theme)
        selected_label = (
            f"CLIENTE {self.selected_customer_index + 1} · RUTA RESALTADA"
            if self.selected_customer_index is not None
            else "SELECCIONA UN CLIENTE"
        )
        self._draw_label(
            surface,
            f"{self.layout.name.upper()}  ·  {selected_label}",
            (self.grid_rect.x + 16, self.grid_rect.y + 14),
            fonts["small"],
            theme.foreground,
        )
        self._draw_grid(surface, fonts, theme)
        self._draw_metrics(surface, fonts, theme)

    def _draw_layout_thumbnails(self, surface, fonts, theme: Theme) -> None:
        for index, (layout, rect) in enumerate(
            zip(self.layouts, self.layout_thumb_rects, strict=True)
        ):
            selected = index == self.layout_index
            draw_panel(
                surface,
                rect,
                theme.deep if selected else theme.panel,
                radius=7,
                border_color=theme.primary if selected else theme.border,
            )
            grid = pygame.Rect(rect.x + 6, rect.y + 5, rect.width - 12, 26)
            cell_w = max(1, grid.width // layout.grid.width)
            cell_h = max(1, grid.height // layout.grid.height)
            for row in range(layout.grid.height):
                for column in range(layout.grid.width):
                    blocked = (column, row) in layout.grid.blocked
                    color = theme.muted if blocked else theme.elevated
                    pygame.draw.rect(
                        surface,
                        color,
                        (
                            grid.x + column * cell_w,
                            grid.y + row * cell_h,
                            max(1, cell_w - 1),
                            max(1, cell_h - 1),
                        ),
                        border_radius=1,
                    )
            label = ("Clásico", "Abierto", "Serpentina")[index]
            rendered = fonts["label"].render(label, True, theme.text if selected else theme.muted)
            surface.blit(rendered, rendered.get_rect(center=(rect.centerx, rect.bottom - 9)))

    def _cell_rect(self, position: tuple[int, int]) -> pygame.Rect:
        x, y = position
        return pygame.Rect(
            self.grid_origin[0] + x * self.cell_size,
            self.grid_origin[1] + y * self.cell_size,
            self.cell_size - 2,
            self.cell_size - 2,
        )

    @staticmethod
    def _heat_color(value: int, maximum: int) -> tuple[int, int, int]:
        """Interpola desde azul profundo hasta cielo según el tráfico relativo."""
        ratio = value / max(1, maximum)
        cold = (18, 48, 107)
        hot = (124, 196, 255)
        return tuple(
            round(first + ratio * (second - first)) for first, second in zip(cold, hot, strict=True)
        )

    def _draw_grid(
        self,
        surface: pygame.Surface,
        fonts: dict[str, pygame.font.Font],
        theme: Theme,
    ) -> None:
        panel_clip = surface.get_clip()
        surface.set_clip(self.grid_rect)
        traffic = self.selected_result.traffic if self.selected_result else {}
        max_traffic = max(traffic.values(), default=1)
        section_colors = {
            "Beauty": theme.category_beauty,
            "Clothing": theme.category_clothing,
            "Electronics": theme.category_electronics,
        }
        for y in range(self.layout.grid.height):
            for x in range(self.layout.grid.width):
                position = (x, y)
                rect = self._cell_rect(position)
                if position in self.layout.grid.blocked:
                    color = (theme.elevated.r, theme.elevated.g, theme.elevated.b)
                else:
                    color = (
                        round(theme.deep.r * 0.72 + theme.panel.r * 0.28),
                        round(theme.deep.g * 0.72 + theme.panel.g * 0.28),
                        round(theme.deep.b * 0.72 + theme.panel.b * 0.28),
                    )
                if self.heatmap_enabled and position in traffic:
                    heat_color = self._heat_color(traffic[position], max_traffic)
                    color = tuple(
                        round(base * 0.55 + heat * 0.45)
                        for base, heat in zip(color, heat_color, strict=True)
                    )
                pygame.draw.rect(surface, color, rect, border_radius=3)

        grid_width = self.cell_size * self.layout.grid.width
        grid_height = self.cell_size * self.layout.grid.height
        boundary = pygame.Rect(
            self.grid_origin[0] - 3,
            self.grid_origin[1] - 3,
            grid_width + 2,
            grid_height + 2,
        )
        pygame.draw.rect(surface, theme.border, boundary, width=3, border_radius=5)
        pygame.draw.line(surface, theme.muted, boundary.topleft, boundary.topright, 3)
        pygame.draw.line(surface, theme.muted, boundary.bottomleft, boundary.bottomright, 3)

        blocked = set(self.layout.grid.blocked)
        for row in range(self.layout.grid.height):
            column = 0
            while column < self.layout.grid.width:
                if (column, row) not in blocked:
                    column += 1
                    continue
                start = column
                while column + 1 < self.layout.grid.width and (column + 1, row) in blocked:
                    column += 1
                first = self._cell_rect((start, row))
                last = self._cell_rect((column, row))
                shelf = pygame.Rect(
                    first.left + 2,
                    first.top + 3,
                    last.right - first.left - 2,
                    max(4, self.cell_size - 8),
                )
                pygame.draw.rect(surface, theme.elevated, shelf, border_radius=6)
                pygame.draw.line(
                    surface,
                    theme.border,
                    (shelf.left + 4, shelf.top + 3),
                    (shelf.right - 4, shelf.top + 3),
                    2,
                )
                column += 1

        section_tags: list[tuple[str, pygame.Color, pygame.Rect]] = []
        for category, position in self.layout.sections.items():
            rect = self._cell_rect(position).inflate(-3, -3)
            pygame.draw.rect(surface, section_colors[category], rect, border_radius=6)
            label_font = pygame.font.Font(None, max(13, min(18, self.cell_size // 2)))
            label = label_font.render(category, True, theme.background)
            tag_width = min(self.grid_rect.width - 12, label.get_width() + 12)
            tag_height = label.get_height() + 6
            tag_x = min(
                max(rect.centerx - tag_width // 2, self.grid_rect.left + 4),
                self.grid_rect.right - tag_width - 4,
            )
            tag_y = min(rect.top, self.grid_rect.bottom - 44 - tag_height)
            section_tags.append(
                (
                    category,
                    section_colors[category],
                    pygame.Rect(tag_x, tag_y, tag_width, tag_height),
                )
            )

        entrance_rect = self._cell_rect(self.layout.entrance).inflate(-4, -4)
        pygame.draw.rect(surface, theme.sky, entrance_rect, border_radius=5)
        draw_vector_icon(
            surface,
            "door",
            (entrance_rect.centerx, entrance_rect.centery - 2),
            theme.background,
            16,
            2,
        )
        checkout_rect = self._cell_rect(self.layout.checkout).inflate(-4, -4)
        pygame.draw.rect(surface, theme.success, checkout_rect, border_radius=5)
        draw_vector_icon(
            surface,
            "register",
            (checkout_rect.centerx, checkout_rect.centery - 3),
            theme.background,
            14,
            2,
        )
        exit_rect = self._cell_rect(self.layout.exit).inflate(-4, -4)
        pygame.draw.rect(surface, theme.text, exit_rect, border_radius=5)
        draw_vector_icon(surface, "next", exit_rect.center, theme.background, 14, 2)

        if self.selected_result and self.selected_customer_index is not None:
            selected_route = self.selected_result.customers[self.selected_customer_index].positions
            route_points = [
                (
                    self.grid_origin[0] + x * self.cell_size + self.cell_size // 2,
                    self.grid_origin[1] + y * self.cell_size + self.cell_size // 2,
                )
                for x, y in selected_route
            ]
            if len(route_points) > 1:
                pygame.draw.lines(
                    surface, theme.sky, False, route_points, max(2, self.cell_size // 9)
                )

        if self.selected_result:
            for index, (customer, progress) in enumerate(
                zip(self.selected_result.customers, self.customer_progress, strict=True)
            ):
                center = self._customer_center(index)
                trail = customer.positions[max(0, int(progress) - 3) : int(progress) + 1]
                if len(trail) > 1:
                    trail_points = [
                        (
                            self.grid_origin[0] + x * self.cell_size + self.cell_size // 2,
                            self.grid_origin[1] + y * self.cell_size + self.cell_size // 2,
                        )
                        for x, y in trail
                    ]
                    pygame.draw.lines(surface, (124, 196, 255), False, trail_points, 2)
                halo = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                pygame.draw.circle(
                    halo,
                    (124, 196, 255, 65),
                    (self.cell_size // 2, self.cell_size // 2),
                    max(5, self.cell_size // 2 - 2),
                )
                surface.blit(halo, halo.get_rect(center=center))
                pygame.draw.circle(surface, theme.text, center, max(4, self.cell_size // 7))
                if index == self.selected_customer_index:
                    pygame.draw.circle(
                        surface,
                        theme.sky,
                        center,
                        max(8, self.cell_size // 3),
                        width=2,
                    )

        for category, color, tag in section_tags:
            draw_panel(surface, tag, color, radius=4, gradient=False)
            label_font = pygame.font.Font(None, max(13, min(18, self.cell_size // 2)))
            label = label_font.render(category, True, theme.background)
            surface.blit(label, label.get_rect(center=tag.center))

        legend_y = self.grid_rect.bottom - 22
        legend_items = [*section_colors.items(), ("Clientes", theme.text)]
        legend_font = fonts["small"]
        widths = [legend_font.size(label)[0] + 20 for label, _ in legend_items]
        gap = 10
        cursor_x = self.grid_rect.centerx - (sum(widths) + gap * (len(widths) - 1)) // 2
        for (label, color), item_width in zip(legend_items, widths, strict=True):
            pygame.draw.circle(surface, color, (cursor_x + 5, legend_y + 7), 4)
            self._draw_label(
                surface,
                label,
                (cursor_x + 13, legend_y),
                legend_font,
                theme.muted,
            )
            cursor_x += item_width + gap
        surface.set_clip(panel_clip)

    def _draw_metrics(
        self,
        surface: pygame.Surface,
        fonts: dict[str, pygame.font.Font],
        theme: Theme,
    ) -> None:
        x = self.metrics_rect.x + 14
        y = self.metrics_rect.y + 14
        self._draw_label(surface, "EN VIVO", (x, y), fonts["small"], theme.sky)
        if not self.comparison:
            empty = pygame.Rect(x, y + 34, self.metrics_rect.width - 28, 104)
            draw_panel(surface, empty, theme.panel, radius=8, border_color=theme.border)
            draw_vector_icon(surface, "play", (empty.x + 24, empty.y + 26), theme.sky, 18)
            message = "Datos pendientes" if self.data_warning else "Listo para simular"
            self._draw_label(
                surface,
                message,
                (empty.x + 44, empty.y + 18),
                fonts["body"],
                theme.text,
            )
            self._draw_label(
                surface,
                (
                    "Inicia para comparar BFS y A*"
                    if not self.data_warning
                    else self.data_warning[:32]
                ),
                (empty.x + 12, empty.y + 58),
                fonts["small"],
                theme.muted,
            )
            if self.data_warning:
                pygame.draw.line(
                    surface,
                    theme.error,
                    (empty.x + 12, empty.bottom - 12),
                    (empty.right - 12, empty.bottom - 12),
                    2,
                )
        else:
            gap = 8
            card_width = (self.metrics_rect.width - 28 - gap) // 2
            results = [self.comparison[algorithm] for algorithm in self.algorithms]
            max_nodes = max(result.explored_nodes for result in results) or 1
            max_time = max(result.mean_search_ms for result in results) or 1
            for index, (algorithm, result) in enumerate(zip(self.algorithms, results, strict=True)):
                card = pygame.Rect(x + index * (card_width + gap), y + 34, card_width, 112)
                draw_panel(
                    surface,
                    card,
                    theme.deep if algorithm == self.algorithm else theme.panel,
                    radius=8,
                    border_color=theme.primary if algorithm == self.algorithm else theme.border,
                )
                self._draw_label(
                    surface, algorithm, (card.x + 10, card.y + 8), fonts["body"], theme.text
                )
                self._draw_label(
                    surface,
                    f"{result.explored_nodes} nodos",
                    (card.x + 10, card.y + 34),
                    fonts["label"],
                    theme.muted,
                )
                self._draw_label(
                    surface,
                    f"{result.mean_path_length:.1f} celdas",
                    (card.x + 10, card.y + 52),
                    fonts["label"],
                    theme.text,
                )
                self._draw_label(
                    surface,
                    f"{result.mean_search_ms:.2f} ms",
                    (card.x + 10, card.y + 70),
                    fonts["label"],
                    theme.muted,
                )
                node_bar = pygame.Rect(card.x + 10, card.bottom - 14, card.width - 20, 4)
                pygame.draw.rect(surface, theme.border, node_bar, border_radius=2)
                node_bar.width = round(node_bar.width * result.explored_nodes / max_nodes)
                pygame.draw.rect(surface, theme.primary, node_bar, border_radius=2)
                time_bar = pygame.Rect(card.x + 10, card.bottom - 8, card.width - 20, 3)
                pygame.draw.rect(surface, theme.border, time_bar, border_radius=2)
                time_bar.width = round(time_bar.width * result.mean_search_ms / max_time)
                pygame.draw.rect(surface, theme.sky, time_bar, border_radius=2)
            self._draw_label(
                surface,
                "ZONAS MÁS TRANSITADAS",
                (x, y + 157),
                fonts["small"],
                theme.muted,
            )
            top_congested = self.selected_result.top_congested[:5] if self.selected_result else ()
            max_traffic = max((traffic for _, traffic in top_congested), default=1)
            for index, (position, traffic) in enumerate(top_congested):
                row_y = y + 180 + index * 28
                self._draw_label(
                    surface,
                    f"{index + 1}",
                    (x, row_y),
                    fonts["label"],
                    theme.sky,
                )
                bar = pygame.Rect(x + 18, row_y + 4, max(20, card_width - 62), 8)
                pygame.draw.rect(surface, theme.border, bar, border_radius=4)
                filled = bar.copy()
                filled.width = round(bar.width * traffic / max_traffic)
                pygame.draw.rect(surface, theme.primary, filled, border_radius=4)
                self._draw_label(
                    surface,
                    f"{position[0] + 1},{position[1] + 1}  {traffic}",
                    (bar.right + 5, row_y),
                    fonts["label"],
                    theme.text,
                )
        state = "PAUSADO" if self.paused else ("EN CURSO" if self.running else "COMPLETADO")
        state_color = theme.gold if self.paused else theme.accent
        self._draw_label(
            surface,
            f"{state}  ·  {self.elapsed_seconds:.1f} s",
            (x, self.metrics_rect.bottom - 28),
            fonts["small"],
            state_color,
        )
