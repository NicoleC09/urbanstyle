"""Vista Pygame de precios dinámicos y comparación de políticas."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from src.urbanstyle.task_c_pricing.data import (
    load_forecast_units,
    load_product_data,
    make_environment,
)
from src.urbanstyle.task_c_pricing.demand_model import expected_demand
from src.urbanstyle.task_c_pricing.env import DynamicPricingEnv, ProductPricingData
from src.urbanstyle.task_c_pricing.policies import (
    theoretical_optimal_price,
    theoretical_price_policy,
)
from src.urbanstyle.utils.config import resolve_path
from stable_baselines3 import DQN, PPO

from app.ui.layout import Track, WeightedLayout
from app.ui.theme import Theme, draw_panel
from app.ui.widgets import Button, SegmentedControl, Slider

LOGGER = logging.getLogger("urbanstyle.app.task_c")
MODEL_TYPES = {"PPO": PPO, "DQN": DQN}


class TaskCTab:
    """Controles y visualizaciones para comparar un agente con precio medio."""

    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.forecast_units = load_forecast_units(config)
        self.algorithm = "PPO"
        self.categories = config["data"]["categories"]
        self.category_index = 0
        pricing = config["pricing"]
        self.price_slider = Slider(
            pygame.Rect(0, 0, 150, 4),
            float(pricing["price_min_multiplier"]),
            float(pricing["price_max_multiplier"]),
            1.0,
            formatter=lambda value: f"{value:.2f}×",
        )
        self.speed_slider = Slider(
            pygame.Rect(0, 0, 150, 4),
            0.25,
            2.0,
            1.0,
            formatter=lambda value: f"{value:.1f}×",
        )
        self.models = {}
        self.model_status: dict[str, str] = {}
        self.products: list[ProductPricingData] = []
        self.evaluation_rows: list[dict] = []
        evaluation_path = resolve_path(config["paths"]["outputs"]) / "pricing_evaluation.csv"
        if evaluation_path.is_file():
            try:
                self.evaluation_rows = pd.read_csv(evaluation_path).to_dict(orient="records")
            except (OSError, pd.errors.ParserError, ValueError) as error:
                LOGGER.warning("No se pudo cargar la evaluación: %s", error)
        self.load_error: str | None = None
        try:
            self.products = load_product_data(config)
            model_dir = resolve_path(config["paths"]["models"])
            for algorithm, model_type in MODEL_TYPES.items():
                model_path = model_dir / algorithm.lower()
                zip_path = model_path.with_suffix(".zip")
                if zip_path.is_file():
                    try:
                        self.models[algorithm] = model_type.load(str(model_path), device="cpu")
                        self.model_status[algorithm] = "cargado"
                    except (OSError, ValueError, RuntimeError) as error:
                        LOGGER.warning("No se pudo cargar %s: %s", zip_path, error)
                        self.model_status[algorithm] = "error"
                else:
                    self.model_status[algorithm] = "faltante"
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.load_error = str(error)

        self.category_control = SegmentedControl(pygame.Rect(0, 0, 212, 38), self.categories)
        self.algorithm_control = SegmentedControl(pygame.Rect(0, 0, 212, 38), ("PPO", "DQN"))
        self.chart_metric_control = SegmentedControl(
            pygame.Rect(0, 0, 156, 28), ("Margen", "Ingreso")
        )
        self.buttons = {
            "run": Button(pygame.Rect(0, 0, 130, 38), "Ejecutar episodio", self.start_episode),
            "pause": Button(pygame.Rect(0, 0, 100, 38), "Pausar", self.toggle_pause),
            "reset": Button(pygame.Rect(0, 0, 100, 38), "Reiniciar", self.reset),
        }
        self.control_rect = pygame.Rect(0, 0, 0, 0)
        self.chart_rect = pygame.Rect(0, 0, 0, 0)
        self.metrics_rect = pygame.Rect(0, 0, 0, 0)
        self.agent_env: DynamicPricingEnv | None = None
        self.baseline_env: DynamicPricingEnv | None = None
        self.running = False
        self.paused = False
        self.elapsed = 0.0
        self.step_accumulator = 0.0
        self.step_seconds = float(config["task_c"]["animation_step_seconds"])
        self.agent_cumulative: list[float] = []
        self.baseline_cumulative: list[float] = []
        self.agent_revenue_cumulative: list[float] = []
        self.baseline_revenue_cumulative: list[float] = []
        self.price_history: list[float] = []
        self.agent_steps: list[dict] = []
        self.current_info: dict = {}
        self.last_episode_scores: dict[str, float] = {}
        self.initial_price = 1.0
        self._set_button_states()

    @property
    def category(self) -> str:
        """Categoría elegida actualmente."""
        return self.categories[self.category_index]

    @property
    def product(self) -> ProductPricingData | None:
        """Datos calibrados de la categoría seleccionada."""
        return next((item for item in self.products if item.name == self.category), None)

    def _set_button_states(self) -> None:
        status = self.model_status.get(self.algorithm, "faltante")
        self.buttons["run"].enabled = self.product is not None and status == "cargado"
        self.buttons["pause"].enabled = self.running
        self.buttons["pause"].label = "Reanudar" if self.paused else "Pausar"
        self.buttons["reset"].enabled = self.running or bool(self.current_info)

    def layout_controls(self, content_rect: pygame.Rect) -> None:
        """Recalcula paneles y widgets para el tamaño disponible."""
        top = content_rect.y + 120
        panel_rect = pygame.Rect(
            content_rect.x + 22,
            top,
            content_rect.width - 44,
            max(0, content_rect.height - 130),
        )
        self.control_rect, self.chart_rect, self.metrics_rect = WeightedLayout(gap=16).columns(
            panel_rect,
            [Track(1, 280), Track(2, 420), Track(1, 180)],
        )
        x = self.control_rect.x + 16
        self.category_control.rect.topleft = (x, top + 48)
        self.algorithm_control.rect.topleft = (x, top + 100)
        self.price_slider.rect = pygame.Rect(x, top + 214, 170, 4)
        self.speed_slider.rect = pygame.Rect(x, top + 266, 170, 4)
        self.buttons["run"].rect = pygame.Rect(x, top + 296, self.control_rect.width - 32, 32)
        self.buttons["pause"].rect = pygame.Rect(x, top + 334, 92, 32)
        self.buttons["reset"].rect = pygame.Rect(x + 100, top + 334, 92, 32)
        self.chart_metric_control.rect = pygame.Rect(
            self.chart_rect.right - 172, self.chart_rect.y + 10, 156, 28
        )

    def handle_event(self, event: pygame.event.Event) -> None:
        """Maneja controles, selector y atajos de pausa/reinicio."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.toggle_pause()
            elif event.key == pygame.K_r:
                self.reset()
        if self.category_control.handle_event(event):
            self.category_index = self.category_control.selected
            self._set_button_states()
        if self.algorithm_control.handle_event(event):
            self.algorithm = self.algorithm_control.value
            self._set_button_states()
        self.chart_metric_control.handle_event(event)
        self.price_slider.handle_event(event)
        self.speed_slider.handle_event(event)
        for button in self.buttons.values():
            button.handle_event(event)

    def _new_environment(self, product: ProductPricingData) -> DynamicPricingEnv:
        """Crea un entorno con las calibraciones del proyecto."""
        environment = make_environment(self.config, [product], forecast_units=self.forecast_units)
        return environment

    def start_episode(self) -> None:
        """Inicia un episodio pareado si hay modelo entrenado disponible."""
        product = self.product
        model = self.models.get(self.algorithm)
        if product is None or model is None:
            self._set_button_states()
            return
        self.reset()
        self.initial_price = self.price_slider.value
        self.agent_env = self._new_environment(product)
        self.baseline_env = self._new_environment(product)
        options = {
            "product": product.name,
            "segment": 1,
            "initial_price": self.initial_price,
        }
        agent_observation, _ = self.agent_env.reset(
            seed=int(self.config["project"]["seed"]), options=options
        )
        baseline_observation, _ = self.baseline_env.reset(
            seed=int(self.config["project"]["seed"]), options=options
        )
        self.agent_observation = agent_observation
        self.baseline_observation = baseline_observation
        self.agent_cumulative = [0.0]
        self.baseline_cumulative = [0.0]
        self.agent_revenue_cumulative = [0.0]
        self.baseline_revenue_cumulative = [0.0]
        self.price_history = [self.agent_env.price * product.reference_price]
        self.agent_steps = []
        self.running = True
        self.paused = False
        self.elapsed = 0.0
        self.step_accumulator = 0.0
        self.last_episode_scores = {}
        self._set_button_states()

    def toggle_pause(self) -> None:
        """Pausa o continúa el episodio activo."""
        if self.running:
            self.paused = not self.paused
            self._set_button_states()

    def reset(self) -> None:
        """Cierra entornos y reinicia el estado visual del episodio."""
        if self.agent_env:
            self.agent_env.close()
        if self.baseline_env:
            self.baseline_env.close()
        self.agent_env = None
        self.baseline_env = None
        self.running = False
        self.paused = False
        self.elapsed = 0.0
        self.step_accumulator = 0.0
        self.agent_cumulative = []
        self.baseline_cumulative = []
        self.agent_revenue_cumulative = []
        self.baseline_revenue_cumulative = []
        self.price_history = []
        self.agent_steps = []
        self.current_info = {}
        self.last_episode_scores = {}
        self._set_button_states()

    def _step_episode(self) -> None:
        """Avanza los dos entornos con semillas pareadas por paso."""
        if not self.agent_env or not self.baseline_env:
            return
        shared_random_state = deepcopy(self.agent_env.np_random.bit_generator.state)
        model = self.models[self.algorithm]
        action, _ = model.predict(self.agent_observation, deterministic=True)
        self.agent_observation, _, agent_done, agent_truncated, agent_info = self.agent_env.step(
            int(action)
        )
        self.baseline_env.np_random.bit_generator.state = deepcopy(shared_random_state)
        self.baseline_observation, _, base_done, base_truncated, base_info = self.baseline_env.step(
            theoretical_price_policy(self.baseline_observation, self.baseline_env)
        )
        self.agent_env.np_random.bit_generator.state = deepcopy(
            self.baseline_env.np_random.bit_generator.state
        )
        self.agent_steps.append(agent_info)
        self.agent_cumulative.append(self.agent_cumulative[-1] + agent_info["profit"])
        self.baseline_cumulative.append(self.baseline_cumulative[-1] + base_info["profit"])
        self.agent_revenue_cumulative.append(
            self.agent_revenue_cumulative[-1] + agent_info["revenue"]
        )
        self.baseline_revenue_cumulative.append(
            self.baseline_revenue_cumulative[-1] + base_info["revenue"]
        )
        self.price_history.append(float(agent_info["price"]))
        self.current_info = agent_info
        if agent_done or agent_truncated or base_done or base_truncated:
            self.running = False
            self.last_episode_scores = {
                f"{self.algorithm} margen": self.agent_cumulative[-1],
                "Óptimo teórico": self.baseline_cumulative[-1],
            }
            self._set_button_states()

    def update(self, delta_seconds: float) -> None:
        """Avanza un paso del episodio según el reloj visual."""
        if not self.running or self.paused:
            return
        self.elapsed += delta_seconds
        self.step_accumulator += delta_seconds
        step_interval = self.step_seconds / max(0.25, float(self.speed_slider.value))
        while self.step_accumulator >= step_interval and self.running:
            self.step_accumulator -= step_interval
            self._step_episode()

    @staticmethod
    def _draw_label(surface, text, position, font, color) -> None:
        surface.blit(font.render(text, True, color), position)

    def draw(self, surface: pygame.Surface, content_rect: pygame.Rect, fonts, theme: Theme) -> None:
        """Dibuja controles, ingreso acumulado y detalles del paso actual."""
        self.layout_controls(content_rect)
        self._draw_label(
            surface,
            "PRECIOS DINÁMICOS",
            (content_rect.x + 32, content_rect.y + 30),
            fonts["small"],
            theme.accent,
        )
        self._draw_label(
            surface,
            "Precio · demanda · margen",
            (content_rect.x + 32, content_rect.y + 55),
            fonts["title"],
            theme.foreground,
        )
        draw_panel(surface, self.control_rect, theme.surface_alt, radius=10)
        draw_panel(surface, self.chart_rect, theme.surface_alt, radius=10)
        draw_panel(surface, self.metrics_rect, theme.surface_alt, radius=10)
        x = self.control_rect.x + 16
        self._draw_label(
            surface, "PRODUCTO", (x, self.control_rect.y + 18), fonts["small"], theme.accent
        )
        self.category_control.draw(surface, fonts["small"], theme)
        self._draw_label(
            surface, "AGENTE", (x, self.control_rect.y + 104), fonts["small"], theme.accent
        )
        self.algorithm_control.draw(surface, fonts["small"], theme)
        model_status = self.model_status.get(self.algorithm, "faltante")
        if self.load_error:
            status_text = self.load_error[:40]
            status_color = theme.coral
        elif model_status == "faltante":
            status_text = (
                f"Falta modelo. Ejecuta: python scripts/train_rl.py --algorithm {self.algorithm}"
            )
            status_color = theme.coral
        elif model_status == "error":
            status_text = f"No se pudo cargar models/{self.algorithm.lower()}.zip"
            status_color = theme.coral
        else:
            status_text = f"{self.algorithm} listo · objetivo margen"
            status_color = theme.success
        self._draw_label(
            surface, status_text[:43], (x, self.control_rect.y + 170), fonts["small"], status_color
        )
        self._draw_label(
            surface,
            "PRECIO INICIAL · MULTIPLICADOR",
            (x, self.control_rect.y + 188),
            fonts["small"],
            theme.muted,
        )
        self.price_slider.draw(surface, fonts["small"], theme, label="")
        self._draw_label(
            surface,
            f"VELOCIDAD  {self.speed_slider.value:.1f}×",
            (x, self.control_rect.y + 238),
            fonts["small"],
            theme.muted,
        )
        self.speed_slider.draw(surface, fonts["small"], theme, label="")
        for button in self.buttons.values():
            button.draw(surface, fonts["small"], theme)
        self.chart_metric_control.draw(surface, fonts["label"], theme)
        self._draw_chart(surface, fonts, theme)
        self._draw_metrics(surface, fonts, theme)

    def _draw_chart(self, surface: pygame.Surface, fonts, theme: Theme) -> None:
        rect = self.chart_rect
        x, y = rect.x + 48, rect.y + 54
        width, height = rect.width - 72, rect.height - 142
        if self.chart_metric_control.value == "Ingreso":
            agent_values = self.agent_revenue_cumulative
            baseline_values = self.baseline_revenue_cumulative
            chart_heading = "INGRESO ACUMULADO · AGENTE VS ÓPTIMO"
        else:
            agent_values = self.agent_cumulative
            baseline_values = self.baseline_cumulative
            chart_heading = "MARGEN ACUMULADO · AGENTE VS ÓPTIMO"
        self._draw_label(
            surface, chart_heading, (rect.x + 16, rect.y + 14), fonts["small"], theme.foreground
        )
        if not self.agent_cumulative:
            self._draw_demand_margin_curve(surface, fonts, theme, x, y, width, height)
            return

        timeline_height = max(42, height // 4)
        margin_height = height - timeline_height - 18
        maximum = max([1.0, *agent_values, *baseline_values])
        pygame.draw.line(
            surface, theme.line, (x, y + margin_height), (x + width, y + margin_height), 1
        )
        pygame.draw.line(surface, theme.line, (x, y), (x, y + margin_height), 1)
        for values, color in (
            (agent_values, theme.primary),
            (baseline_values, theme.success),
        ):
            if len(values) < 2:
                continue
            points = [
                (
                    round(x + index * width / max(1, len(values) - 1)),
                    round(y + margin_height * (1 - value / maximum)),
                )
                for index, value in enumerate(values)
            ]
            pygame.draw.lines(surface, color, False, points, 3)

        timeline_y = y + margin_height + 18
        prices = self.price_history
        if prices:
            low = min(prices)
            high = max(prices)
            span = max(1.0, high - low)
            price_points = [
                (
                    round(x + index * width / max(1, len(prices) - 1)),
                    round(timeline_y + timeline_height * (1 - (price - low) / span)),
                )
                for index, price in enumerate(prices)
            ]
            if len(price_points) > 1:
                pygame.draw.lines(surface, theme.category_beauty, False, price_points, 2)
            action_colors = (theme.error, theme.muted, theme.success)
            for point, info in zip(price_points[1:], self.agent_steps, strict=False):
                pygame.draw.circle(
                    surface,
                    action_colors[int(info["action"])],
                    point,
                    4,
                )
        self._draw_label(
            surface,
            "PRECIO EN EL TIEMPO · puntos = acciones",
            (x, timeline_y - 14),
            fonts["label"],
            theme.muted,
        )

    def _draw_demand_margin_curve(self, surface, fonts, theme, x, y, width, height) -> None:
        product = self.product
        if product is None:
            self._draw_label(
                surface,
                "Calibración de producto no disponible",
                (x, y + 20),
                fonts["body"],
                theme.muted,
            )
            return
        minimum = float(self.config["pricing"]["price_min_multiplier"])
        maximum = float(self.config["pricing"]["price_max_multiplier"])
        forecast = float(self.forecast_units.get(product.name, product.baseline_units))
        prices = np.linspace(minimum, maximum, 80)
        demand = np.asarray(
            [expected_demand(price, 1.0, forecast, product.elasticity) for price in prices]
        )
        cost_fraction = float(self.config["pricing"]["unit_cost_fraction"])
        margin = (prices - cost_fraction) * product.reference_price * demand
        max_demand = max(1e-9, float(demand.max()))
        max_margin = max(1e-9, float(margin.max()))
        demand_points = [
            (
                round(x + index * width / (len(prices) - 1)),
                round(y + height * (1 - value / max_demand)),
            )
            for index, value in enumerate(demand)
        ]
        margin_points = [
            (
                round(x + index * width / (len(prices) - 1)),
                round(y + height * (1 - value / max_margin)),
            )
            for index, value in enumerate(margin)
        ]
        pygame.draw.line(surface, theme.border, (x, y + height), (x + width, y + height), 1)
        pygame.draw.lines(surface, theme.sky, False, demand_points, 2)
        pygame.draw.lines(surface, theme.primary, False, margin_points, 3)
        reference_environment = self._new_environment(product)
        reference_environment.reset(options={"product": product.name, "segment": 1})
        optimum = theoretical_optimal_price(reference_environment)
        optimum_x = x + round((optimum - minimum) / max(1e-9, maximum - minimum) * width)
        pygame.draw.line(surface, theme.success, (optimum_x, y), (optimum_x, y + height), 2)
        self._draw_label(
            surface,
            f"Óptimo {optimum:.2f}×",
            (min(x + width - 105, optimum_x + 5), y + 4),
            fonts["label"],
            theme.success,
        )
        current = self.agent_env.price if self.agent_env else self.price_slider.value
        current_x = x + round((current - minimum) / max(1e-9, maximum - minimum) * width)
        pygame.draw.line(surface, theme.text, (current_x, y), (current_x, y + height), 1)
        self._draw_label(surface, "Demanda", (x + 4, y + 4), fonts["label"], theme.sky)
        self._draw_label(surface, "Margen", (x + 4, y + 20), fonts["label"], theme.primary)

    def _draw_metrics(self, surface: pygame.Surface, fonts, theme: Theme) -> None:
        x = self.metrics_rect.x + 14
        y = self.metrics_rect.y + 18
        self._draw_label(surface, "PASO ACTUAL", (x, y), fonts["label"], theme.sky)
        action_label = (
            ("Bajar", "Mantener", "Subir")[self.current_info["action"]]
            if self.current_info
            else "—"
        )
        reference_price = self.product.reference_price if self.product else 0.0
        displayed_price = (
            float(self.current_info["price"])
            if self.current_info
            else reference_price * float(self.price_slider.value)
        )
        kpis = (
            ("PRECIO", f"{displayed_price:.2f}"),
            ("ACCIÓN", action_label),
            ("UNIDADES", str(self.current_info.get("units_sold", "—"))),
            ("MARGEN", f"{self.current_info['profit']:,.1f}" if self.current_info else "—"),
        )
        gap = 6
        card_width = (self.metrics_rect.width - 28 - gap) // 2
        for index, (label, value) in enumerate(kpis):
            card = pygame.Rect(
                x + (index % 2) * (card_width + gap),
                y + 22 + (index // 2) * 42,
                card_width,
                38,
            )
            draw_panel(surface, card, theme.panel, radius=6, border_color=theme.border)
            self._draw_label(surface, label, (card.x + 6, card.y + 4), fonts["label"], theme.muted)
            self._draw_label(surface, value, (card.x + 6, card.y + 19), fonts["label"], theme.text)

        self._draw_label(
            surface, "EVALUACIÓN · MARGEN ± DESV.", (x, y + 108), fonts["label"], theme.sky
        )
        policies = (
            "PPO agente",
            "DQN agente",
            "Precio bajo",
            "Precio medio",
            "Precio alto",
            "Óptimo teórico",
        )
        evaluation_rows = {str(row.get("policy")): row for row in self.evaluation_rows}
        values = []
        for label in policies:
            row = evaluation_rows.get(label)
            if row is None:
                continue
            if "mean_margin" in row:
                mean = float(row["mean_margin"])
                deviation = float(row.get("std_margin", 0.0))
            else:
                mean = float(row.get("mean_revenue", 0.0))
                deviation = float(row.get("std_revenue", 0.0))
            values.append((label, mean, deviation))
        best = max((value for _, value, _ in values), default=None)
        row_height = 18
        table_y = y + 128
        label_width = 62
        bar_width = max(24, self.metrics_rect.width - 34 - label_width - 76)
        max_mean = max((value for _, value, _ in values), default=1.0) or 1.0
        for index, (label, mean, deviation) in enumerate(values):
            row_y = table_y + index * row_height
            short_label = {
                "PPO agente": "PPO",
                "DQN agente": "DQN",
                "Precio bajo": "Bajo",
                "Precio medio": "Medio",
                "Precio alto": "Alto",
                "Óptimo teórico": "Óptimo",
            }.get(label, label[:9])
            color = (
                theme.success if mean == best else theme.primary if "agente" in label else theme.sky
            )
            self._draw_label(surface, short_label, (x, row_y), fonts["label"], theme.text)
            bar = pygame.Rect(x + label_width, row_y + 6, bar_width, 6)
            pygame.draw.rect(surface, theme.border, bar, border_radius=4)
            fill = bar.copy()
            fill.width = round(bar.width * max(0.0, mean) / max_mean)
            pygame.draw.rect(surface, color, fill, border_radius=4)
            error_half = round(bar.width * deviation / max_mean)
            center_x = min(bar.right - 1, bar.x + fill.width)
            pygame.draw.line(
                surface,
                theme.text,
                (max(bar.x, center_x - error_half), bar.centery),
                (min(bar.right, center_x + error_half), bar.centery),
                2,
            )
            caption_x = bar.right + 5
            self._draw_label(
                surface,
                f"{mean:,.0f}±{deviation:,.0f}",
                (caption_x, row_y),
                fonts["label"],
                theme.muted,
            )
        self._draw_label(
            surface,
            "MARGEN ACUMULADO",
            (x, y + 246),
            fonts["label"],
            theme.muted,
        )
        if self.agent_cumulative:
            self._draw_label(
                surface,
                f"Agente {self.agent_cumulative[-1]:,.0f}",
                (x, y + 264),
                fonts["label"],
                theme.primary,
            )
            self._draw_label(
                surface,
                f"Óptimo {self.baseline_cumulative[-1]:,.0f}",
                (x, y + 281),
                fonts["label"],
                theme.success,
            )
        fallback = self.product.elasticity_method if self.product else "sin datos"
        self._draw_label(
            surface,
            f"ε {self.product.elasticity:.1f} · {fallback}" if self.product else "Sin calibración",
            (x, self.metrics_rect.bottom - 18),
            fonts["small"],
            theme.gold if fallback == "fallback" else theme.muted,
        )
