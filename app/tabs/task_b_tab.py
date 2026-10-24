"""Vista de demanda semanal, métricas comparativas y exportación."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd
import pygame
from src.urbanstyle.utils.config import resolve_path

from app.ui.layout import Track, WeightedLayout
from app.ui.theme import Theme, draw_panel
from app.ui.widgets import Button, Chip, SegmentedControl, Table, Toast, Toggle, Tooltip

LOGGER = logging.getLogger("urbanstyle.app.task_b")


class TaskBTab:
    """Visualiza seis series semanales, test, baselines y pronóstico futuro."""

    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.output_dir = resolve_path(config["paths"]["outputs"])
        self.weekly = pd.DataFrame()
        self.test_predictions = pd.DataFrame()
        self.metrics = pd.DataFrame()
        self.future_predictions = pd.DataFrame()
        self.load_error: str | None = None
        try:
            self._load_outputs()
        except (OSError, ValueError, pd.errors.ParserError) as error:
            self.load_error = str(error)
            LOGGER.warning("No se pudieron cargar los resultados Task B: %s", error)
        self._series_cache: dict[
            tuple[str, int], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ] = {}
        self._metric_lookup: dict[tuple[int, str, str, str], tuple[float, float]] = {}
        self._average_lookup: dict[tuple[int, str], tuple[float, float]] = {}
        self._comparison_lookup: dict[int, tuple[int, int]] = {}
        self._build_metric_cache()

        if not self.weekly.empty:
            self.series_labels = sorted(self.weekly.apply(self._series_label, axis=1).unique())
        else:
            self.series_labels = [
                f"{category} · {gender}"
                for category in config["data"]["categories"]
                for gender in config["data"]["genders"]
            ]
        self.series_chips = [
            Chip(pygame.Rect(0, 0, 80, 28), label, selected=index == 0)
            for index, label in enumerate(self.series_labels)
        ]
        self.context_windows = sorted({int(value) for value in config["task_b"]["context_windows"]})
        lookback = int(config["task_b"]["lookback"])
        selected = self.context_windows.index(lookback) if lookback in self.context_windows else 0
        self.context_control = SegmentedControl(
            pygame.Rect(0, 0, 180, 28),
            tuple(f"{window} semanas" for window in self.context_windows),
            selected=selected,
        )
        self.show_baseline = True
        self.zoom_test = False
        self.baseline_toggle = Toggle(pygame.Rect(0, 0, 42, 24), True)
        self.zoom_toggle = Toggle(pygame.Rect(0, 0, 42, 24), False)
        self.message = ""
        self.toast: Toast | None = None
        self.embedding_info_rect = pygame.Rect(0, 0, 16, 16)
        self.embedding_tooltip = Tooltip(
            self.embedding_info_rect,
            "LSTM global con embedding categoría × género.",
        )
        self.buttons = {
            "export": Button(
                pygame.Rect(0, 0, 160, 36),
                "Exportar CSV",
                self.export_csv,
                kind="primary",
                icon="download",
            )
        }
        self.control_rect = pygame.Rect(0, 0, 0, 0)
        self.chart_rect = pygame.Rect(0, 0, 0, 0)
        self.metrics_rect = pygame.Rect(0, 0, 0, 0)
        self.chart_plot_rect = pygame.Rect(0, 0, 0, 0)
        self._hover_value: tuple[str, float, tuple[int, int]] | None = None
        self._set_control_states()

    @staticmethod
    def _series_label(row: pd.Series) -> str:
        gender = "F" if row["Gender"] == "Female" else "M"
        return f"{row['Product Category']} · {gender}"

    def _load_outputs(self) -> None:
        processed_dir = resolve_path(self.config["paths"]["processed_data"])
        weekly_path = processed_dir / "weekly_sales.csv"
        metrics_path = self.output_dir / "forecast_metrics.csv"
        predictions_path = self.output_dir / "forecast_test_predictions.csv"
        future_path = self.output_dir / "predictions.csv"
        if not weekly_path.is_file():
            raise FileNotFoundError(
                f"Faltan las series semanales: {weekly_path}. "
                "Ejecuta python scripts/prepare_data.py."
            )
        self.weekly = pd.read_csv(weekly_path, parse_dates=["week"])
        if metrics_path.is_file() and predictions_path.is_file() and future_path.is_file():
            self.metrics = pd.read_csv(metrics_path)
            self.test_predictions = pd.read_csv(predictions_path, parse_dates=["week"])
            self.future_predictions = pd.read_csv(future_path)
        else:
            raise FileNotFoundError(
                "Faltan resultados del pronóstico. Ejecuta python scripts/train_lstm.py."
            )

    def _build_metric_cache(self) -> None:
        if self.metrics.empty:
            return
        for row in self.metrics.itertuples(index=False):
            context = int(row.context_weeks)
            self._metric_lookup[(context, row.category, row.gender, row.model)] = (
                float(row.mae),
                float(row.rmse),
            )
        for (context, model), rows in self.metrics.groupby(["context_weeks", "model"]):
            self._average_lookup[(int(context), model)] = (
                float(rows["mae"].mean()),
                float(rows["rmse"].mean()),
            )
        for context, rows in self.metrics.groupby("context_weeks"):
            paired = rows.pivot_table(index=["category", "gender"], columns="model", values="mae")
            if "LSTM" in paired and "Media móvil" in paired:
                comparison = paired[["LSTM", "Media móvil"]].dropna()
                self._comparison_lookup[int(context)] = (
                    int((comparison["LSTM"] < comparison["Media móvil"]).sum()),
                    int(len(comparison)),
                )

    @property
    def selected_series(self) -> str:
        return self.series_labels[self.selected_index]

    @property
    def selected_index(self) -> int:
        return next((index for index, chip in enumerate(self.series_chips) if chip.selected), 0)

    @property
    def selected_window(self) -> int:
        return self.context_windows[self.context_control.selected]

    def _set_control_states(self) -> None:
        self.buttons["export"].enabled = not self.future_predictions.empty

    def layout_controls(self, content_rect: pygame.Rect) -> None:
        """Calcula paneles y widgets para el espacio disponible."""
        panel_rect = pygame.Rect(
            content_rect.x + 20,
            content_rect.y + 90,
            content_rect.width - 40,
            max(0, content_rect.height - 108),
        )
        self.control_rect, self.chart_rect, self.metrics_rect = WeightedLayout(gap=14).columns(
            panel_rect,
            [Track(0.82, 226), Track(2.0, 500), Track(0.98, 270)],
        )
        left_x = self.control_rect.x + 12
        left_width = self.control_rect.width - 24
        gap = 5
        chip_width = (left_width - gap) // 2
        for index, chip in enumerate(self.series_chips):
            row, column = divmod(index, 2)
            chip.rect = pygame.Rect(
                left_x + column * (chip_width + gap),
                self.control_rect.y + 38 + row * 32,
                chip_width,
                27,
            )
        self.context_control.rect = pygame.Rect(left_x, self.control_rect.y + 159, left_width, 26)
        self.baseline_toggle.rect.topleft = (
            self.control_rect.right - 52,
            self.control_rect.y + 205,
        )
        self.zoom_toggle.rect.topleft = (
            self.control_rect.right - 52,
            self.control_rect.y + 241,
        )
        self.buttons["export"].rect = pygame.Rect(
            left_x, self.control_rect.bottom - 48, left_width, 36
        )
        self.embedding_info_rect.topleft = (left_x, self.control_rect.y + 269)
        self.chart_plot_rect = pygame.Rect(
            self.chart_rect.x + 44,
            self.chart_rect.y + 76,
            max(1, self.chart_rect.width - 60),
            max(1, self.chart_rect.height - 126),
        )

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for selected_index, chip in enumerate(self.series_chips):
                if chip.rect.collidepoint(event.pos):
                    for index, item in enumerate(self.series_chips):
                        item.selected = index == selected_index
                    self.message = ""
                    break
            if self.baseline_toggle.handle_event(event):
                self.show_baseline = self.baseline_toggle.value
            if self.zoom_toggle.handle_event(event):
                self.zoom_test = self.zoom_toggle.value
        self.context_control.handle_event(event)
        self._set_control_states()
        for button in self.buttons.values():
            button.handle_event(event)
        if event.type == pygame.MOUSEMOTION:
            self._hover_value = None

    def export_csv(self) -> None:
        source = self.output_dir / "predictions.csv"
        if not source.is_file():
            self.message = "No hay pronósticos para exportar."
            return
        destination = self.output_dir / "predictions_export.csv"
        shutil.copy2(source, destination)
        self.message = f"Exportado: {destination.name}"
        self.toast = Toast("Pronóstico CSV exportado")

    def _series_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.weekly.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        key = (self.selected_series, self.selected_window)
        if key in self._series_cache:
            return self._series_cache[key]
        category, gender_short = self.selected_series.split(" · ", maxsplit=1)
        gender = "Female" if gender_short == "F" else "Male"
        actual = self.weekly.loc[
            (self.weekly["Product Category"] == category) & (self.weekly["Gender"] == gender)
        ].sort_values("week")
        predictions = self.test_predictions.loc[
            (self.test_predictions["category"] == category)
            & (self.test_predictions["gender"] == gender)
            & (self.test_predictions["context_weeks"] == self.selected_window)
        ].sort_values("week")
        metrics = self.metrics.loc[
            (self.metrics["category"] == category)
            & (self.metrics["gender"] == gender)
            & (self.metrics["context_weeks"] == self.selected_window)
        ]
        result = (
            actual.sort_values("week").reset_index(drop=True),
            predictions.sort_values("week").reset_index(drop=True),
            metrics,
        )
        self._series_cache[key] = result
        return result

    def _comparison_summary(self) -> tuple[int, int]:
        return self._comparison_lookup.get(self.selected_window, (0, 6))

    def _draw_label(self, surface, text, position, font, color) -> None:
        surface.blit(font.render(text, True, color), position)

    def draw(self, surface: pygame.Surface, content_rect: pygame.Rect, fonts, theme: Theme) -> None:
        self.layout_controls(content_rect)
        self._draw_label(
            surface,
            "PRONÓSTICO SEMANAL",
            (content_rect.x + 28, content_rect.y + 16),
            fonts["label"],
            theme.sky,
        )
        self._draw_label(
            surface,
            "Demanda · seis series",
            (content_rect.x + 28, content_rect.y + 35),
            fonts["title"],
            theme.text,
        )
        for rect in (self.control_rect, self.chart_rect, self.metrics_rect):
            draw_panel(surface, rect, theme.elevated, radius=10, border_color=theme.border)
        self._draw_controls(surface, fonts, theme)
        self._draw_chart(surface, fonts, theme)
        self._draw_metrics(surface, fonts, theme)
        if self.toast:
            self.toast.draw(surface, fonts["label"], theme)

    def _draw_controls(self, surface, fonts, theme: Theme) -> None:
        rect = self.control_rect
        x = rect.x + 12
        self._draw_label(surface, "SERIE", (x, rect.y + 12), fonts["label"], theme.muted)
        for chip in self.series_chips:
            chip.draw(surface, fonts["label"], theme)
        self._draw_label(surface, "VENTANA", (x, rect.y + 141), fonts["label"], theme.muted)
        self.context_control.draw(surface, fonts["label"], theme)
        self._draw_label(surface, "Baselines", (x, rect.y + 189), fonts["label"], theme.text)
        self.baseline_toggle.draw(surface, theme)
        self._draw_label(surface, "Zoom en test", (x, rect.y + 225), fonts["label"], theme.text)
        self.zoom_toggle.draw(surface, theme)
        self._draw_label(
            surface,
            "LSTM global · ℹ",
            (x + 8, rect.y + 263),
            fonts["label"],
            theme.sky,
        )
        self.embedding_tooltip.rect = pygame.Rect(x + 8, rect.y + 259, 92, 22)
        self.embedding_tooltip.draw(surface, fonts["label"], theme)
        if self.load_error:
            self._draw_label(
                surface, self.load_error[:28], (x, rect.y + 286), fonts["label"], theme.error
            )
        elif self.message:
            self._draw_label(
                surface, self.message[:28], (x, rect.y + 286), fonts["label"], theme.sky
            )
        self.buttons["export"].draw(surface, fonts["label"], theme)

    @staticmethod
    def _draw_dashed_line(surface, color, start, end, *, dash=8, gap=5, width=2, dotted=False):
        x1, y1 = start
        x2, y2 = end
        distance = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
        step = 3 if dotted else dash + gap
        for offset in range(0, distance, step):
            segment_end = min(distance, offset + (2 if dotted else dash))
            p0 = (
                round(x1 + (x2 - x1) * offset / distance),
                round(y1 + (y2 - y1) * offset / distance),
            )
            p1 = (
                round(x1 + (x2 - x1) * segment_end / distance),
                round(y1 + (y2 - y1) * segment_end / distance),
            )
            pygame.draw.line(surface, color, p0, p1, width)

    def _draw_chart(self, surface, fonts, theme: Theme) -> None:
        rect = self.chart_rect
        plot = self.chart_plot_rect
        actual, predictions, _ = self._series_frames()
        self._draw_label(
            surface,
            f"{self.selected_series} · unidades/semana",
            (rect.x + 14, rect.y + 12),
            fonts["label"],
            theme.text,
        )
        legend = [("Real", theme.text), ("LSTM", theme.primary)]
        if self.show_baseline:
            legend.extend((("Media móvil", theme.sky), ("Ingenuo", theme.muted)))
        cursor = rect.x + 14
        for name, color in legend:
            start, end = (cursor, rect.y + 50), (cursor + 18, rect.y + 50)
            if name == "Real":
                pygame.draw.line(surface, color, start, end, 2)
            elif name == "LSTM":
                pygame.draw.line(surface, color, start, end, 4)
                pygame.draw.circle(surface, color, (cursor + 9, rect.y + 50), 3)
            else:
                self._draw_dashed_line(
                    surface,
                    color,
                    start,
                    end,
                    dash=7,
                    gap=4,
                    width=2,
                    dotted=name == "Ingenuo",
                )
            self._draw_label(surface, name, (cursor + 23, rect.y + 42), fonts["label"], theme.muted)
            cursor += fonts["label"].size(name)[0] + 48
        if actual.empty:
            self._draw_label(
                surface,
                "Prepara los datos y entrena el LSTM",
                (plot.x + 14, plot.y + 24),
                fonts["body"],
                theme.muted,
            )
            return
        all_weeks = list(pd.to_datetime(actual["week"]))
        if self.zoom_test and not predictions.empty:
            first_test = pd.to_datetime(predictions["week"]).min()
            first_index = next((i for i, week in enumerate(all_weeks) if week >= first_test), 0)
            actual = actual.iloc[max(0, first_index - 2) :].reset_index(drop=True)
            all_weeks = list(pd.to_datetime(actual["week"]))
        values = actual["units"].astype(float).to_numpy()
        week_indices = {week: index for index, week in enumerate(all_weeks)}
        visible_predictions = predictions.loc[predictions["week"].isin(all_weeks)].copy()
        max_value = max(1.0, float(max(values, default=0)))
        if not visible_predictions.empty:
            for column in ("actual", "lstm", "moving_average", "naive"):
                if column in visible_predictions:
                    max_value = max(max_value, float(visible_predictions[column].max()))
        plot_width, plot_height = plot.width, plot.height
        test_indices = [
            week_indices[week]
            for week in pd.to_datetime(visible_predictions["week"])
            if week in week_indices
        ]
        if test_indices:
            test_x = plot.x + round(min(test_indices) * plot_width / max(1, len(values) - 1))
            pygame.draw.rect(
                surface,
                (23, 47, 89),
                pygame.Rect(test_x, plot.y, plot.right - test_x, plot.height),
                border_radius=3,
            )
            self._draw_label(surface, "Test", (test_x + 6, plot.y + 5), fonts["label"], theme.sky)
        for tick in range(5):
            tick_y = plot.y + round(plot_height * tick / 4)
            pygame.draw.line(surface, theme.border, (plot.x, tick_y), (plot.right, tick_y), 1)
            tick_value = max_value * (1 - tick / 4)
            label = fonts["label"].render(f"{tick_value:.0f}", True, theme.muted)
            surface.blit(label, label.get_rect(midright=(plot.x - 5, tick_y)))
        if len(all_weeks) > 1:
            for index in sorted({0, len(all_weeks) // 2, len(all_weeks) - 1}):
                tick_x = plot.x + round(index * plot_width / (len(all_weeks) - 1))
                label = fonts["label"].render(all_weeks[index].strftime("%b %y"), True, theme.muted)
                surface.blit(label, label.get_rect(midtop=(tick_x, plot.bottom + 4)))

        def xy(index, value):
            return (
                plot.x + round(index * plot_width / max(1, len(values) - 1)),
                plot.bottom - round(float(value) / max_value * plot_height),
            )

        actual_points = [xy(index, value) for index, value in enumerate(values)]
        if len(actual_points) > 1:
            pygame.draw.lines(surface, theme.text, False, actual_points, 2)
        if not visible_predictions.empty:
            for column, color, width, line_style in (
                ("lstm", theme.primary, 4, "solid"),
                ("moving_average", theme.sky, 2, "dashed"),
                ("naive", theme.muted, 2, "dotted"),
            ):
                if column != "lstm" and not self.show_baseline:
                    continue
                points = [
                    xy(week_indices[pd.Timestamp(row.week)], getattr(row, column))
                    for row in visible_predictions.itertuples()
                    if pd.Timestamp(row.week) in week_indices
                ]
                if len(points) > 1:
                    for first, second in zip(points[:-1], points[1:], strict=True):
                        if line_style == "solid":
                            pygame.draw.line(surface, color, first, second, width)
                        else:
                            self._draw_dashed_line(
                                surface,
                                color,
                                first,
                                second,
                                dotted=line_style == "dotted",
                                width=width,
                            )
                if column == "lstm":
                    for point in points:
                        pygame.draw.circle(surface, color, point, 4)

        category, gender_short = self.selected_series.split(" · ", maxsplit=1)
        gender = "Female" if gender_short == "F" else "Male"
        future = (
            self.future_predictions.loc[
                (self.future_predictions["category"] == category)
                & (self.future_predictions["gender"] == gender)
            ]
            if not self.future_predictions.empty
            else pd.DataFrame()
        )
        if not future.empty:
            forecast = float(future.iloc[0]["lstm"])
            marker_index = len(values)
            marker = xy(marker_index, forecast) if len(values) else (plot.right - 4, plot.centery)
            previous = actual_points[-1]
            self._draw_dashed_line(surface, theme.primary, previous, marker, dash=6, gap=4, width=2)
            pygame.draw.circle(surface, theme.sky, marker, 6)
            tag = fonts["label"].render(f"{forecast:.1f}", True, theme.text)
            surface.blit(tag, tag.get_rect(topright=(marker[0] - 8, marker[1] - 16)))
        self._draw_crosshair(surface, plot, all_weeks, values, max_value, fonts, theme)

    def _draw_crosshair(self, surface, plot, weeks, values, maximum, fonts, theme):
        mouse = pygame.mouse.get_pos()
        if not plot.collidepoint(mouse) or len(values) < 2:
            return
        index = min(
            len(values) - 1,
            max(0, round((mouse[0] - plot.x) * (len(values) - 1) / max(1, plot.width))),
        )
        px = plot.x + round(index * plot.width / max(1, len(values) - 1))
        py = plot.bottom - round(float(values[index]) / maximum * plot.height)
        pygame.draw.line(surface, theme.sky, (px, plot.y), (px, plot.bottom), 1)
        pygame.draw.circle(surface, theme.sky, (px, py), 5)
        caption = fonts["label"].render(
            f"{weeks[index]:%d %b %Y} · {values[index]:.1f} u", True, theme.text
        )
        tooltip = caption.get_rect(
            topleft=(min(plot.right - caption.get_width() - 12, px + 8), max(plot.y, py - 28))
        ).inflate(16, 10)
        draw_panel(surface, tooltip, theme.deep, radius=6, border_color=theme.border)
        surface.blit(caption, caption.get_rect(center=tooltip.center))

    def _draw_metrics(self, surface, fonts, theme):
        rect = self.metrics_rect
        x, y = rect.x + 12, rect.y + 12
        self._draw_label(surface, "TEST · MAE / RMSE", (x, y), fonts["label"], theme.muted)
        names = self.series_labels
        visible_models = ("LSTM", "Media móvil", "Ingenuo") if self.show_baseline else ("LSTM",)
        display_headers = {
            "LSTM": "LSTM",
            "Media móvil": "Media",
            "Ingenuo": "Ing.",
        }
        header = ["Serie", *(display_headers[model] for model in visible_models)]
        short_categories = {"Beauty": "Bea", "Clothing": "Clo", "Electronics": "Ele"}
        rows = []
        best_cells = set()
        for series_index, label in enumerate(names):
            category, gender_short = label.split(" · ", maxsplit=1)
            gender = "Female" if gender_short == "F" else "Male"
            cell_values = []
            mae_values = {}
            for model in visible_models:
                score = self._metric_lookup.get(
                    (self.selected_window, category, gender, model)
                )
                if score is None:
                    cell_values.append("—")
                else:
                    mae, rmse = score
                    cell_values.append(f"{mae:.1f}/{rmse:.1f}")
                    mae_values[model] = mae
            rows.append(
                [f"{short_categories.get(category, category[:3])}·{gender_short}", *cell_values]
            )
            visible_mae = {
                model: mae_values[model] for model in visible_models if model in mae_values
            }
            if visible_mae:
                best_model = min(visible_mae, key=visible_mae.get)
                best_cells.add((series_index, visible_models.index(best_model) + 1))
        average_values = []
        for model in visible_models:
            average = self._average_lookup.get((self.selected_window, model))
            if average is not None:
                average_values.append(
                    f"{average[0]:.1f}/{average[1]:.1f}"
                )
            else:
                average_values.append("—")
        rows.append(["Prom.", *average_values])
        averages = {
            model: self._average_lookup[(self.selected_window, model)][0]
            for model in visible_models
            if (self.selected_window, model) in self._average_lookup
        }
        if averages:
            average_mae = {
                model: averages[model]
                for model in visible_models
                if model in averages
            }
            best_average = min(average_mae, key=average_mae.get)
            best_average_column = visible_models.index(best_average) + 1
            best_cells.add((len(names), best_average_column))
        table_rect = pygame.Rect(x, y + 26, rect.width - 24, min(228, rect.height - 170))
        Table(table_rect, header, rows).draw(surface, fonts["label"], theme, best_cells)
        wins, total = self._comparison_summary()
        summary_y = table_rect.bottom + 6
        self._draw_label(
            surface,
            f"LSTM supera a la media móvil en {wins} de {total} series",
            (x, summary_y),
            fonts["label"],
            theme.muted,
        )
        self._draw_next_week(surface, fonts, theme, x, rect.bottom - 92)

    def _draw_next_week(self, surface, fonts, theme, x, y):
        if self.future_predictions.empty:
            return
        category, gender_short = self.selected_series.split(" · ", maxsplit=1)
        gender = "Female" if gender_short == "F" else "Male"
        future = self.future_predictions.loc[
            (self.future_predictions["category"] == category)
            & (self.future_predictions["gender"] == gender)
        ]
        actual = self.weekly.loc[
            (self.weekly["Product Category"] == category) & (self.weekly["Gender"] == gender)
        ].sort_values("week")
        if future.empty or actual.empty:
            return
        prediction = float(future.iloc[0]["lstm"])
        delta = prediction - float(actual.iloc[-1]["units"])
        card = pygame.Rect(x, y, self.metrics_rect.width - 24, 68)
        draw_panel(surface, card, theme.deep, radius=8, border_color=theme.primary)
        self._draw_label(
            surface, "PRÓXIMA SEMANA", (card.x + 9, card.y + 7), fonts["label"], theme.sky
        )
        self._draw_label(
            surface, f"{prediction:.1f} u", (card.x + 9, card.y + 28), fonts["subtitle"], theme.text
        )
        self._draw_label(
            surface,
            f"{delta:+.1f} vs anterior",
            (card.x + 93, card.y + 36),
            fonts["label"],
            theme.success if delta >= 0 else theme.error,
        )
