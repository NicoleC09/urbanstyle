"""Widgets Pygame con tokens y estados del sistema UrbanStyle."""

from __future__ import annotations

from collections.abc import Sequence

import pygame

from app.ui.theme import Theme, draw_panel, draw_vector_icon


class Button:
    """Botón primary, secondary o ghost con icono vectorial opcional."""

    def __init__(self, rect, label, callback=None, *, kind="secondary", icon=None):
        if kind not in {"primary", "secondary", "ghost"}:
            raise ValueError("kind debe ser primary, secondary o ghost.")
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.kind = kind
        self.icon = icon
        self.enabled = True
        self.pressed = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.pressed = self.enabled and self.rect.collidepoint(event.pos)
            return self.pressed
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and self.enabled and self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
                return True
        return False

    def draw(self, surface, font, theme: Theme) -> None:
        hovered = self.enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        rect = self.rect.inflate(-2, -2) if self.pressed and self.enabled else self.rect
        if not self.enabled:
            fill, text_color, border, shadow = theme.panel, theme.disabled, theme.border, False
        elif self.kind == "primary":
            fill = theme.hover if hovered else theme.primary
            text_color, border, shadow = theme.text, fill, not self.pressed
        elif self.kind == "secondary":
            fill = theme.elevated if hovered else theme.panel
            text_color = theme.text
            border = theme.primary if hovered else theme.border
            shadow = False
        else:
            fill = theme.elevated if hovered else theme.panel
            text_color = theme.hover if hovered else theme.sky
            border, shadow = theme.border if hovered else None, False
        if self.kind != "ghost" or hovered or not self.enabled:
            draw_panel(surface, rect, fill, radius=10, shadow=shadow, border_color=border)
        rendered = font.render(self.label, True, text_color)
        if self.icon:
            draw_vector_icon(surface, self.icon, (rect.x + 18, rect.centery), text_color)
            surface.blit(rendered, (rect.x + 34, rendered.get_rect(centery=rect.centery).y))
        else:
            surface.blit(rendered, rendered.get_rect(center=rect.center))


class SegmentedControl:
    """Selector horizontal para opciones mutuamente excluyentes."""

    def __init__(self, rect, options: Sequence[str], selected=0, on_change=None):
        if len(options) < 2:
            raise ValueError("SegmentedControl requiere al menos dos opciones.")
        self.rect = pygame.Rect(rect)
        self.options = list(options)
        self.selected = min(max(int(selected), 0), len(self.options) - 1)
        self.on_change = on_change
        self.enabled = True

    @property
    def value(self):
        return self.options[self.selected]

    @property
    def rects(self):
        width = self.rect.width // len(self.options)
        return [
            pygame.Rect(
                self.rect.x + index * width,
                self.rect.y,
                width if index < len(self.options) - 1 else self.rect.width - width * index,
                self.rect.height,
            )
            for index in range(len(self.options))
        ]

    def handle_event(self, event):
        if not self.enabled or event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return False
        for index, rect in enumerate(self.rects):
            if rect.collidepoint(event.pos):
                self.selected = index
                if self.on_change:
                    self.on_change(index)
                return True
        return False

    def draw(self, surface, font, theme: Theme):
        draw_panel(surface, self.rect, theme.panel, radius=10, border_color=theme.border)
        for index, (option, rect) in enumerate(zip(self.options, self.rects, strict=True)):
            if index == self.selected:
                draw_panel(surface, rect.inflate(-4, -4), theme.primary, radius=8)
            color = theme.text if self.enabled and index == self.selected else theme.muted
            if not self.enabled:
                color = theme.disabled
            rendered = font.render(option, True, color)
            surface.blit(rendered, rendered.get_rect(center=rect.center))


class Chip:
    """Chip seleccionable para filtros, series o categorías."""

    def __init__(self, rect, label, selected=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.selected = selected
        self.enabled = True

    def handle_event(self, event):
        if (
            self.enabled
            and event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        ):
            self.selected = not self.selected
            return True
        return False

    def draw(self, surface, font, theme: Theme):
        hovered = self.enabled and self.rect.collidepoint(pygame.mouse.get_pos())
        fill = theme.primary if self.selected else theme.elevated if hovered else theme.panel
        border = theme.primary if self.selected or hovered else theme.border
        color = theme.text if self.enabled else theme.disabled
        draw_panel(surface, self.rect, fill, radius=10, border_color=border)
        rendered = font.render(self.label, True, color)
        surface.blit(rendered, rendered.get_rect(center=self.rect.center))


class Slider:
    """Slider con formateador opcional y label/value de un solo lugar."""

    def __init__(self, rect, minimum, maximum, value, *, label="", formatter=None):
        self.rect = pygame.Rect(rect)
        self.minimum = minimum
        self.maximum = maximum
        self.value = min(maximum, max(minimum, value))
        self.dragging = False
        self.label = label
        self.formatter = formatter or (lambda current: f"{current:.0f}")
        self.enabled = True
        self.hovered = False

    def handle_event(self, event):
        if not self.enabled:
            return False
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.inflate(0, 20).collidepoint(event.pos)
        ):
            self.dragging = True
            self._set_from_x(event.pos[0])
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_dragging = self.dragging
            self.dragging = False
            return was_dragging
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.inflate(0, 20).collidepoint(event.pos)
            if self.dragging:
                self._set_from_x(event.pos[0])
                return True
        return False

    def _set_from_x(self, x):
        fraction = min(1.0, max(0.0, (x - self.rect.left) / max(1, self.rect.width)))
        self.value = self.minimum + fraction * (self.maximum - self.minimum)

    def draw(self, surface, font, theme: Theme, *, label=None):
        text_label = self.label if label is None else label
        if text_label:
            rendered = font.render(text_label, True, theme.muted)
            surface.blit(rendered, (self.rect.left, self.rect.top - rendered.get_height() - 8))
        track = theme.border if self.enabled else theme.disabled
        pygame.draw.line(surface, track, self.rect.midleft, self.rect.midright, 4)
        ratio = (self.value - self.minimum) / max(1e-9, self.maximum - self.minimum)
        thumb_x = round(self.rect.left + ratio * self.rect.width)
        pygame.draw.line(
            surface,
            theme.primary if self.enabled else theme.disabled,
            self.rect.midleft,
            (thumb_x, self.rect.centery),
            4,
        )
        value = font.render(
            self.formatter(self.value), True, theme.text if self.enabled else theme.disabled
        )
        value_rect = value.get_rect(midright=(self.rect.right + 38, self.rect.centery))
        value_rect.clamp_ip(surface.get_rect())
        surface.blit(value, value_rect)
        if self.enabled and (self.hovered or self.dragging):
            halo = pygame.Surface((28, 28), pygame.SRCALPHA)
            halo_color = (theme.primary.r, theme.primary.g, theme.primary.b, 56)
            pygame.draw.circle(halo, halo_color, (14, 14), 13)
            surface.blit(halo, halo.get_rect(center=(thumb_x, self.rect.centery)))
        pygame.draw.circle(
            surface, theme.text if self.enabled else theme.disabled, (thumb_x, self.rect.centery), 8
        )


class Toggle:
    """Interruptor binario con animación de posición."""

    def __init__(self, rect, value=False):
        self.rect = pygame.Rect(rect)
        self.value = value
        self.position = float(value)
        self.enabled = True

    def handle_event(self, event):
        if (
            self.enabled
            and event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        ):
            self.value = not self.value
            return True
        return False

    def draw(self, surface, theme: Theme, delta_seconds=1 / 60):
        target = float(self.value)
        self.position += (target - self.position) * min(1.0, delta_seconds * 14)
        fill = theme.primary if self.value and self.enabled else theme.border
        draw_panel(
            surface, self.rect, fill, radius=self.rect.height // 2, border_color=theme.border
        )
        thumb_x = (
            self.rect.x
            + self.rect.height // 2
            + round(self.position * (self.rect.width - self.rect.height))
        )
        pygame.draw.circle(
            surface,
            theme.text if self.enabled else theme.disabled,
            (thumb_x, self.rect.centery),
            max(1, self.rect.height // 2 - 3),
        )


class Dropdown:
    """Selector compacto con flecha vectorial para listas extensas."""

    def __init__(self, rect, options: Sequence[str], selected=0):
        if not options:
            raise ValueError("Dropdown necesita al menos una opción.")
        self.rect = pygame.Rect(rect)
        self.options = list(options)
        self.selected = min(max(selected, 0), len(self.options) - 1)
        self.open = False
        self.enabled = True
        self.scroll_offset = 0
        self.popup_rows = 7

    @property
    def value(self):
        return self.options[self.selected]

    def handle_event(self, event):
        if not self.enabled:
            return False
        if self.open and event.type == pygame.MOUSEWHEEL:
            pointer = getattr(event, "pos", pygame.mouse.get_pos())
            if not self._popup_rect().collidepoint(pointer):
                return False
            self.scroll_offset = min(
                max(0, len(self.options) - self.popup_rows),
                max(0, self.scroll_offset - event.y),
            )
            return True
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return False
        if self.rect.collidepoint(event.pos):
            self.open = not self.open
            return True
        if self.open:
            for index in range(
                self.scroll_offset,
                min(len(self.options), self.scroll_offset + self.popup_rows),
            ):
                option = self._option_rect(index)
                if option.collidepoint(event.pos):
                    self.selected = index
                    self.open = False
                    return True
            self.open = False
        return False

    def _option_rect(self, index):
        row_height = min(self.rect.height, 32)
        return pygame.Rect(
            self.rect.x,
            self.rect.bottom + row_height * (index - self.scroll_offset),
            self.rect.width,
            row_height,
        )

    def _popup_rect(self):
        return pygame.Rect(
            self.rect.x,
            self.rect.bottom,
            self.rect.width,
            self.popup_rows * min(self.rect.height, 32),
        )

    def draw(self, surface, font, theme: Theme):
        draw_panel(surface, self.rect, theme.panel, radius=10, border_color=theme.border)
        rendered = font.render(self.value, True, theme.text if self.enabled else theme.disabled)
        surface.blit(rendered, (self.rect.x + 14, self.rect.centery - rendered.get_height() // 2))
        x = self.rect.right - 18
        y = self.rect.centery + (3 if self.open else -2)
        pygame.draw.polygon(surface, theme.sky, ((x - 5, y - 2), (x + 5, y - 2), (x, y + 4)))
        if self.open:
            clip = surface.get_clip()
            popup_clip = self._popup_rect().clip(surface.get_rect())
            surface.set_clip(popup_clip)
            for index in range(
                self.scroll_offset,
                min(len(self.options), self.scroll_offset + self.popup_rows),
            ):
                option = self.options[index]
                rect = self._option_rect(index)
                draw_panel(surface, rect, theme.panel, radius=8, border_color=theme.border)
                text = font.render(option, True, theme.text)
                surface.blit(text, (rect.x + 14, rect.centery - text.get_height() // 2))
            surface.set_clip(clip)


class Tabs:
    """Barra de pestañas con indicador animado y hitboxes estables."""

    def __init__(self, labels: Sequence[str], active=0):
        self.labels = list(labels)
        self.active = active
        self.rects: list[pygame.Rect] = []
        self.indicator_x: float | None = None

    def layout(self, rect, gap=8):
        if not self.labels:
            self.rects = []
            return
        width = (rect.width - gap * (len(self.labels) - 1)) // len(self.labels)
        self.rects = [
            pygame.Rect(rect.x + i * (width + gap), rect.y, width, rect.height)
            for i in range(len(self.labels))
        ]

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for index, rect in enumerate(self.rects):
                if rect.collidepoint(event.pos):
                    self.active = index
                    return True
        return False

    def draw(self, surface, font, theme: Theme):
        for index, (label, rect) in enumerate(zip(self.labels, self.rects, strict=True)):
            active = index == self.active
            draw_panel(
                surface,
                rect,
                theme.elevated if active else theme.panel,
                radius=12,
                border_color=theme.border,
            )
            text = font.render(label, True, theme.text if active else theme.muted)
            surface.blit(text, text.get_rect(center=rect.center))
        if self.rects:
            target = float(self.rects[self.active].centerx)
            self.indicator_x = (
                target
                if self.indicator_x is None
                else self.indicator_x + (target - self.indicator_x) * 0.32
            )
            indicator = pygame.Rect(
                0,
                self.rects[self.active].bottom - 4,
                max(24, self.rects[self.active].width - 36),
                3,
            )
            indicator.centerx = round(self.indicator_x)
            pygame.draw.rect(surface, theme.primary, indicator, border_radius=2)


class Card:
    """Tarjeta reutilizable con borde, sombra suave y dos niveles de texto."""

    def __init__(self, rect, title, body=""):
        self.rect, self.title, self.body = pygame.Rect(rect), title, body

    def draw(self, surface, title_font, body_font, theme: Theme):
        draw_panel(
            surface, self.rect, theme.elevated, radius=16, shadow=True, border_color=theme.border
        )
        surface.blit(
            title_font.render(self.title, True, theme.text), (self.rect.x + 20, self.rect.y + 18)
        )
        if self.body:
            surface.blit(
                body_font.render(self.body, True, theme.muted), (self.rect.x + 20, self.rect.y + 54)
            )


class KpiCard:
    """Tarjeta KPI con valor principal, etiqueta y tendencia opcional."""

    def __init__(self, rect, label, value="—", trend=""):
        self.rect, self.label, self.value, self.trend = pygame.Rect(rect), label, value, trend

    def draw(self, surface, label_font, value_font, theme: Theme):
        draw_panel(
            surface, self.rect, theme.elevated, radius=16, shadow=True, border_color=theme.border
        )
        surface.blit(
            label_font.render(self.label.upper(), True, theme.muted),
            (self.rect.x + 16, self.rect.y + 14),
        )
        surface.blit(
            value_font.render(self.value, True, theme.text), (self.rect.x + 16, self.rect.y + 38)
        )
        if self.trend:
            surface.blit(
                label_font.render(self.trend, True, theme.sky),
                (self.rect.right - 48, self.rect.y + 16),
            )


class ProgressBar:
    """Barra de progreso acotada al intervalo [0,1]."""

    def __init__(self, rect, value=0.0):
        self.rect = pygame.Rect(rect)
        self.value = min(1.0, max(0.0, float(value)))

    def draw(self, surface, theme: Theme):
        draw_panel(
            surface, self.rect, theme.panel, radius=self.rect.height // 2, border_color=theme.border
        )
        fill = self.rect.copy()
        fill.width = round(self.rect.width * self.value)
        if fill.width > 0:
            draw_panel(surface, fill, theme.primary, radius=self.rect.height // 2)


class EmptyState:
    """Estado vacío con icono vectorial y acción opcional."""

    def __init__(self, rect, message, icon="info", action: Button | None = None):
        self.rect, self.message, self.icon, self.action = pygame.Rect(rect), message, icon, action

    def draw(self, surface, font, theme: Theme):
        draw_vector_icon(
            surface, self.icon, (self.rect.centerx, self.rect.centery - 26), theme.disabled, 36, 2
        )
        text = font.render(self.message, True, theme.muted)
        surface.blit(text, text.get_rect(center=(self.rect.centerx, self.rect.centery + 10)))
        if self.action:
            self.action.draw(surface, font, theme)


class Tooltip:
    """Tooltip de texto breve que se muestra al pasar el cursor."""

    def __init__(self, rect, text):
        self.rect, self.text = pygame.Rect(rect), text

    def draw(self, surface, font, theme: Theme, mouse_pos=None):
        point = mouse_pos or pygame.mouse.get_pos()
        if not self.rect.collidepoint(point):
            return
        text = font.render(self.text, True, theme.text)
        panel = text.get_rect(topleft=(point[0] + 12, point[1] + 14)).inflate(24, 16)
        panel.clamp_ip(surface.get_rect())
        draw_panel(
            surface, panel, theme.elevated, radius=10, shadow=True, border_color=theme.border
        )
        surface.blit(text, text.get_rect(center=panel.center))


class Toast:
    """Notificación temporal en la esquina inferior derecha."""

    def __init__(self, message, duration_ms=2400):
        self.message = message
        self.duration_ms = duration_ms
        self.started_at = pygame.time.get_ticks()

    @property
    def visible(self):
        return pygame.time.get_ticks() - self.started_at < self.duration_ms

    def draw(self, surface, font, theme: Theme):
        if not self.visible:
            return
        label = font.render(self.message, True, theme.text)
        rect = label.get_rect(
            bottomright=(surface.get_width() - 24, surface.get_height() - 20)
        ).inflate(32, 18)
        rect.clamp_ip(surface.get_rect())
        draw_panel(
            surface, rect, theme.elevated, radius=12, shadow=True, border_color=theme.primary
        )
        surface.blit(label, label.get_rect(center=rect.center))


class Table:
    """Tabla sencilla con cabecera y celdas de mejor resultado resaltadas."""

    def __init__(self, rect, headers: Sequence[str], rows: Sequence[Sequence[str]]):
        self.rect, self.headers = pygame.Rect(rect), list(headers)
        self.rows = [list(row) for row in rows]

    def draw(self, surface, font, theme: Theme, best_cells=None):
        best_cells = best_cells or set()
        if not self.headers:
            return
        width = self.rect.width / len(self.headers)
        row_height = min(38, max(24, self.rect.height // (len(self.rows) + 1)))
        for row_index, row in enumerate([self.headers, *self.rows]):
            y = self.rect.y + row_index * row_height
            for col_index, value in enumerate(row[: len(self.headers)]):
                cell = pygame.Rect(
                    round(self.rect.x + col_index * width), y, round(width), row_height
                )
                if row_index == 0:
                    pygame.draw.rect(surface, theme.deep, cell, border_radius=4)
                    color = theme.text
                elif (row_index - 1, col_index) in best_cells:
                    pygame.draw.rect(surface, theme.primary, cell, border_radius=4)
                    color = theme.text
                else:
                    color = theme.muted if row_index % 2 == 0 else theme.text
                rendered = font.render(str(value), True, color)
                surface.blit(rendered, rendered.get_rect(midleft=(cell.x + 8, cell.centery)))
