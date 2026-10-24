"""Primitivas simples de layout ponderado para interfaces Pygame."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class Track:
    """Una pista de layout con peso relativo y tamaño mínimo."""

    weight: float = 1.0
    minimum: int = 0


class WeightedLayout:
    """Distribuye filas o columnas dentro de un rectángulo padre sin desbordar."""

    def __init__(self, *, gap: int = 8, padding: int = 0):
        self.gap = max(0, int(gap))
        self.padding = max(0, int(padding))

    def columns(self, rect: pygame.Rect, tracks: list[Track]) -> list[pygame.Rect]:
        """Parte el ancho entre columnas por peso, respetando mínimos posibles."""
        return self._split(rect, tracks, horizontal=True)

    def rows(self, rect: pygame.Rect, tracks: list[Track]) -> list[pygame.Rect]:
        """Parte el alto entre filas por peso, respetando mínimos posibles."""
        return self._split(rect, tracks, horizontal=False)

    def _split(
        self,
        rect: pygame.Rect,
        tracks: list[Track],
        *,
        horizontal: bool,
    ) -> list[pygame.Rect]:
        if not tracks:
            return []
        inner = rect.inflate(-2 * self.padding, -2 * self.padding)
        total = inner.width if horizontal else inner.height
        gap_total = self.gap * (len(tracks) - 1)
        available = max(0, total - gap_total)
        minimums = [max(0, track.minimum) for track in tracks]
        minimum_total = sum(minimums)
        if minimum_total > available and minimum_total:
            sizes = [int(available * value / minimum_total) for value in minimums]
            sizes[-1] += available - sum(sizes)
        else:
            remaining = available - minimum_total
            weights = [max(0.0, track.weight) for track in tracks]
            total_weight = sum(weights) or float(len(weights))
            sizes = [
                minimums[index] + int(remaining * ((weights[index] or 1.0) / total_weight))
                for index in range(len(tracks))
            ]
            sizes[-1] += available - sum(sizes)

        result = []
        cursor = inner.x if horizontal else inner.y
        for size in sizes:
            if horizontal:
                result.append(pygame.Rect(cursor, inner.y, size, inner.height))
            else:
                result.append(pygame.Rect(inner.x, cursor, inner.width, size))
            cursor += size + self.gap
        return result


class ScrollArea:
    """Viewport con desplazamiento vertical discreto por rueda."""

    def __init__(self, viewport: pygame.Rect, content_height: int, scroll: int = 0):
        self.viewport = viewport.copy()
        self.content_height = max(viewport.height, int(content_height))
        self.scroll = max(0, min(int(scroll), self.max_scroll))

    @property
    def max_scroll(self) -> int:
        """Desplazamiento máximo para no sacar el contenido del viewport."""
        return max(0, self.content_height - self.viewport.height)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Desplaza contenido bajo el puntero y limita el rango."""
        if event.type == pygame.MOUSEWHEEL and self.viewport.collidepoint(pygame.mouse.get_pos()):
            self.scroll = max(0, min(self.max_scroll, self.scroll - event.y * 32))
            return True
        return False

    def content_rect(self, rect: pygame.Rect) -> pygame.Rect:
        """Traslada un rectángulo de contenido al viewport y devuelve el resultado."""
        moved = rect.move(self.viewport.x - rect.x, self.viewport.y - rect.y - self.scroll)
        return moved.clip(self.viewport)
