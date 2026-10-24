"""Punto de entrada para la interfaz de escritorio UrbanStyle."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.utils.config import load_config
from src.urbanstyle.utils.logging import configure_logging

from app.tabs import TABS
from app.tabs.home_tab import HomeTab
from app.tabs.task_a_tab import TaskATab
from app.tabs.task_b_tab import TaskBTab
from app.tabs.task_c_tab import TaskCTab
from app.tabs.task_d_tab import TaskDTab
from app.ui.theme import Theme, draw_panel, load_fonts
from app.ui.widgets import Tabs, Toast

LOGGER = logging.getLogger("urbanstyle.app")


def _draw_text(
    surface: pygame.Surface,
    text: str,
    position: tuple[int, int],
    font: pygame.font.Font,
    color: pygame.Color,
) -> None:
    """Pinta texto alineado a la izquierda."""
    surface.blit(font.render(text, True, color), position)


def _draw_header_chips(
    surface: pygame.Surface,
    width: int,
    font: pygame.font.Font,
    theme: Theme,
    config: dict,
    pricing_tab: TaskCTab,
    campaign_tab: TaskDTab,
) -> None:
    """Dibuja estado de dispositivo/modelos y versión en chips compactos."""
    try:
        import torch

        if torch.cuda.is_available():
            device_name = "GPU"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_name = "MPS"
        else:
            device_name = "CPU"
    except ImportError:
        device_name = "CPU"
    model_dir = PROJECT_ROOT / config["paths"]["models"]
    models = (
        ("LSTM", (model_dir / "lstm.pt").is_file()),
        ("PPO", pricing_tab.model_status.get("PPO") == "cargado"),
        ("DQN", pricing_tab.model_status.get("DQN") == "cargado"),
        ("cGAN", campaign_tab.generator is not None),
    )
    labels = [(device_name, None), *models, (f"v{config['app']['version']}", None)]
    gap = 6
    chips = []
    for label, loaded in labels:
        text = label
        rendered = font.render(text, True, theme.text)
        chip_width = rendered.get_width() + (30 if loaded is not None else 20)
        chips.append((text, loaded, max(44, chip_width), rendered))
    right = width - 36
    for _label, loaded, chip_width, rendered in reversed(chips):
        rect = pygame.Rect(right - chip_width, 32, chip_width, 28)
        border = theme.error if loaded is False else theme.border
        draw_panel(surface, rect, theme.panel, radius=10, border_color=border)
        text_x = rect.x + 10
        if loaded is not None:
            dot_color = theme.success if loaded else theme.error
            pygame.draw.circle(surface, dot_color, (rect.x + 12, rect.centery), 4)
            text_x = rect.x + 21
        surface.blit(rendered, rendered.get_rect(midleft=(text_x, rect.centery)))
        right = rect.left - gap


def render_frame(
    screen: pygame.Surface,
    config: dict,
    fonts: dict[str, pygame.font.Font],
    theme: Theme,
    tabs: Tabs,
    home_tab: HomeTab,
    task_a_tab: TaskATab,
    task_b_tab: TaskBTab,
    task_c_tab: TaskCTab,
    task_d_tab: TaskDTab,
    *,
    delta_seconds: float = 1 / 60,
    toast: Toast | None = None,
    help_visible: bool = False,
) -> None:
    """Dibuja un frame completo, reutilizable por la app y las capturas QA."""
    width, height = screen.get_size()
    screen.fill(theme.background)
    margin = 36
    _draw_text(screen, "URBANSTYLE", (margin, 25), fonts["brand"], theme.accent)
    _draw_text(
        screen,
        "RETAIL INTELLIGENCE  /  LABORATORIO DE DECISIONES",
        (margin, 54),
        fonts["small"],
        theme.muted,
    )
    _draw_header_chips(screen, width, fonts["label"], theme, config, task_c_tab, task_d_tab)

    tabs.layout(pygame.Rect(margin, 86, width - 2 * margin, 54))
    tabs.draw(screen, fonts["tab"], theme)
    current = TABS[tabs.active]
    content_rect = pygame.Rect(margin, 174, width - 2 * margin, height - 210)
    task_d_tab.update()

    if tabs.active == 0:
        home_tab.update(delta_seconds)
        home_tab.draw(screen, content_rect, fonts, theme)
    elif tabs.active == 1:
        task_a_tab.update(delta_seconds)
        task_a_tab.draw(screen, content_rect, fonts, theme)
    elif tabs.active == 2:
        task_b_tab.draw(screen, content_rect, fonts, theme)
    elif tabs.active == 3:
        task_c_tab.update(delta_seconds)
        task_c_tab.draw(screen, content_rect, fonts, theme)
    elif tabs.active == 4:
        task_d_tab.draw(screen, content_rect, fonts, theme)
    else:
        _draw_text(
            screen,
            current.title,
            (content_rect.x + 32, content_rect.y + 32),
            fonts["title"],
            theme.foreground,
        )
        _draw_text(
            screen,
            current.description,
            (content_rect.x + 34, content_rect.y + 82),
            fonts["body"],
            theme.muted,
        )
        tag = fonts["small"].render("EN PREPARACIÓN", True, theme.background)
        tag_rect = tag.get_rect(topleft=(content_rect.x + 34, content_rect.y + 150)).inflate(
            24, 14
        )
        draw_panel(screen, tag_rect, theme.accent, radius=7)
        screen.blit(tag, tag.get_rect(center=tag_rect.center))

    _draw_text(
        screen,
        "1–5 pestañas     F11 pantalla completa     Ctrl+S captura     Esc salir",
        (margin, height - 25),
        fonts["small"],
        theme.muted,
    )
    if toast:
        toast.draw(screen, fonts["small"], theme)
    if help_visible:
        _draw_help(screen, fonts, theme)


def run() -> None:
    """Inicializa Pygame y ejecuta el bucle interactivo."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    app_config = config["app"]
    pygame.init()
    pygame.display.set_caption(app_config["title"])
    flags = pygame.RESIZABLE
    screen = pygame.display.set_mode(
        (app_config["window_width"], app_config["window_height"]), flags
    )
    clock = pygame.time.Clock()
    theme = Theme.from_config(config)
    fonts = load_fonts(config, PROJECT_ROOT)
    tabs = Tabs([tab.label for tab in TABS])
    home_tab = HomeTab(config, PROJECT_ROOT)
    task_a_tab = TaskATab(config, PROJECT_ROOT)
    task_b_tab = TaskBTab(config, PROJECT_ROOT)
    task_c_tab = TaskCTab(config, PROJECT_ROOT)
    task_d_tab = TaskDTab(config, PROJECT_ROOT)
    toast: Toast | None = None
    fullscreen = False
    help_visible = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and help_visible:
                if event.key in (pygame.K_ESCAPE, pygame.K_F1, pygame.K_h):
                    help_visible = False
                continue
            elif event.type == pygame.VIDEORESIZE and not fullscreen:
                width = max(app_config["min_width"], event.w)
                height = max(app_config["min_height"], event.h)
                screen = pygame.display.set_mode((width, height), flags)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_F1, pygame.K_h):
                    help_visible = not help_visible
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    tabs.active = event.key - pygame.K_1
                elif event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode(
                            (app_config["window_width"], app_config["window_height"]), flags
                        )
                elif event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
                    try:
                        output_dir = PROJECT_ROOT / config["paths"]["outputs"]
                        output_dir.mkdir(parents=True, exist_ok=True)
                        screenshot_path = output_dir / "captura_urbanstyle.png"
                        pygame.image.save(screen, screenshot_path)
                        toast = Toast(f"Captura guardada: {screenshot_path.name}")
                    except (OSError, pygame.error) as error:
                        LOGGER.exception("No se pudo guardar la captura: %s", error)
                        toast = Toast("No se pudo guardar la captura")
            if help_visible:
                continue
            tabs.handle_event(event)
            if tabs.active == 0:
                home_tab.handle_event(event)
            elif tabs.active == 1:
                task_a_tab.handle_event(event)
            elif tabs.active == 2:
                task_b_tab.handle_event(event)
            elif tabs.active == 3:
                task_c_tab.handle_event(event)
            elif tabs.active == 4:
                task_d_tab.handle_event(event)

        delta_seconds = min(0.1, clock.get_time() / 1000)
        render_frame(
            screen,
            config,
            fonts,
            theme,
            tabs,
            home_tab,
            task_a_tab,
            task_b_tab,
            task_c_tab,
            task_d_tab,
            delta_seconds=delta_seconds,
            toast=toast,
            help_visible=help_visible,
        )
        pygame.display.flip()
        clock.tick(app_config["fps"])

    home_tab.close()
    task_d_tab.close()
    pygame.quit()


def main() -> int:
    """Arranque con diagnóstico legible si la configuración de Pygame falla."""
    configure_logging()
    try:
        run()
    except (pygame.error, OSError, KeyError, ValueError, RuntimeError, ImportError) as error:
        LOGGER.exception("No se pudo iniciar UrbanStyle: %s", error)
        return 1
    return 0


def _draw_help(
    surface: pygame.Surface,
    fonts: dict[str, pygame.font.Font],
    theme: Theme,
) -> None:
    """Dibuja una ayuda modal concisa con navegación y controles globales."""
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    surface.blit(overlay, (0, 0))
    width = min(620, surface.get_width() - 64)
    height = 390
    rect = pygame.Rect(0, 0, width, height)
    rect.center = surface.get_rect().center
    draw_panel(surface, rect, theme.surface, radius=14, shadow=True)
    _draw_text(surface, "AYUDA RÁPIDA", (rect.x + 28, rect.y + 24), fonts["brand"], theme.accent)
    lines = (
        "1  Inicio   ·   escenario de negocio completo",
        "2  A        ·   rutas, clientes y congestión",
        "3  B        ·   pronóstico semanal y baselines",
        "4  C        ·   precios dinámicos y políticas",
        "5  D        ·   imágenes Fashion-MNIST y póster",
        "",
        "Espacio  Pausar/reanudar en A y C",
        "R        Reiniciar la simulación activa",
        "F11      Pantalla completa",
        "Ctrl+S   Guardar captura en outputs/",
        "F1 / H   Abrir o cerrar esta ayuda",
        "Esc      Cerrar ayuda o salir",
    )
    for index, line in enumerate(lines):
        _draw_text(
            surface,
            line,
            (rect.x + 30, rect.y + 72 + index * 24),
            fonts["small"],
            theme.muted if not line else theme.foreground,
        )


if __name__ == "__main__":
    raise SystemExit(main())
