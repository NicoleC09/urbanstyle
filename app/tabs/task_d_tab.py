"""Campañas de Fashion-MNIST y pósteres publicitarios UrbanStyle."""

from __future__ import annotations

import json
import logging
import random
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pygame
import torch
from PIL import Image
from src.urbanstyle.task_c_pricing.data import (
    load_forecast_units,
    load_product_data,
    make_environment,
)
from src.urbanstyle.task_d_gan.classifier import FashionClassifier
from src.urbanstyle.task_d_gan.generate import (
    FashionGenerator,
    generator_checkpoint,
    load_fashion_classifier,
)
from src.urbanstyle.task_d_gan.poster import (
    TEMPLATES,
    agent_offer_copy,
    campaign_offer_copy,
    create_campaign_poster,
    rank_generated_images,
)
from src.urbanstyle.task_d_gan.train import FASHION_CLASSES
from src.urbanstyle.utils.config import resolve_path
from stable_baselines3 import DQN, PPO

from app.ui.layout import Track, WeightedLayout
from app.ui.theme import Theme, draw_panel, draw_vector_icon
from app.ui.widgets import Button, Chip, SegmentedControl, Slider, Toast

LOGGER = logging.getLogger("urbanstyle.app.task_d")
MODEL_TYPES = {"PPO": PPO, "DQN": DQN}
HISTORY_TABS = ("Muestras", "Curvas de pérdida", "Calidad")


class TaskDTab:
    """Genera prendas, permite elegir un héroe y compone pósteres 4:5."""

    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        task_config = config["task_d"]
        self.classes = FASHION_CLASSES[: int(task_config["classes"])]
        self.categories = list(config["data"]["categories"])
        self.category_control = SegmentedControl(pygame.Rect(0, 0, 280, 34), self.categories)
        self.category_maps = task_config["category_class_map"]
        self.class_chips = [
            Chip(pygame.Rect(0, 0, 90, 34), name, selected=index == 7)
            for index, name in enumerate(self.classes)
        ]
        self.class_id = 7
        self.seed_value = int(config["project"]["seed"])
        self.seed_text = str(self.seed_value)
        self.seed_focused = False
        self.count_slider = Slider(pygame.Rect(0, 0, 180, 4), 1, 9, 4)
        self.variety_slider = Slider(
            pygame.Rect(0, 0, 180, 4),
            0.2,
            1.2,
            float(task_config["truncation"]),
            formatter=lambda value: f"{value:.1f}",
        )
        self.discount_slider = Slider(
            pygame.Rect(0, 0, 180, 4),
            0,
            70,
            int(task_config["discount_default"]),
            formatter=lambda value: f"{round(value)}%",
        )
        self.template_control = SegmentedControl(
            pygame.Rect(0, 0, 150, 32), ("Noche", "Hielo", "Eléc.")
        )
        self.style_control = SegmentedControl(pygame.Rect(0, 0, 100, 32), ("Neón", "Pixel"))
        self.generator: FashionGenerator | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="urbanstyle-campaign")
        self._future: Future | None = None
        self._future_action: str | None = None
        self._followup_action: str | None = None
        self.classifier: FashionClassifier | None = None
        self.model_error: str | None = None
        try:
            self.generator = FashionGenerator(generator_checkpoint(config, "cgan"))
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.model_error = str(error)
            LOGGER.warning("No se pudo cargar cGAN: %s", error)
        classifier_path = resolve_path(config["paths"]["models"]) / "fmnist_classifier.pt"
        if classifier_path.is_file():
            try:
                self.classifier = load_fashion_classifier(classifier_path)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                LOGGER.warning("No se pudo cargar el clasificador de calidad: %s", error)

        self.images: torch.Tensor | None = None
        self._generated_surface_cache: dict[tuple[int, int], pygame.Surface] = {}
        self._image_surface_cache: dict[tuple[str, int, tuple[int, int]], pygame.Surface] = {}
        self.hero_index = 0
        self.poster_path: Path | None = None
        self.message = ""
        self.toast: Toast | None = None
        self.modal_open = False
        self.history_tab = 0
        self.epoch_slider = Slider(pygame.Rect(0, 0, 240, 4), 1, int(task_config["epochs"]), 80)
        self.training_dir = resolve_path(config["paths"]["outputs"]) / "gan"
        self.epoch_paths = self._load_epoch_paths()
        self.epoch_values = sorted(self.epoch_paths)
        self.epoch = self.epoch_values[-1] if self.epoch_values else 1
        self.loss_history = self._load_json(self.training_dir / "losses.json")
        self.quality_history = self._load_json(self.training_dir / "metrics.json")
        self.curves_path = self.training_dir / "cgan" / "losses.csv"
        self.poster_product_name = self.classes[self.class_id]
        self.offer_copy, self.discount_percent = self._price_copy()
        self.discount_slider.value = self.discount_percent
        self.manual_discount_override = False

        self.control_rect = pygame.Rect(0, 0, 0, 0)
        self.preview_rect = pygame.Rect(0, 0, 0, 0)
        self.gallery_rect = pygame.Rect(0, 0, 0, 0)
        self.seed_rect = pygame.Rect(0, 0, 0, 0)
        self.dice_button = Button(pygame.Rect(0, 0, 38, 34), "", self.random_seed, icon="dice")
        self.buttons = {
            "generate": Button(
                pygame.Rect(0, 0, 120, 38), "Generar", self.generate, kind="primary"
            ),
            "random": Button(pygame.Rect(0, 0, 110, 38), "Aleatorio", self.random_generate),
            "poster": Button(
                pygame.Rect(0, 0, 180, 40), "Crear póster", self.create_poster, kind="primary"
            ),
            "download": Button(
                pygame.Rect(0, 0, 160, 36), "Descargar PNG", self.download_poster, icon="download"
            ),
            "large": Button(pygame.Rect(0, 0, 150, 36), "Ver en grande", self.open_modal),
        }
        self.history_buttons = [
            Button(
                pygame.Rect(0, 0, 92, 30), label, lambda index=index: self._set_history_tab(index)
            )
            for index, label in enumerate(HISTORY_TABS)
        ]
        self._sync_class_availability()

    @staticmethod
    def _load_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_epoch_paths(self) -> dict[int, Path]:
        paths = {}
        for path in self.training_dir.glob("samples_epoch_*.png"):
            try:
                paths[int(path.stem.rsplit("_", maxsplit=1)[1])] = path
            except ValueError:
                continue
        class_dir = self.training_dir / "cgan"
        for path in class_dir.glob("epoch_*.png"):
            try:
                paths[int(path.stem.split("_")[-1])] = path
            except ValueError:
                continue
        return paths

    @property
    def category(self) -> str:
        return self.categories[self.category_control.selected]

    @property
    def truncation(self) -> float:
        return float(self.variety_slider.value)

    def _sync_class_availability(self) -> None:
        available = set(self.category_maps.get(self.category, self.classes))
        for index, (name, chip) in enumerate(zip(self.classes, self.class_chips, strict=True)):
            chip.enabled = name in available
            chip.selected = index == self.class_id
        if self.classes[self.class_id] not in available:
            self.class_id = next(
                index for index, name in enumerate(self.classes) if name in available
            )
            for index, chip in enumerate(self.class_chips):
                chip.selected = index == self.class_id
        self.poster_product_name = self.classes[self.class_id]

    def _price_copy(self) -> tuple[str, int]:
        """Obtiene el copy de un paso determinista del agente si está disponible."""
        algorithm = self.config.get("integration", {}).get("pricing_algorithm", "PPO")
        try:
            product = next(
                item for item in load_product_data(self.config) if item.name == self.category
            )
            model_path = resolve_path(self.config["paths"]["models"]) / f"{algorithm.lower()}.zip"
            if not model_path.is_file():
                return campaign_offer_copy(self.discount_slider.value)
            model = MODEL_TYPES[algorithm].load(str(model_path), device="cpu")
            env = make_environment(
                self.config,
                [product],
                forecast_units=load_forecast_units(self.config),
            )
            observation, _ = env.reset(
                seed=int(self.config["project"]["seed"]),
                options={"product": self.category, "segment": 1, "initial_price": 1.0},
            )
            action, _ = model.predict(observation, deterministic=True)
            direction = int(action) - 1
            price_index = float(
                np.clip(
                    1.0 + direction * float(self.config["pricing"]["price_step_fraction"]),
                    float(self.config["pricing"]["price_min_multiplier"]),
                    float(self.config["pricing"]["price_max_multiplier"]),
                )
            )
            return agent_offer_copy(
                int(action), product.reference_price, price_index * product.reference_price
            )
        except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as error:
            LOGGER.info("Se usa el descuento manual del póster: %s", error)
            return campaign_offer_copy(self.discount_slider.value)

    def layout_controls(self, content_rect: pygame.Rect) -> None:
        """Distribuye controles, preview y evolución usando columnas ponderadas."""
        panel_rect = pygame.Rect(
            content_rect.x + 20,
            content_rect.y + 90,
            content_rect.width - 40,
            max(0, content_rect.height - 114),
        )
        self.control_rect, self.preview_rect, self.gallery_rect = WeightedLayout(gap=14).columns(
            panel_rect,
            [Track(1, 292), Track(1.75, 450), Track(1.15, 300)],
        )
        x = self.control_rect.x + 14
        width = self.control_rect.width - 28
        self.category_control.rect = pygame.Rect(x, self.control_rect.y + 32, width, 32)
        chip_width = (width - 4 * 5) // 5
        for index, chip in enumerate(self.class_chips):
            row, column = divmod(index, 5)
            chip.rect = pygame.Rect(
                x + column * (chip_width + 5),
                self.control_rect.y + 82 + row * 34,
                chip_width,
                29,
            )
        self.seed_rect = pygame.Rect(x, self.control_rect.y + 154, width - 46, 30)
        self.dice_button.rect.topleft = (self.seed_rect.right + 8, self.seed_rect.y)
        slider_width = width - 52
        self.count_slider.rect = pygame.Rect(x, self.control_rect.y + 202, slider_width, 4)
        self.variety_slider.rect = pygame.Rect(x, self.control_rect.y + 238, slider_width, 4)
        self.discount_slider.rect = pygame.Rect(x, self.control_rect.y + 274, slider_width, 4)
        selector_gap = 6
        template_width = round((width - selector_gap) * 0.62)
        self.template_control.rect = pygame.Rect(x, self.control_rect.y + 306, template_width, 28)
        self.style_control.rect = pygame.Rect(
            x + template_width + selector_gap,
            self.control_rect.y + 306,
            width - template_width - selector_gap,
            28,
        )
        self.buttons["generate"].rect = pygame.Rect(
            self.preview_rect.x + 14, self.preview_rect.y + 68, 132, 34
        )
        self.buttons["random"].rect = pygame.Rect(
            self.preview_rect.x + 154, self.preview_rect.y + 68, 120, 34
        )
        self.buttons["poster"].rect = pygame.Rect(x, self.control_rect.bottom - 38, width, 34)
        self.buttons["download"].rect = pygame.Rect(
            self.preview_rect.right - 294, self.preview_rect.y + 10, 138, 32
        )
        self.buttons["large"].rect = pygame.Rect(
            self.preview_rect.right - 146, self.preview_rect.y + 10, 134, 32
        )
        right_x = self.gallery_rect.x + 12
        tab_width = max(64, (self.gallery_rect.width - 24 - 12) // 3)
        for index, button in enumerate(self.history_buttons):
            button.rect = pygame.Rect(
                right_x + index * (tab_width + 6), self.gallery_rect.y + 46, tab_width, 30
            )
        self.epoch_slider.rect = pygame.Rect(
            right_x + 8, self.gallery_rect.bottom - 42, self.gallery_rect.width - 58, 4
        )

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.modal_open:
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONUP):
                self.modal_open = False
            return
        if event.type == pygame.KEYDOWN and self.seed_focused:
            if event.key == pygame.K_RETURN:
                self._commit_seed()
            elif event.key == pygame.K_BACKSPACE:
                self.seed_text = self.seed_text[:-1]
            elif event.unicode.isdigit() and len(self.seed_text) < 8:
                self.seed_text += event.unicode
            return
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.seed_focused = self.seed_rect.collidepoint(event.pos)
            for index, chip in enumerate(self.class_chips):
                if chip.handle_event(event):
                    self.class_id = index
                    self.poster_product_name = self.classes[index]
                    self.poster_path = None
                    self.offer_copy, self.discount_percent = self._price_copy()
                    self.discount_slider.value = self.discount_percent
                    self.manual_discount_override = False
            if self.category_control.handle_event(event):
                self._sync_class_availability()
                self.poster_path = None
                self.offer_copy, self.discount_percent = self._price_copy()
                self.discount_slider.value = self.discount_percent
                self.manual_discount_override = False
            self.template_control.handle_event(event)
            self.style_control.handle_event(event)
            for button in self.history_buttons:
                button.handle_event(event)
        self.count_slider.handle_event(event)
        self.variety_slider.handle_event(event)
        if self.discount_slider.handle_event(event):
            self.manual_discount_override = True
            self.offer_copy, self.discount_percent = campaign_offer_copy(self.discount_slider.value)
            self.poster_path = None
        self.epoch_slider.handle_event(event)
        for button in self.buttons.values():
            button.handle_event(event)
        if self.epoch_values:
            self.epoch = min(
                self.epoch_values, key=lambda value: abs(value - round(self.epoch_slider.value))
            )
        self._handle_gallery_click(event)

    def _handle_gallery_click(self, event: pygame.event.Event) -> None:
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1 or self.images is None:
            return
        card_rects = self._generated_card_rects()
        for index, rect in enumerate(card_rects):
            if rect.collidepoint(event.pos):
                self.hero_index = index
                self.message = f"Héroe seleccionado: {self.poster_product_name} {index + 1}"
                return

    def _commit_seed(self) -> None:
        try:
            self.seed_value = max(0, int(self.seed_text or 0))
            self.seed_text = str(self.seed_value)
            self.seed_focused = False
        except ValueError:
            self.seed_text = str(self.seed_value)

    def random_seed(self) -> None:
        self.seed_value = random.randrange(10_000_000)
        self.seed_text = str(self.seed_value)
        self.seed_focused = False

    def _set_history_tab(self, index: int) -> None:
        self.history_tab = index

    def generate(self) -> None:
        if self.generator is None:
            self.message = self.model_error or "El generador todavía no está listo."
            return
        if self._future is not None:
            return
        self._commit_seed()
        self._submit(
            "generate",
            self.generator.generate,
            self.class_id,
            round(self.count_slider.value),
            self.seed_value,
            truncation=self.truncation,
        )

    def random_generate(self) -> None:
        self._sync_class_availability()
        options = [index for index, chip in enumerate(self.class_chips) if chip.enabled]
        self.class_id = random.choice(options)
        self.random_seed()
        self._sync_class_availability()
        self.generate()

    def _ranked_poster_images(self, class_id: int, seed: int, truncation: float) -> torch.Tensor:
        if self.generator is None:
            raise RuntimeError(self.model_error or "No hay generador cGAN cargado.")
        candidates = self.generator.generate(class_id, 64, seed, truncation=truncation)
        best = rank_generated_images(candidates, class_id, self.classifier)
        return candidates[best[:3]]

    def _create_poster_worker(
        self,
        class_id: int,
        seed: int,
        truncation: float,
        images: torch.Tensor | None,
        hero_index: int,
        product_name: str,
        template: str,
        style: str,
        discount: int,
        offer_copy: str,
        destination: Path,
    ) -> Path:
        ranked = self._ranked_poster_images(class_id, seed, truncation)
        hero = images[hero_index] if images is not None else ranked[0]
        return create_campaign_poster(
            ranked,
            product_name=product_name,
            class_id=class_id,
            destination=destination,
            hero_image=hero,
            template=template,
            style=style,
            discount_percent=discount,
            offer_copy=offer_copy,
            seed=seed,
        )

    def _submit(self, action: str, function, *args, followup: str | None = None, **kwargs) -> None:
        self._future_action = action
        self._followup_action = followup
        self.message = "Generando muestras…" if action == "generate" else "Componiendo póster…"
        labels = {
            "generate": "Generando…" if action == "generate" else "Generar",
            "random": "Aleatorio",
            "poster": "Componiendo…" if action == "poster" else "Crear póster",
            "download": "Descargar PNG",
            "large": "Ver en grande",
        }
        for key, label in labels.items():
            self.buttons[key].enabled = False
            self.buttons[key].label = label
        self._future = self._executor.submit(function, *args, **kwargs)

    def update(self) -> None:
        """Recoge resultados de inferencia sin bloquear el hilo de Pygame."""
        if self._future is None or not self._future.done():
            return
        future = self._future
        action = self._future_action
        followup = self._followup_action
        self._future = None
        self._future_action = None
        self._followup_action = None
        try:
            result = future.result()
            if action == "generate":
                self.images = result
                self._generated_surface_cache.clear()
                self.hero_index = 0
                self.poster_path = None
                self.message = f"{len(self.images)} muestras generadas · selecciona el héroe"
            else:
                self.poster_path = result
                self.message = f"Póster listo · {self.offer_copy}"
                self.toast = Toast("Póster creado · listo para descargar")
                if followup == "download":
                    self.download_poster()
                elif followup == "modal":
                    self.open_modal()
        except (OSError, ValueError, RuntimeError) as error:
            self.message = str(error)
            LOGGER.exception("Falló la tarea de campaña")
        finally:
            labels = {
                "generate": "Generar",
                "random": "Aleatorio",
                "poster": "Crear póster",
                "download": "Descargar PNG",
                "large": "Ver en grande",
            }
            for key, label in labels.items():
                self.buttons[key].enabled = True
                self.buttons[key].label = label

    def create_poster(self) -> None:
        if self.generator is None:
            self.message = self.model_error or "No hay generador cGAN cargado."
            return
        if self._future is not None:
            return
        if self.manual_discount_override:
            offer_copy, discount = campaign_offer_copy(self.discount_slider.value)
        else:
            offer_copy = self.offer_copy
            discount = self.discount_percent if self.offer_copy.startswith("-") else 0
        output_dir = resolve_path(self.config["paths"]["outputs"]) / "posters"
        filename = f"campaign_{self.category.lower()}_{self.class_id}_{self.seed_value}.png"
        path = output_dir / filename
        self._submit(
            "poster",
            self._create_poster_worker,
            self.class_id,
            self.seed_value,
            self.truncation,
            self.images.detach().cpu().clone() if self.images is not None else None,
            self.hero_index,
            self.poster_product_name,
            TEMPLATES[self.template_control.selected],
            self.style_control.value,
            discount,
            offer_copy,
            path,
        )

    def download_poster(self) -> None:
        if self.poster_path is None or not self.poster_path.is_file():
            self.create_poster()
            if self._future is not None:
                self._followup_action = "download"
            return
        if self.poster_path is not None and self.poster_path.is_file():
            output_dir = resolve_path(self.config["paths"]["outputs"]) / "posters"
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / f"urbanstyle_{self.seed_value}_{self.class_id}.png"
            if destination != self.poster_path:
                destination.write_bytes(self.poster_path.read_bytes())
            self.message = f"Descargado: {destination.name}"
            self.toast = Toast(f"PNG guardado: {destination.name}")

    def open_modal(self) -> None:
        if self.poster_path is None or not self.poster_path.is_file():
            self.create_poster()
            if self._future is not None:
                self._followup_action = "modal"
            return
        self.modal_open = True

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _draw_label(self, surface, text, position, font, color) -> None:
        surface.blit(font.render(text, True, color), position)

    def _scaled_image(self, path: Path, size: tuple[int, int]) -> pygame.Surface | None:
        if size[0] <= 0 or size[1] <= 0 or not path.is_file():
            return None
        try:
            modified = path.stat().st_mtime_ns
            key = (str(path), modified, size)
            if key not in self._image_surface_cache:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.thumbnail(size, Image.Resampling.LANCZOS)
                    self._image_surface_cache[key] = pygame.image.fromstring(
                        image.tobytes(), image.size, "RGB"
                    )
            return self._image_surface_cache[key]
        except (OSError, pygame.error):
            LOGGER.warning("No se pudo mostrar imagen %s", path)
            return None

    def _blit_image(self, surface: pygame.Surface, path: Path, rect: pygame.Rect) -> None:
        if rect.width <= 0 or rect.height <= 0 or not path.is_file():
            return
        bitmap = self._scaled_image(path, rect.size)
        if bitmap is not None:
            surface.blit(bitmap, bitmap.get_rect(center=rect.center))

    def _draw_control_panel(self, surface, fonts, theme: Theme) -> None:
        rect = self.control_rect
        x = rect.x + 14
        self._draw_label(surface, "CAMPAÑA", (x, rect.y + 14), fonts["label"], theme.sky)
        self.category_control.draw(surface, fonts["label"], theme)
        self._draw_label(surface, "PRODUCTO", (x, rect.y + 67), fonts["label"], theme.muted)
        for chip in self.class_chips:
            chip.draw(surface, fonts["label"], theme)

        pygame.draw.rect(surface, theme.panel, self.seed_rect, border_radius=8)
        pygame.draw.rect(surface, theme.border, self.seed_rect, 1, border_radius=8)
        seed_font = fonts["body"]
        seed_color = theme.text if self.seed_focused else theme.muted
        seed_value = f"SEED  {self.seed_text}{'|' if self.seed_focused else ''}"
        self._draw_label(
            surface,
            seed_value,
            (self.seed_rect.x + 12, self.seed_rect.y + 7),
            seed_font,
            seed_color,
        )
        self.dice_button.draw(surface, fonts["label"], theme)

        self._draw_label(surface, "CANTIDAD", (x, rect.y + 190), fonts["label"], theme.muted)
        self.count_slider.draw(surface, fonts["label"], theme)
        self._draw_label(surface, "VARIEDAD", (x, rect.y + 229), fonts["label"], theme.muted)
        self.variety_slider.draw(surface, fonts["label"], theme)
        self._draw_label(surface, "DESCUENTO", (x, rect.y + 268), fonts["label"], theme.muted)
        self.discount_slider.draw(surface, fonts["label"], theme)
        self.template_control.draw(surface, fonts["label"], theme)
        self.style_control.draw(surface, fonts["label"], theme)

    def _generated_card_rects(self) -> list[pygame.Rect]:
        if self.images is None:
            return []
        count = len(self.images)
        visible = count
        gap = 8
        left, right = self.preview_rect.x + 14, self.preview_rect.right - 14
        top = self.preview_rect.bottom - 70
        card_width = (right - left - gap * (visible - 1)) // visible
        return [
            pygame.Rect(left + index * (card_width + gap), top, card_width, 58)
            for index in range(visible)
        ]

    def _draw_generated_gallery(self, surface, fonts, theme: Theme) -> None:
        rects = self._generated_card_rects()
        if self.images is None:
            empty = pygame.Rect(
                self.preview_rect.x + 18,
                self.preview_rect.bottom - 120,
                self.preview_rect.width - 36,
                70,
            )
            draw_vector_icon(surface, "info", empty.midleft, theme.disabled, 24)
            self._draw_label(
                surface,
                "Genera prendas y elige el héroe del póster",
                (empty.x + 24, empty.y + 24),
                fonts["label"],
                theme.muted,
            )
            return
        for index, rect in enumerate(rects):
            selected = index == self.hero_index
            draw_panel(
                surface,
                rect,
                theme.deep if selected else theme.panel,
                radius=8,
                border_color=theme.primary if selected else theme.border,
            )
            image = self.images[index]
            side = min(44, rect.width - 4)
            key = (index, side)
            if key not in self._generated_surface_cache:
                grayscale = ((image[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).numpy()
                bitmap = pygame.image.fromstring(
                    np.repeat(grayscale[:, :, None], 3, axis=2).tobytes(), (28, 28), "RGB"
                )
                self._generated_surface_cache[key] = pygame.transform.smoothscale(
                    bitmap, (side, side)
                )
            bitmap = self._generated_surface_cache[key]
            surface.blit(bitmap, bitmap.get_rect(center=rect.center))
            if selected:
                pygame.draw.circle(surface, theme.success, (rect.right - 8, rect.y + 8), 5)

    def _draw_training_history(self, surface, fonts, theme: Theme) -> None:
        rect = self.gallery_rect
        self._draw_label(
            surface,
            "EVOLUCIÓN DEL ENTRENAMIENTO",
            (rect.x + 12, rect.y + 14),
            fonts["label"],
            theme.muted,
        )
        for index, button in enumerate(self.history_buttons):
            button.draw(surface, fonts["label"], theme)
            if index == self.history_tab:
                pygame.draw.line(
                    surface, theme.primary, button.rect.bottomleft, button.rect.bottomright, 2
                )
        content = pygame.Rect(rect.x + 12, rect.y + 88, rect.width - 24, rect.height - 148)
        if self.history_tab == 0:
            path = self.epoch_paths.get(self.epoch)
            if path:
                bitmap = self._scaled_image(path, content.size)
                if bitmap is not None:
                    surface.blit(bitmap, bitmap.get_rect(center=content.center))
            else:
                draw_panel(surface, content, theme.panel, radius=8, border_color=theme.border)
                self._draw_label(
                    surface,
                    "Sin muestras para esta época",
                    (content.x + 12, content.centery),
                    fonts["label"],
                    theme.muted,
                )
        elif self.history_tab == 1:
            self._draw_loss_curves(surface, content, fonts, theme)
        else:
            self._draw_quality(surface, content, fonts, theme)
        self.epoch_slider.draw(surface, fonts["label"], theme)
        self._draw_label(
            surface,
            f"Época {self.epoch} / {max(self.epoch_values, default=80)}",
            (rect.x + 12, rect.bottom - 30),
            fonts["label"],
            theme.muted,
        )

    def _draw_loss_curves(self, surface, rect, fonts, theme: Theme) -> None:
        try:
            import pandas as pd

            frame = pd.read_csv(self.curves_path)
            generator = frame["generator_loss"].astype(float).tolist()
            discriminator = frame["discriminator_loss"].astype(float).tolist()
        except (OSError, ValueError, KeyError, ImportError):
            self._draw_label(
                surface, "Curvas aún no disponibles", rect.topleft, fonts["label"], theme.muted
            )
            return
        maximum = max(1.0, *generator, *discriminator)
        for index, (values, color) in enumerate(
            ((generator, theme.primary), (discriminator, theme.sky))
        ):
            points = [
                (
                    rect.x + round(i * rect.width / max(1, len(values) - 1)),
                    rect.bottom - round(value / maximum * rect.height),
                )
                for i, value in enumerate(values)
            ]
            if len(points) > 1:
                pygame.draw.lines(surface, color, False, points, 2)
            self._draw_label(
                surface,
                "Generador" if index == 0 else "Discriminador",
                (rect.x + 8, rect.y + 8 + 20 * index),
                fonts["label"],
                color,
            )

    def _draw_quality(self, surface, rect, fonts, theme) -> None:
        epochs = self.quality_history.get("epochs", [])
        if not epochs:
            self._draw_label(
                surface,
                "Métricas de calidad no disponibles",
                rect.topleft,
                fonts["label"],
                theme.muted,
            )
            return
        selected = min(epochs, key=lambda row: abs(int(row["epoch"]) - self.epoch))
        accuracy = float(selected.get("conditional_accuracy", 0.0))
        confidence = float(selected.get("mean_confidence", 0.0))
        self._draw_label(
            surface,
            f"Accuracy condicional · época {selected['epoch']}",
            (rect.x + 8, rect.y + 14),
            fonts["label"],
            theme.muted,
        )
        value = fonts["subtitle"].render(f"{accuracy:.1%}", True, theme.text)
        surface.blit(value, (rect.x + 8, rect.y + 48))
        bar = pygame.Rect(rect.x + 8, rect.y + 94, rect.width - 16, 12)
        draw_panel(surface, bar, theme.panel, radius=6, border_color=theme.border)
        fill = bar.copy()
        fill.width = round(bar.width * min(1, max(0, accuracy)))
        draw_panel(surface, fill, theme.primary, radius=6)
        self._draw_label(
            surface,
            f"Confianza media {confidence:.1%}",
            (rect.x + 8, rect.y + 122),
            fonts["label"],
            theme.sky,
        )
        self._draw_label(
            surface,
            "Mide reconocimiento de clase; no calidad perceptual.",
            (rect.x + 8, rect.y + 150),
            fonts["label"],
            theme.muted,
        )

    def _draw_preview(self, surface, fonts, theme) -> None:
        rect = self.preview_rect
        self._draw_label(
            surface, "VISTA PREVIA", (rect.x + 16, rect.y + 14), fonts["label"], theme.muted
        )
        self._draw_label(
            surface,
            f"{self.category} · {self.poster_product_name}",
            (rect.x + 16, rect.y + 36),
            fonts["subtitle"],
            theme.text,
        )
        poster_area = pygame.Rect(rect.x + 18, rect.y + 106, rect.width - 36, rect.height - 190)
        if self.poster_path and self.poster_path.is_file():
            self._blit_image(surface, self.poster_path, poster_area)
        else:
            draw_panel(surface, poster_area, theme.deep, radius=12, border_color=theme.border)
            self._draw_label(
                surface,
                self.offer_copy,
                (poster_area.x + 20, poster_area.y + 22),
                fonts["title"],
                theme.text,
            )
            self._draw_label(
                surface,
                self.poster_product_name.upper(),
                (poster_area.x + 20, poster_area.y + 64),
                fonts["body"],
                theme.sky,
            )
            self._draw_label(
                surface,
                "Tu campaña aparecerá aquí",
                (poster_area.x + 20, poster_area.bottom - 34),
                fonts["label"],
                theme.muted,
            )
        if self.images is not None:
            self._draw_generated_gallery(surface, fonts, theme)
        else:
            empty = pygame.Rect(rect.x + 18, rect.bottom - 96, rect.width - 36, 70)
            draw_panel(surface, empty, theme.panel, radius=8, border_color=theme.border)
            draw_vector_icon(surface, "info", (empty.x + 28, empty.centery), theme.sky, 24)
            self._draw_label(
                surface,
                "Genera muestras para seleccionar el héroe",
                (empty.x + 52, empty.y + 26),
                fonts["label"],
                theme.muted,
            )
        for key in ("generate", "random", "poster", "download", "large"):
            self.buttons[key].draw(surface, fonts["label"], theme)

    def draw(self, surface: pygame.Surface, content_rect: pygame.Rect, fonts, theme: Theme) -> None:
        self.layout_controls(content_rect)
        self._sync_class_availability()
        for rect in (self.control_rect, self.preview_rect, self.gallery_rect):
            draw_panel(surface, rect, theme.elevated, radius=10, border_color=theme.border)
        self._draw_label(
            surface,
            "CAMPAÑAS GENERATIVAS",
            (content_rect.x + 28, content_rect.y + 24),
            fonts["label"],
            theme.sky,
        )
        self._draw_label(
            surface,
            "Diseña una nueva colección",
            (content_rect.x + 28, content_rect.y + 46),
            fonts["title"],
            theme.text,
        )
        self._draw_control_panel(surface, fonts, theme)
        self._draw_preview(surface, fonts, theme)
        self._draw_training_history(surface, fonts, theme)
        if self.toast:
            self.toast.draw(surface, fonts["label"], theme)
        if self.modal_open and self.poster_path:
            self._draw_modal(surface, fonts, theme)

    def _draw_modal(self, surface, fonts, theme) -> None:
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 205))
        surface.blit(overlay, (0, 0))
        modal = pygame.Rect(0, 0, min(surface.get_width() - 80, 680), surface.get_height() - 80)
        modal.center = surface.get_rect().center
        draw_panel(
            surface, modal, theme.elevated, radius=14, shadow=True, border_color=theme.border
        )
        self._blit_image(surface, self.poster_path, modal.inflate(-32, -52))
        self._draw_label(
            surface,
            "Clic o Esc para cerrar",
            (modal.x + 18, modal.bottom - 24),
            fonts["label"],
            theme.muted,
        )
