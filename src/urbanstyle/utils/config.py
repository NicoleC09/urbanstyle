"""Carga de configuración YAML y resolución de rutas del proyecto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Carga el YAML indicado o el config.yaml de la raíz."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"La configuración debe ser un objeto YAML: {path}")
    return config


def resolve_path(path: str | Path) -> Path:
    """Convierte una ruta relativa a una ruta absoluta desde la raíz del proyecto."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
