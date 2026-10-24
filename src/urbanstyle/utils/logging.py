"""Configuración común del logging de UrbanStyle."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Activa mensajes legibles y consistentes en consola."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
