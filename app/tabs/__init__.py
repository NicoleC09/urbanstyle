"""Vistas iniciales de las cinco pestañas de UrbanStyle."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TabInfo:
    """Contenido textual mínimo de una pestaña aún no implementada."""

    label: str
    title: str
    description: str


TABS = (
    TabInfo(
        "Inicio",
        "La tienda, en una sola vista",
        "Conecta datos, decisiones y experiencia de compra.",
    ),
    TabInfo(
        "A · Tienda",
        "Recorridos y distribución",
        "Explora rutas de clientes y zonas de mayor tráfico.",
    ),
    TabInfo(
        "B · Demanda", "Una semana por delante", "Compara el pronóstico con referencias simples."
    ),
    TabInfo("C · Precios", "Decisiones de precio", "Observa cómo cambian demanda e ingresos."),
    TabInfo(
        "D · Campañas",
        "Ideas para la próxima campaña",
        "Genera conceptos visuales para cada categoría.",
    ),
)
