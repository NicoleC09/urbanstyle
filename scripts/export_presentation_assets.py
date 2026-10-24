"""Genera y copia figuras verificadas para la presentación de UrbanStyle."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.urbanstyle.utils.config import load_config, resolve_path

COLORS = {
    "background": "#060A14",
    "panel": "#0C1424",
    "elevated": "#121D33",
    "border": "#1F2E4D",
    "primary": "#2F6BFF",
    "sky": "#7CC4FF",
    "text": "#F4F7FC",
    "muted": "#8FA0BF",
    "success": "#4ADE9B",
}


def _style_axis(axis) -> None:
    axis.set_facecolor(COLORS["panel"])
    axis.tick_params(colors=COLORS["text"])
    axis.xaxis.label.set_color(COLORS["muted"])
    axis.yaxis.label.set_color(COLORS["muted"])
    axis.title.set_color(COLORS["text"])
    for spine in axis.spines.values():
        spine.set_color(COLORS["border"])
    axis.grid(axis="y", color=COLORS["border"], alpha=0.7)
    axis.set_axisbelow(True)


def _save(figure, path: Path) -> Path:
    figure.savefig(path, dpi=180, facecolor=COLORS["background"], bbox_inches="tight")
    plt.close(figure)
    return path


def _problem_asset(path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(11, 5))
    figure.patch.set_facecolor(COLORS["background"])
    axis.set_facecolor(COLORS["background"])
    axis.axis("off")
    columns = (
        ("OBSERVAR", "Ventas semanales", COLORS["sky"]),
        ("DECIDIR", "Precio y margen", COLORS["primary"]),
        ("EXPLORAR", "Rutas y tráfico", COLORS["success"]),
        ("COMUNICAR", "Campaña visual", COLORS["text"]),
    )
    for index, (eyebrow, label, color) in enumerate(columns):
        x = 0.04 + index * 0.245
        axis.text(
            x, 0.69, eyebrow, color=color, fontsize=12, weight="bold", transform=axis.transAxes
        )
        axis.text(
            x,
            0.48,
            label,
            color=COLORS["text"],
            fontsize=17,
            weight="bold",
            transform=axis.transAxes,
        )
        axis.text(
            x,
            0.29,
            ("B", "C", "A", "D")[index],
            color=color,
            fontsize=28,
            weight="bold",
            transform=axis.transAxes,
        )
    axis.text(
        0.04,
        0.08,
        "Laboratorio reproducible · entorno sintético · no es recomendación comercial",
        color=COLORS["muted"],
        fontsize=11,
        transform=axis.transAxes,
    )
    return _save(figure, path)


def _architecture_asset(path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(11, 5))
    figure.patch.set_facecolor(COLORS["background"])
    axis.set_facecolor(COLORS["background"])
    axis.axis("off")
    nodes = (
        (0.04, 0.52, "CSV + calibración", COLORS["muted"]),
        (0.26, 0.69, "B · LSTM\npronóstico", COLORS["sky"]),
        (0.48, 0.69, "C · PPO / DQN\nprecio y margen", COLORS["primary"]),
        (0.70, 0.69, "A · BFS / A*\ntráfico", COLORS["success"]),
        (0.70, 0.30, "D · cGAN\ncampaña proxy", COLORS["text"]),
        (0.91, 0.52, "Inicio\nresumen", COLORS["sky"]),
    )
    for x, y, label, color in nodes:
        axis.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color=COLORS["text"],
            fontsize=12,
            bbox={
                "boxstyle": "round,pad=0.7",
                "facecolor": COLORS["elevated"],
                "edgecolor": color,
                "linewidth": 1.5,
            },
            transform=axis.transAxes,
        )
    arrows = (
        ((0.12, 0.54), (0.21, 0.66)),
        ((0.36, 0.69), (0.39, 0.69)),
        ((0.58, 0.69), (0.62, 0.69)),
        ((0.83, 0.67), (0.87, 0.55)),
        ((0.83, 0.32), (0.88, 0.48)),
        ((0.59, 0.64), (0.71, 0.35)),
    )
    for start, end in arrows:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": COLORS["sky"], "lw": 2},
        )
    axis.text(
        0.48,
        0.15,
        "Flujo integrado: B → C → A → D",
        color=COLORS["muted"],
        ha="center",
        fontsize=12,
        transform=axis.transAxes,
    )
    return _save(figure, path)


def _data_asset(path: Path, config: dict) -> Path:
    raw = pd.read_csv(resolve_path(config["paths"]["raw_data"]))
    weekly = pd.read_csv(resolve_path(config["paths"]["processed_data"]) / "weekly_sales.csv")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    figure.patch.set_facecolor(COLORS["background"])
    axis.set_facecolor(COLORS["panel"])
    bars = axis.bar(
        ["Transacciones", "Semanas", "Series"],
        [
            len(raw),
            weekly["week"].nunique(),
            weekly[["Product Category", "Gender"]].drop_duplicates().shape[0],
        ],
        color=[COLORS["primary"], COLORS["sky"], COLORS["success"]],
    )
    axis.set_title("Base observada y preparada")
    axis.set_ylabel("Cantidad")
    _style_axis(axis)
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{int(bar.get_height())}",
            ha="center",
            color=COLORS["text"],
        )
    return _save(figure, path)


def _layout_asset(path: Path) -> Path:
    frame = pd.read_csv(PROJECT_ROOT / "outputs" / "layout_experiment.csv")
    congestion = frame.pivot(index="layout", columns="algorithm", values="max_congestion_mean")
    nodes = frame.pivot(index="layout", columns="algorithm", values="explored_nodes_mean")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    figure.patch.set_facecolor(COLORS["background"])
    congestion.plot(kind="bar", ax=axes[0], color=[COLORS["sky"], COLORS["primary"]])
    axes[0].set_title("Congestión máxima media")
    axes[0].set_ylabel("Visitas/celda")
    nodes.plot(kind="bar", ax=axes[1], color=[COLORS["sky"], COLORS["primary"]])
    axes[1].set_title("Nodos expandidos")
    axes[1].set_ylabel("Media por corrida")
    for axis in axes:
        _style_axis(axis)
        axis.tick_params(axis="x", rotation=0)
        axis.legend(frameon=False, labelcolor=COLORS["text"])
    return _save(figure, path)


def _forecast_asset(path: Path) -> Path:
    frame = pd.read_csv(PROJECT_ROOT / "outputs" / "forecast_metrics.csv")
    frame = frame[frame["context_weeks"] == 8]
    models = list(frame["model"].drop_duplicates())
    rows = []
    for series in frame["series"].drop_duplicates():
        values = frame[frame["series"] == series].set_index("model")
        rows.append(
            [
                series,
                *[
                    f"{values.loc[name, 'mae']:.2f} / {values.loc[name, 'rmse']:.2f}"
                    for name in models
                ],
            ]
        )
    averages = frame.groupby("model")[["mae", "rmse"]].mean()
    rows.append(
        [
            "Promedio",
            *[
                f"{averages.loc[name, 'mae']:.2f} / {averages.loc[name, 'rmse']:.2f}"
                for name in models
            ],
        ]
    )
    figure, axis = plt.subplots(figsize=(11, 4.8))
    figure.patch.set_facecolor(COLORS["background"])
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=["Serie", *models], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["border"])
        cell.set_facecolor(COLORS["elevated"] if row else COLORS["primary"])
        cell.get_text().set_color(COLORS["text"])
    axis.set_title("MAE / RMSE · contexto 8 semanas · unidades", color=COLORS["text"], pad=18)
    return _save(figure, path)


def _pricing_curve_asset(path: Path, config: dict) -> Path:
    pricing = config["pricing"]
    elasticity = float(pricing["elasticity_fallback"])
    cost = float(pricing["unit_cost_fraction"])
    low, high = float(pricing["price_min_multiplier"]), float(pricing["price_max_multiplier"])
    prices = np.linspace(low, high, 240)
    demand = prices**elasticity
    margin = (prices - cost) * demand
    optimum = elasticity / (elasticity + 1) * cost
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    figure.patch.set_facecolor(COLORS["background"])
    axis.plot(prices, demand / demand.max(), color=COLORS["sky"], label="Demanda (normalizada)")
    axis.plot(
        prices,
        margin / margin.max(),
        color=COLORS["primary"],
        linewidth=2.5,
        label="Margen (normalizado)",
    )
    axis.axvline(optimum, color=COLORS["success"], linestyle="--", label=f"Óptimo {optimum:.2f}×")
    axis.set(
        title="Curva isoelástica asumida",
        xlabel="Precio / referencia",
        ylabel="Escala normalizada",
    )
    _style_axis(axis)
    axis.legend(frameon=False, labelcolor=COLORS["text"])
    return _save(figure, path)


def _pricing_evaluation_asset(path: Path) -> Path:
    frame = pd.read_csv(PROJECT_ROOT / "outputs" / "pricing_evaluation.csv")
    colors = [
        COLORS["success"] if "agente" in name else COLORS["primary"] for name in frame["policy"]
    ]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    figure.patch.set_facecolor(COLORS["background"])
    axis.bar(
        frame["policy"],
        frame["mean_margin"],
        yerr=frame["std_margin"],
        capsize=5,
        color=colors,
        error_kw={"ecolor": COLORS["text"]},
    )
    axis.set(
        title="Margen medio ± desviación · 100 episodios", ylabel="Margen acumulado por episodio"
    )
    axis.tick_params(axis="x", rotation=18)
    _style_axis(axis)
    return _save(figure, path)


def _gan_evolution_asset(path: Path) -> Path:
    sources = (
        ("Época 5", PROJECT_ROOT / "outputs/gan/samples_epoch_05.png"),
        ("Época 20", PROJECT_ROOT / "outputs/gan/samples_epoch_20.png"),
        ("Época 60", PROJECT_ROOT / "outputs/gan/samples_epoch_60.png"),
        ("Época 80", PROJECT_ROOT / "outputs/gan/samples_epoch_80.png"),
        ("Mejor EMA", PROJECT_ROOT / "outputs/gan/samples_best_ema.png"),
    )
    width, height = 1400, 420
    canvas = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    item_width = width // len(sources)
    for index, (label, source) in enumerate(sources):
        if not source.is_file():
            raise FileNotFoundError(f"Falta snapshot GAN: {source}")
        with Image.open(source) as image_source:
            image = image_source.convert("RGB")
            image.thumbnail((item_width - 24, height - 70), Image.Resampling.LANCZOS)
        x = index * item_width + (item_width - image.width) // 2
        canvas.paste(image, (x, 48))
        draw.text((index * item_width + 12, 12), label, font=font, fill=COLORS["text"])
    canvas.save(path)
    return path


def _gan_losses_asset(path: Path) -> Path:
    source = PROJECT_ROOT / "outputs/gan/cgan/losses.csv"
    frame = pd.read_csv(source)
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    figure.patch.set_facecolor(COLORS["background"])
    axis.plot(frame["epoch"], frame["generator_loss"], color=COLORS["primary"], label="Generador")
    axis.plot(
        frame["epoch"], frame["discriminator_loss"], color=COLORS["sky"], label="Discriminador"
    )
    axis.set(title="cGAN · pérdidas de entrenamiento", xlabel="Época", ylabel="Pérdida")
    _style_axis(axis)
    axis.legend(frameon=False, labelcolor=COLORS["text"])
    return _save(figure, path)


def _results_asset(path: Path) -> Path:
    text = (
        ("52", "semanas completas"),
        ("3.352", "MAE LSTM · unidades"),
        ("4 / 6", "series vs media móvil"),
        ("31 877", "margen PPO · simulación"),
        ("98.75 %", "accuracy condicional cGAN"),
    )
    figure, axis = plt.subplots(figsize=(11, 4))
    figure.patch.set_facecolor(COLORS["background"])
    axis.axis("off")
    for index, (value, label) in enumerate(text):
        x = 0.03 + index * 0.195
        axis.text(
            x,
            0.62,
            value,
            color=(COLORS["sky"], COLORS["primary"], COLORS["success"])[index % 3],
            fontsize=22,
            weight="bold",
            transform=axis.transAxes,
        )
        axis.text(x, 0.35, label, color=COLORS["muted"], fontsize=10, transform=axis.transAxes)
    return _save(figure, path)


def _copy_if_present(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    shutil.copy2(source, destination)
    return True


def _optional_pptx(output_dir: Path, assets: list[Path]) -> Path | None:
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.util import Inches, Pt
    except ImportError:
        return None
    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)
    for asset in assets:
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor(6, 10, 20)
        title = slide.shapes.add_textbox(Inches(0.55), Inches(0.25), Inches(12.2), Inches(0.55))
        paragraph = title.text_frame.paragraphs[0]
        paragraph.text = asset.stem.replace("_", " ").title()
        paragraph.font.size = Pt(26)
        paragraph.font.bold = True
        paragraph.font.color.rgb = RGBColor(244, 247, 252)
        slide.shapes.add_picture(
            str(asset), Inches(0.55), Inches(1.0), width=Inches(12.2), height=Inches(6.0)
        )
    destination = output_dir / "UrbanStyle.pptx"
    presentation.save(destination)
    return destination


def export_assets() -> tuple[list[Path], list[Path]]:
    """Genera figuras sintéticas/reporte y copia capturas y curvas del proyecto."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    output_dir = PROJECT_ROOT / "outputs" / "presentacion"
    output_dir.mkdir(parents=True, exist_ok=True)
    generators = (
        ("01_problema.png", _problem_asset),
        ("02_arquitectura.png", _architecture_asset),
        ("03_datos.png", lambda path: _data_asset(path, config)),
        ("04_bfs_vs_astar.png", _layout_asset),
        ("05_forecast_metrics.png", _forecast_asset),
        ("06_pricing_curves.png", lambda path: _pricing_curve_asset(path, config)),
        ("07_pricing_evaluation.png", _pricing_evaluation_asset),
        ("08_gan_evolution.png", _gan_evolution_asset),
        ("09_gan_losses.png", _gan_losses_asset),
        ("10_resultados.png", _results_asset),
    )
    generated = [function(output_dir / name) for name, function in generators]
    copied = []
    extra_sources = (
        (PROJECT_ROOT / "docs/forecast_training_curve.png", "11_lstm_training_curve.png"),
        (PROJECT_ROOT / "outputs/pricing_evaluation.png", "12_pricing_evaluation_source.png"),
        (PROJECT_ROOT / "outputs/layout_experiment.png", "13_layout_experiment_source.png"),
        (PROJECT_ROOT / "outputs/gan/losses.png", "14_gan_losses_source.png"),
        (PROJECT_ROOT / "outputs/gan/samples_best_ema.png", "15_gan_best_ema_source.png"),
    )
    for source, name in extra_sources:
        destination = output_dir / name
        if _copy_if_present(source, destination):
            copied.append(destination)
    screenshots = (
        ("captura_inicio.png", "16_captura_inicio.png"),
        ("captura_task_a.png", "17_captura_task_a.png"),
        ("captura_task_b.png", "18_captura_task_b.png"),
        ("captura_task_c.png", "19_captura_task_c.png"),
        ("captura_task_d.png", "20_captura_task_d.png"),
    )
    for source_name, destination_name in screenshots:
        destination = output_dir / destination_name
        if _copy_if_present(PROJECT_ROOT / "docs" / source_name, destination):
            copied.append(destination)
    deck = _optional_pptx(output_dir, generated)
    if deck:
        copied.append(deck)
    return generated, copied


def main() -> int:
    try:
        generated, copied = export_assets()
    except (FileNotFoundError, OSError, KeyError, ValueError) as error:
        print(f"No se pudieron exportar los recursos: {error}", file=sys.stderr)
        return 1
    for path in generated:
        print(f"Generado: {path}")
    for path in copied:
        print(f"Copiado: {path}")
    if not any(path.suffix == ".pptx" for path in copied):
        print("python-pptx no está instalado; se omitió el PPTX opcional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
