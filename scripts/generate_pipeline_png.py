"""Genera un pipeline horizontal y legible para el estudio de Tg."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "figures" / "pipeline_analisis_base_datos.png"

BLUE = "#00629B"
DARK_BLUE = "#004A74"
GRAY = "#58595B"
LIGHT_BLUE = "#EAF4F9"
WHITE = "#FFFFFF"

STAGES = [
    (
        "1",
        "FUENTE DE DATOS",
        ["SciGlass / GlassPy", "Composición química", "Tg experimental"],
    ),
    (
        "2",
        "PREPARACIÓN",
        ["Integración de datos", "Aplicación de filtros", "Repeticiones conservadas"],
    ),
    (
        "3",
        "ENTRENAMIENTO",
        ["Definición de variables", "Ridge y bosque aleatorio", "XGBoost"],
    ),
    (
        "4",
        "VALIDACIÓN",
        ["Cinco pliegues", "Separación por publicación", "Validación purgada"],
    ),
    (
        "5",
        "ANÁLISIS DE\nRESULTADOS",
        ["Métricas y predicciones", "Tablas y gráficos", "Comparación de modelos"],
    ),
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 4.5), dpi=200)
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 28)
    ax.axis("off")

    ax.text(
        50,
        25.5,
        "PIPELINE DE ANÁLISIS DE LA BASE DE DATOS",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=DARK_BLUE,
    )

    box_width = 17.0
    box_height = 14.0
    box_y = 6.0
    x_positions = [1.0, 21.25, 41.5, 61.75, 82.0]

    for index, ((number, title, lines), box_x) in enumerate(
        zip(STAGES, x_positions)
    ):
        box = FancyBboxPatch(
            (box_x, box_y),
            box_width,
            box_height,
            boxstyle="round,pad=0.42,rounding_size=1.0",
            linewidth=2.0,
            edgecolor=BLUE,
            facecolor=LIGHT_BLUE if index < 4 else "#DCEEF7",
            zorder=2,
        )
        ax.add_patch(box)

        circle = plt.Circle(
            (box_x + 2.1, box_y + box_height - 2.2),
            1.25,
            facecolor=BLUE,
            edgecolor=BLUE,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            box_x + 2.1,
            box_y + box_height - 2.2,
            number,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=WHITE,
            zorder=4,
        )

        ax.text(
            box_x + box_width / 2,
            box_y + box_height - 5.0,
            title,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            linespacing=1.05,
            color=DARK_BLUE,
            zorder=4,
        )
        ax.text(
            box_x + box_width / 2,
            box_y + 4.5,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=10.5,
            linespacing=1.55,
            color=GRAY,
            zorder=4,
        )

        if index < len(STAGES) - 1:
            next_x = x_positions[index + 1]
            arrow = FancyArrowPatch(
                (box_x + box_width + 0.45, box_y + box_height / 2),
                (next_x - 0.45, box_y + box_height / 2),
                arrowstyle="-|>",
                mutation_scale=19,
                linewidth=2.0,
                color=GRAY,
                zorder=1,
            )
            ax.add_patch(arrow)

    ax.text(
        50,
        2.1,
        "Flujo principal del estudio; no incluye modelos especializados por familia.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=GRAY,
    )

    fig.savefig(
        OUTPUT,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor=WHITE,
    )
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
