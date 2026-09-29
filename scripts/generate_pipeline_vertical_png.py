"""Genera la versión vertical del pipeline de análisis de Tg."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from generate_pipeline_png import (
    BLUE,
    DARK_BLUE,
    GRAY,
    LIGHT_BLUE,
    STAGES,
    WHITE,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "artifacts"
    / "figures"
    / "pipeline_analisis_base_datos_vertical.png"
)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 13), dpi=200)
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(
        50,
        96,
        "PIPELINE DE ANÁLISIS\nDE LA BASE DE DATOS",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        linespacing=1.15,
        color=DARK_BLUE,
    )

    box_x = 12.0
    box_width = 76.0
    box_height = 13.0
    y_positions = [77.5, 61.5, 45.5, 29.5, 13.5]

    for index, ((number, title, lines), box_y) in enumerate(
        zip(STAGES, y_positions)
    ):
        box = FancyBboxPatch(
            (box_x, box_y),
            box_width,
            box_height,
            boxstyle="round,pad=0.42,rounding_size=1.4",
            linewidth=2.2,
            edgecolor=BLUE,
            facecolor=LIGHT_BLUE if index < 4 else "#DCEEF7",
            zorder=2,
        )
        ax.add_patch(box)

        circle_x = box_x + 6.2
        circle_y = box_y + box_height / 2
        circle = plt.Circle(
            (circle_x, circle_y),
            2.65,
            facecolor=BLUE,
            edgecolor=BLUE,
            zorder=3,
        )
        ax.add_patch(circle)
        ax.text(
            circle_x,
            circle_y,
            number,
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color=WHITE,
            zorder=4,
        )

        ax.text(
            box_x + box_width / 2,
            box_y + 9.2,
            title,
            ha="center",
            va="center",
            fontsize=13.0,
            fontweight="bold",
            linespacing=1.05,
            color=DARK_BLUE,
            zorder=4,
        )
        ax.text(
            box_x + box_width / 2,
            box_y + 4.1,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=11.0,
            linespacing=1.35,
            color=GRAY,
            zorder=4,
        )

        if index < len(STAGES) - 1:
            next_y = y_positions[index + 1]
            arrow = FancyArrowPatch(
                (50, box_y - 0.35),
                (50, next_y + box_height + 0.35),
                arrowstyle="-|>",
                mutation_scale=21,
                linewidth=2.1,
                color=GRAY,
                zorder=1,
            )
            ax.add_patch(arrow)

    ax.text(
        50,
        6.5,
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
        pad_inches=0.22,
        facecolor=WHITE,
    )
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
