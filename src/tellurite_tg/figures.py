"""Publication figures and explainability outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


RED = "#b51f2e"
GRAY = "#4d4d4d"
LIGHT = "#d9d9d9"
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _formula(label: str) -> str:
    return label.translate(SUBSCRIPT)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    evaluation: dict[str, Any],
    figures_dir: Path,
    results_dir: Path,
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    ax.hist(dataset["Tg_C"], bins=28, color=RED, edgecolor="white", linewidth=0.4)
    ax.axvline(dataset["Tg_C"].median(), color=GRAY, linestyle="--", linewidth=1)
    ax.set(xlabel="$T_g$ experimental (°C)", ylabel="Vidrios")
    ax.text(
        0.98,
        0.96,
        f"n = {len(dataset):,}\nmediana = {dataset['Tg_C'].median():.1f} °C",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    paths["tg_distribution"] = figures_dir / "figure_1_tg_distribution.png"
    _save(fig, paths["tg_distribution"])

    frequencies = pd.DataFrame(
        {
            "component": [feature for feature in feature_columns if feature != "Other"],
            "frequency": [
                int((dataset[feature] > 0).sum())
                for feature in feature_columns
                if feature != "Other"
            ],
        }
    ).nlargest(15, "frequency")
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.barh(
        frequencies["component"].map(_formula)[::-1],
        frequencies["frequency"][::-1],
        color=RED,
    )
    ax.set(xlabel="Número de vidrios", ylabel="")
    paths["component_frequency"] = figures_dir / "figure_2_component_frequency.png"
    _save(fig, paths["component_frequency"])

    predictions = evaluation["predictions"]
    group_predictions = predictions.loc[
        predictions["validation"] == "GroupKFold by publication"
    ]
    models = list(group_predictions["model"].drop_duplicates())
    fig, axes = plt.subplots(len(models), 1, figsize=(3.35, 5.4))
    if len(models) == 1:
        axes = [axes]
    for panel_index, (ax, model_name) in enumerate(zip(axes, models)):
        subset = group_predictions.loc[group_predictions["model"] == model_name]
        lower = min(subset["observed_Tg_C"].min(), subset["predicted_Tg_C"].min())
        upper = max(subset["observed_Tg_C"].max(), subset["predicted_Tg_C"].max())
        margin = max(10.0, (upper - lower) * 0.03)
        ax.scatter(
            subset["observed_Tg_C"],
            subset["predicted_Tg_C"],
            s=7,
            alpha=0.38,
            color=RED,
            linewidths=0,
        )
        ax.plot([lower, upper], [lower, upper], color=GRAY, linewidth=0.8)
        ax.set_xlim(lower - margin, upper + margin)
        ax.set_ylim(lower - margin, upper + margin)
        display_model = {
            "Ridge": "Ridge",
            "Random forest": "Bosque aleatorio",
            "XGBoost": "XGBoost",
        }[model_name]
        ax.set_title(display_model, pad=2)
        if panel_index == len(models) - 1:
            ax.set_xlabel("$T_g$ observado (°C)")
        ax.set_ylabel("$T_g$ predicho (°C)")
        ax.tick_params(labelsize=7)
    fig.subplots_adjust(hspace=0.32)
    paths["observed_predicted"] = figures_dir / "figure_3_observed_predicted.png"
    _save(fig, paths["observed_predicted"])

    summary = evaluation["summary"].copy()
    display_names = {
        "GroupKFold by publication": "Publicación\n(repeticiones)",
        "Purged GroupKFold by publication": "Publicación\n(purgada)",
        "Random 80/20 reference": "Aleatoria\n80/20",
    }
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.45))
    for axis, metric, uncertainty, label in (
        (axes[0], "MAE_mean_C", "MAE_std_C", "MAE (°C)"),
        (axes[1], "RMSE_mean_C", "RMSE_std_C", "RMSE (°C)"),
    ):
        validations = list(summary["validation"].drop_duplicates())
        width = 0.22
        positions = np.arange(len(validations))
        for index, model_name in enumerate(models):
            values = [
                summary.loc[
                    (summary["validation"] == validation)
                    & (summary["model"] == model_name),
                    metric,
                ].iloc[0]
                for validation in validations
            ]
            errors = [
                summary.loc[
                    (summary["validation"] == validation)
                    & (summary["model"] == model_name),
                    uncertainty,
                ].iloc[0]
                for validation in validations
            ]
            axis.bar(
                positions + (index - 1) * width,
                values,
                width,
                yerr=errors,
                capsize=2,
                label={
                    "Ridge": "Ridge",
                    "Random forest": "Bosque aleatorio",
                    "XGBoost": "XGBoost",
                }[model_name],
                color=[LIGHT, GRAY, RED][index],
            )
        axis.set_xticks(
            positions, [display_names[validation] for validation in validations]
        )
        axis.set_ylabel(label)
        axis.tick_params(axis="x", labelsize=6.0)
        axis.tick_params(axis="y", labelsize=7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.subplots_adjust(top=0.78, bottom=0.23, wspace=0.52)
    paths["model_errors"] = figures_dir / "figure_4_model_errors.png"
    _save(fig, paths["model_errors"])

    explained_model_name = evaluation["report"]["best_model_by_mean_rmse"]
    explained_model = evaluation["fitted_models"][explained_model_name]
    x = dataset[feature_columns].astype(float)
    explainer = shap.TreeExplainer(explained_model)
    shap_values = explainer.shap_values(x)
    importance = pd.DataFrame(
        {
            "model": explained_model_name,
            "component": feature_columns,
            "mean_abs_shap_C": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap_C", ascending=False)
    importance.to_csv(results_dir / "shap_importance.csv", index=False)
    top_features = list(importance.head(12)["component"])
    top_indices = [feature_columns.index(feature) for feature in top_features]
    order = top_indices[::-1]
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    rng = np.random.default_rng(42)
    scatter = None
    for row, feature_index in enumerate(order):
        values = shap_values[:, feature_index]
        chemistry = x.iloc[:, feature_index].to_numpy()
        scaled = (chemistry - chemistry.min()) / (
            chemistry.max() - chemistry.min() + 1e-12
        )
        jitter = rng.normal(0, 0.07, size=len(values))
        scatter = ax.scatter(
            values,
            row + jitter,
            c=scaled,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=5,
            alpha=0.45,
            linewidths=0,
        )
    ax.axvline(0, color=GRAY, linewidth=0.7)
    ax.set_yticks(
        range(len(order)), [_formula(feature_columns[index]) for index in order]
    )
    ax.set_xlabel("Valor SHAP (°C)")
    ax.set_ylabel("")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
        colorbar.set_ticks([0, 1], labels=["baja", "alta"])
        colorbar.set_label("Fracción del componente")
    paths["shap"] = figures_dir / "figure_5_shap.png"
    _save(fig, paths["shap"])

    replicate_summary_path = (
        results_dir.parent.parent
        / "data"
        / "processed"
        / "cross_publication_replicate_summary.csv"
    )
    if replicate_summary_path.exists():
        replicate_summary = pd.read_csv(replicate_summary_path)
        fig, ax = plt.subplots(figsize=(3.35, 2.15))
        median_range = replicate_summary["Tg_range_C"].median()
        if len(replicate_summary) <= 10:
            ordered = replicate_summary.sort_values("Tg_range_C")
            labels = [f"C{i + 1}" for i in range(len(ordered))]
            ax.bar(labels, ordered["Tg_range_C"], color=RED)
            ax.axhline(median_range, color=GRAY, linestyle="--", linewidth=1)
            ax.set(
                xlabel="Composición repetida",
                ylabel="Rango de $T_g$ (°C)",
            )
            annotation_x = 0.04
            annotation_alignment = "left"
        else:
            ax.hist(
                replicate_summary["Tg_range_C"],
                bins=np.arange(
                    0, replicate_summary["Tg_range_C"].max() + 10, 10
                ),
                color=RED,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.axvline(median_range, color=GRAY, linestyle="--", linewidth=1)
            ax.set(
                xlabel="Rango de $T_g$ entre publicaciones (°C)",
                ylabel="Composiciones repetidas",
            )
            annotation_x = 0.98
            annotation_alignment = "right"
        ax.text(
            annotation_x,
            0.95,
            f"n = {len(replicate_summary)}\nmediana = {median_range:.1f} °C",
            transform=ax.transAxes,
            ha=annotation_alignment,
            va="top",
        )
        paths["reproducibility"] = (
            figures_dir / "figure_6_reproducibility.png"
        )
        _save(fig, paths["reproducibility"])
    return paths
