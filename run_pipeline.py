"""Run the full, reproducible tellurite-glass Tg study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tellurite_tg.data import ensure_raw_data, load_external_dataset, prepare_dataset
from tellurite_tg.evaluation import evaluate_models, regression_metrics
from tellurite_tg.figures import create_figures
from tellurite_tg.models import build_models


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pipeline.json",
    )
    parser.add_argument(
        "--external",
        type=Path,
        help=(
            "CSV opcional de laboratorio independiente "
            "(columnas en mol%% y Tg_C opcional)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    results_dir = ROOT / "artifacts" / "results"
    figures_dir = ROOT / "artifacts" / "figures"
    models_dir = ROOT / "artifacts" / "models"

    props_path, comp_path = ensure_raw_data(raw_dir)
    dataset, feature_columns, audit = prepare_dataset(
        props_path,
        comp_path,
        processed_dir,
        tellurite_min_wt_percent=float(config["data"]["tellurite_min_wt_percent"]),
        apply_component_domain_filter=bool(
            config["data"]["apply_component_domain_filter"]
        ),
        zno_min_mol_percent=float(config["data"]["zno_min_mol_percent"]),
        zno_max_mol_percent=float(config["data"]["zno_max_mol_percent"]),
        teo2_min_mol_percent=float(config["data"]["teo2_min_mol_percent"]),
        teo2_max_mol_percent=float(config["data"]["teo2_max_mol_percent"]),
        la2o3_min_mol_percent=float(config["data"]["la2o3_min_mol_percent"]),
        la2o3_max_mol_percent=float(config["data"]["la2o3_max_mol_percent"]),
        retain_repeated_measurements=bool(
            config["data"]["retain_repeated_measurements"]
        ),
        composition_sum_tolerance=float(
            config["data"]["composition_sum_tolerance"]
        ),
        minimum_component_frequency=int(
            config["data"]["minimum_component_frequency"]
        ),
        fingerprint_decimals=int(config["data"]["fingerprint_decimals"]),
        target_min_C=float(config["data"]["target_min_C"]),
        target_max_C=float(config["data"]["target_max_C"]),
    )
    models = build_models(config)
    evaluation = evaluate_models(
        dataset,
        feature_columns,
        models,
        config,
        results_dir,
        models_dir,
    )
    create_figures(
        dataset,
        feature_columns,
        evaluation,
        figures_dir,
        results_dir,
    )

    external_message = "no solicitada"
    if args.external:
        external_x, external_y, sample_ids = load_external_dataset(
            args.external,
            feature_columns,
            zno_min_mol_percent=float(config["data"]["zno_min_mol_percent"]),
            zno_max_mol_percent=float(config["data"]["zno_max_mol_percent"]),
            teo2_min_mol_percent=float(config["data"]["teo2_min_mol_percent"]),
            teo2_max_mol_percent=float(config["data"]["teo2_max_mol_percent"]),
            la2o3_min_mol_percent=float(
                config["data"]["la2o3_min_mol_percent"]
            ),
            la2o3_max_mol_percent=float(
                config["data"]["la2o3_max_mol_percent"]
            ),
            enforce_component_domain=bool(
                config["data"]["apply_component_domain_filter"]
            ),
        )
        predictions = pd.DataFrame({"sample_id": sample_ids})
        for name, fitted in evaluation["fitted_models"].items():
            predictions[f"{name}_predicted_Tg_C"] = fitted.predict(external_x)
        external_report: dict[str, object] = {
            "source_file": str(args.external),
            "rows": int(len(external_x)),
        }
        if external_y is not None and external_y.notna().all():
            external_report["metrics"] = {
                name: regression_metrics(
                    external_y, predictions[f"{name}_predicted_Tg_C"]
                )
                for name in evaluation["fitted_models"]
            }
            predictions.insert(1, "observed_Tg_C", external_y)
        predictions.to_csv(results_dir / "external_predictions.csv", index=False)
        (results_dir / "external_report.json").write_text(
            json.dumps(external_report, indent=2), encoding="utf-8"
        )
        external_message = f"{len(external_x)} filas procesadas"

    report = evaluation["report"]
    print(
        "Completado: "
        f"{audit['counts']['final_rows']} mediciones, "
        f"{audit['counts']['reference_groups']} publicaciones, "
        f"{len(feature_columns)} variables. "
        f"Mejor modelo en validación agrupada: "
        f"{report['best_model_by_mean_rmse']} "
        f"(RMSE {report['best_model_metrics']['RMSE_mean_C']:.2f} °C). "
        f"Validación externa: {external_message}."
    )


if __name__ == "__main__":
    main()
