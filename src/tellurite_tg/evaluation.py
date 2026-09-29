"""Leakage-aware validation and model persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split


def regression_metrics(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(observed, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    return {
        "MAE_C": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_C": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    models: dict[str, Any],
    config: dict[str, Any],
    results_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    """Run GroupKFold primary validation and repeated random holdouts."""

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    x = dataset[feature_columns].astype(float)
    y = dataset["Tg_C"].astype(float)
    groups = dataset["reference_group"].astype(str)
    fingerprints = dataset["composition_fingerprint"].astype(str)
    row_id = np.arange(len(dataset))

    requested_folds = int(config["validation"]["group_folds"])
    group_folds = min(requested_folds, int(groups.nunique()))
    if group_folds < 2:
        raise RuntimeError("At least two independent publication groups are required.")

    split_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    group_cv = GroupKFold(n_splits=group_folds)
    for fold, (train_index, test_index) in enumerate(
        group_cv.split(x, y, groups), start=1
    ):
        train_groups = set(groups.iloc[train_index])
        test_groups = set(groups.iloc[test_index])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise AssertionError(f"Group leakage detected: {sorted(overlap)[:3]}")
        test_fingerprints = set(fingerprints.iloc[test_index])
        train_fingerprints = set(fingerprints.iloc[train_index])
        composition_overlap = train_fingerprints.intersection(test_fingerprints)
        test_seen_count = int(
            fingerprints.iloc[test_index].isin(composition_overlap).sum()
        )

        protocols = [
            (
                "GroupKFold by publication",
                train_index,
                len(composition_overlap),
                test_seen_count,
                composition_overlap,
            ),
            (
                "Purged GroupKFold by publication",
                np.asarray(
                    [
                        index
                        for index in train_index
                        if fingerprints.iloc[index] not in test_fingerprints
                    ],
                    dtype=int,
                ),
                0,
                0,
                set(),
            ),
        ]
        for (
            validation_name,
            protocol_train_index,
            fingerprint_overlap_count,
            seen_test_rows,
            seen_fingerprint_set,
        ) in protocols:
            for model_name, estimator in models.items():
                fitted = clone(estimator).fit(
                    x.iloc[protocol_train_index], y.iloc[protocol_train_index]
                )
                predicted = fitted.predict(x.iloc[test_index])
                split_rows.append(
                    {
                        "validation": validation_name,
                        "split": fold,
                        "model": model_name,
                        "n_train": int(len(protocol_train_index)),
                        "n_test": int(len(test_index)),
                        "publication_overlap": 0,
                        "composition_overlap": int(fingerprint_overlap_count),
                        "test_rows_with_seen_composition": int(seen_test_rows),
                        **regression_metrics(y.iloc[test_index], predicted),
                    }
                )
                prediction_rows.extend(
                    {
                        "validation": validation_name,
                        "split": fold,
                        "model": model_name,
                        "row_id": int(source_index),
                        "reference_group": groups.iloc[source_index],
                        "composition_fingerprint": fingerprints.iloc[source_index],
                        "seen_composition_in_training": (
                            fingerprints.iloc[source_index]
                            in seen_fingerprint_set
                        ),
                        "observed_Tg_C": float(y.iloc[source_index]),
                        "predicted_Tg_C": float(prediction),
                    }
                    for source_index, prediction in zip(test_index, predicted)
                )

    test_size = float(config["validation"]["random_test_size"])
    for seed in config["validation"]["random_split_seeds"]:
        train_index, test_index = train_test_split(
            row_id, test_size=test_size, random_state=int(seed)
        )
        publication_overlap = len(
            set(groups.iloc[train_index]).intersection(groups.iloc[test_index])
        )
        train_fingerprints = set(fingerprints.iloc[train_index])
        test_fingerprints = set(fingerprints.iloc[test_index])
        composition_overlap = train_fingerprints.intersection(test_fingerprints)
        test_seen_count = int(
            fingerprints.iloc[test_index].isin(composition_overlap).sum()
        )
        for model_name, estimator in models.items():
            fitted = clone(estimator).fit(x.iloc[train_index], y.iloc[train_index])
            predicted = fitted.predict(x.iloc[test_index])
            split_rows.append(
                {
                    "validation": "Random 80/20 reference",
                    "split": int(seed),
                    "model": model_name,
                    "n_train": int(len(train_index)),
                    "n_test": int(len(test_index)),
                    "publication_overlap": int(publication_overlap),
                    "composition_overlap": int(len(composition_overlap)),
                    "test_rows_with_seen_composition": test_seen_count,
                    **regression_metrics(y.iloc[test_index], predicted),
                }
            )
            prediction_rows.extend(
                {
                    "validation": "Random 80/20 reference",
                    "split": int(seed),
                    "model": model_name,
                    "row_id": int(source_index),
                    "reference_group": groups.iloc[source_index],
                    "composition_fingerprint": fingerprints.iloc[source_index],
                    "seen_composition_in_training": (
                        fingerprints.iloc[source_index] in composition_overlap
                    ),
                    "observed_Tg_C": float(y.iloc[source_index]),
                    "predicted_Tg_C": float(prediction),
                }
                for source_index, prediction in zip(test_index, predicted)
            )

    split_metrics = pd.DataFrame(split_rows)
    predictions = pd.DataFrame(prediction_rows)
    repetition_rows: list[dict[str, Any]] = []
    for (validation_name, model_name, seen), subset in predictions.groupby(
        ["validation", "model", "seen_composition_in_training"]
    ):
        repetition_rows.append(
            {
                "validation": validation_name,
                "model": model_name,
                "composition_seen_in_training": bool(seen),
                "n": int(len(subset)),
                **regression_metrics(
                    subset["observed_Tg_C"], subset["predicted_Tg_C"]
                ),
            }
        )
    repetition_metrics = pd.DataFrame(repetition_rows)
    summary = (
        split_metrics.groupby(["validation", "model"], as_index=False)
        .agg(
            n_splits=("split", "nunique"),
            MAE_mean_C=("MAE_C", "mean"),
            MAE_std_C=("MAE_C", "std"),
            RMSE_mean_C=("RMSE_C", "mean"),
            RMSE_std_C=("RMSE_C", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            publication_overlap_mean=("publication_overlap", "mean"),
            composition_overlap_mean=("composition_overlap", "mean"),
            test_rows_with_seen_composition_mean=(
                "test_rows_with_seen_composition",
                "mean",
            ),
            n_train_mean=("n_train", "mean"),
        )
        .sort_values(["validation", "RMSE_mean_C"])
    )
    split_metrics.to_csv(results_dir / "split_metrics.csv", index=False)
    predictions.to_csv(results_dir / "predictions.csv", index=False)
    repetition_metrics.to_csv(
        results_dir / "repetition_stratified_metrics.csv", index=False
    )
    summary.to_csv(results_dir / "metrics_summary.csv", index=False)

    fitted_models: dict[str, Any] = {}
    for model_name, estimator in models.items():
        fitted = clone(estimator).fit(x, y)
        fitted_models[model_name] = fitted
        joblib.dump(
            {
                "model": fitted,
                "feature_columns": feature_columns,
                "target_unit": "degrees Celsius",
            },
            models_dir / f"{model_name.lower().replace(' ', '_')}.joblib",
        )

    primary = summary.loc[
        summary["validation"] == "GroupKFold by publication"
    ].sort_values("RMSE_mean_C")
    best_model = str(primary.iloc[0]["model"])
    report = {
        "primary_validation": "GroupKFold by publication",
        "strict_validation": "Purged GroupKFold by publication",
        "group_folds": group_folds,
        "best_model_by_mean_rmse": best_model,
        "best_model_metrics": {
            key: float(primary.iloc[0][key])
            for key in (
                "MAE_mean_C",
                "MAE_std_C",
                "RMSE_mean_C",
                "RMSE_std_C",
                "R2_mean",
                "R2_std",
            )
        },
        "dataset_rows": int(len(dataset)),
        "publication_groups": int(groups.nunique()),
        "features": int(len(feature_columns)),
    }
    (results_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return {
        "split_metrics": split_metrics,
        "predictions": predictions,
        "repetition_metrics": repetition_metrics,
        "summary": summary,
        "fitted_models": fitted_models,
        "report": report,
    }
