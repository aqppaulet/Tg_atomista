"""Evaluate the present study under Riley-comparable and strict protocols.

Riley and Vienna (PNNL-26760) reported an apparent fit and analytical PRESS
for a linear mixture model.  For a nonlinear tree ensemble, the closest
computationally practical row-exclusion analogue is an out-of-bag (OOB)
prediction: every row is predicted only by trees whose bootstrap sample did
not contain that row.  OOB is not called PRESS because each tree also omits
other rows.

The Extra Trees configuration below is fixed to scikit-learn's default
``max_features=1.0`` and ``min_samples_leaf=1``.  The number of trees is raised
to 700 only to stabilize OOB predictions.  No OOB hyperparameter search is
performed in this confirmatory comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from compare_riley_protocol import (
    fit_no_intercept,
    load_mass_fraction_design,
    press_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "artifacts" / "results"


def regression_metrics(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    return {
        "MAE_C": float(mean_absolute_error(observed, predicted)),
        "RMSE_C": float(np.sqrt(mean_squared_error(observed, predicted))),
        "R2": float(r2_score(observed, predicted)),
    }


def build_model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=700,
        max_features=1.0,
        min_samples_leaf=1,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(
        ROOT / "data" / "processed" / "tellurite_tg_dataset.csv",
        low_memory=False,
    )
    audit = json.loads(
        (ROOT / "data" / "processed" / "dataset_audit.json").read_text(
            encoding="utf-8"
        )
    )
    feature_columns = list(audit["feature_columns"])
    x = dataset[feature_columns].to_numpy(dtype=float)
    y = dataset["Tg_C"].to_numpy(dtype=float)
    groups = dataset["reference_group"].astype(str).to_numpy()
    fingerprints = dataset["composition_fingerprint"].astype(str).to_numpy()

    rows: list[dict[str, object]] = [
        {
            "study": "Riley and Vienna (2017)",
            "protocol": "Apparent fit",
            "model": "Linear partial-specific coefficients",
            "n": 1939,
            "MAE_C": np.nan,
            "RMSE_C": 22.83494,
            "R2": 0.840634,
            "R2_std": np.nan,
            "RMSE_std_C": np.nan,
            "note": "Same rows used to fit and score the model.",
        },
        {
            "study": "Riley and Vienna (2017)",
            "protocol": "PRESS row exclusion",
            "model": "Linear partial-specific coefficients",
            "n": 1939,
            "MAE_C": np.nan,
            "RMSE_C": 23.3599096,
            "R2": 0.828806,
            "R2_std": np.nan,
            "RMSE_std_C": np.nan,
            "note": (
                "One row excluded; publications and repeated compositions "
                "were not grouped."
            ),
        },
    ]

    riley_design, _, _ = load_mass_fraction_design(dataset)
    riley_x = riley_design.to_numpy(dtype=float)
    riley_form_model = fit_no_intercept(riley_x, y)
    riley_form_apparent = riley_form_model.predict(riley_x)
    riley_form_press, _ = press_predictions(
        riley_x,
        y,
        riley_form_model,
    )
    for protocol, prediction, note in (
        (
            "Apparent fit, Riley linear form",
            riley_form_apparent,
            (
                "Same mass-fraction linear form as Riley, refitted on all "
                "current rows."
            ),
        ),
        (
            "Analytical PRESS, Riley linear form",
            riley_form_press,
            (
                "Exact analytical row-exclusion PRESS for the refitted "
                "linear mixture model."
            ),
        ),
    ):
        rows.append(
            {
                "study": "Present study",
                "protocol": protocol,
                "model": "Refitted Riley-form linear mixture model",
                "n": int(len(y)),
                **regression_metrics(y, prediction),
                "R2_std": np.nan,
                "RMSE_std_C": np.nan,
                "note": note,
            }
        )

    model = build_model().fit(x, y)
    apparent_prediction = model.predict(x)
    oob_prediction = np.asarray(model.oob_prediction_, dtype=float)
    for protocol, prediction, note in (
        (
            "Apparent fit",
            apparent_prediction,
            "Same 4,017 rows used to fit and score the ensemble.",
        ),
        (
            "OOB row exclusion",
            oob_prediction,
            (
                "Each row predicted only by trees that did not contain it; "
                "publications and repeated compositions were not grouped."
            ),
        ),
    ):
        rows.append(
            {
                "study": "Present study",
                "protocol": protocol,
                "model": "Extra Trees (fixed confirmatory configuration)",
                "n": int(len(y)),
                **regression_metrics(y, prediction),
                "R2_std": np.nan,
                "RMSE_std_C": np.nan,
                "note": note,
            }
        )

    pd.DataFrame(
        {
            "row_id": np.arange(len(dataset)),
            "record_id": dataset["record_id"],
            "reference_group": groups,
            "composition_fingerprint": fingerprints,
            "observed_Tg_C": y,
            "apparent_predicted_Tg_C": apparent_prediction,
            "oob_predicted_Tg_C": oob_prediction,
        }
    ).to_csv(RESULTS_DIR / "riley_comparable_predictions.csv", index=False)

    split_rows: list[dict[str, object]] = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        test_fingerprints = set(fingerprints[test_index])
        protocols = {
            "GroupKFold by publication": train_index,
            "Purged GroupKFold by publication": np.asarray(
                [
                    index
                    for index in train_index
                    if fingerprints[index] not in test_fingerprints
                ],
                dtype=int,
            ),
        }
        for protocol, protocol_train in protocols.items():
            fold_model = build_model().fit(x[protocol_train], y[protocol_train])
            prediction = fold_model.predict(x[test_index])
            split_rows.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "n_train": int(len(protocol_train)),
                    "n_test": int(len(test_index)),
                    **regression_metrics(y[test_index], prediction),
                }
            )

    split_metrics = pd.DataFrame(split_rows)
    split_metrics.to_csv(
        RESULTS_DIR / "riley_comparable_split_metrics.csv",
        index=False,
    )
    split_summary = (
        split_metrics.groupby("protocol", as_index=False)
        .agg(
            n=("n_test", "sum"),
            MAE_C=("MAE_C", "mean"),
            RMSE_C=("RMSE_C", "mean"),
            RMSE_std_C=("RMSE_C", "std"),
            R2=("R2", "mean"),
            R2_std=("R2", "std"),
        )
    )
    for result in split_summary.to_dict(orient="records"):
        rows.append(
            {
                "study": "Present study",
                "protocol": result["protocol"],
                "model": "Extra Trees (fixed confirmatory configuration)",
                "n": int(result["n"]),
                "MAE_C": result["MAE_C"],
                "RMSE_C": result["RMSE_C"],
                "R2": result["R2"],
                "R2_std": result["R2_std"],
                "RMSE_std_C": result["RMSE_std_C"],
                "note": (
                    "Five folds; complete publications excluded."
                    if result["protocol"] == "GroupKFold by publication"
                    else (
                        "Five folds; publications and exact compositions "
                        "excluded from training."
                    )
                ),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        RESULTS_DIR / "riley_comparable_metrics.csv",
        index=False,
    )
    report = {
        "model": {
            "name": "ExtraTreesRegressor",
            "n_estimators": 700,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "bootstrap": True,
            "oob_score": True,
            "random_state": 42,
        },
        "interpretation": {
            "apparent": "Descriptive fit, not generalization evidence.",
            "oob": (
                "Comparable row-exclusion question to PRESS, but not "
                "mathematically identical."
            ),
            "grouped": "Generalization to publications excluded from training.",
            "purged": (
                "Generalization to excluded publications and exact "
                "compositions."
            ),
        },
    }
    (RESULTS_DIR / "riley_comparable_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
