"""Evaluate XGBoost experts for overlapping tellurite-glass families.

This is a secondary appendix analysis.  It does not replace the global-model
results.  Each expert uses the same fixed XGBoost hyperparameters as the main
study and is trained only on rows belonging to its chemical family.

Family memberships intentionally overlap because, for example, a glass may
contain both ZnO and a rare-earth oxide.  This avoids an arbitrary precedence
rule and makes every family criterion independently reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from tellurite_tg.evaluation import regression_metrics
from tellurite_tg.models import build_models


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "artifacts" / "results"

FAMILY_COMPONENTS: dict[str, tuple[str, ...]] = {
    "Teluritos alcalinos": (
        "Li2O",
        "Na2O",
        "K2O",
        "Rb2O",
        "Cs2O",
    ),
    "Teluritos con ZnO": ("ZnO",),
    "Teluritos con oxidos de tierras raras": (
        "Y2O3",
        "La2O3",
        "Ce2O3",
        "CeO2",
        "Pr2O3",
        "Pr6O11",
        "Nd2O3",
        "Sm2O3",
        "Eu2O3",
        "Gd2O3",
        "Dy2O3",
        "Ho2O3",
        "Er2O3",
        "Tm2O3",
        "Yb2O3",
    ),
    "Teluritos con Nb2O5, WO3 o Bi2O3": (
        "Nb2O5",
        "WO3",
        "Bi2O3",
    ),
    "Sistemas teluritos con haluros": (
        "AgI",
        "AlF3",
        "BaCl2",
        "BaF2",
        "CaF2",
        "CdCl2",
        "CdF2",
        "CeCl3",
        "Cl",
        "CsCl",
        "ErF3",
        "F",
        "KCl",
        "KF",
        "LaCl3",
        "LaF3",
        "LiBr",
        "LiCl",
        "LiF",
        "MgF2",
        "NaCl",
        "NaF",
        "NdCl3",
        "PbBr2",
        "PbCl2",
        "PbF2",
        "PrCl3",
        "SmCl3",
        "SrF2",
        "TeCl4",
        "TlF",
        "YF3",
        "YbF3",
        "ZnBr2",
        "ZnCl2",
        "ZnF2",
    ),
}


def family_mask(
    dataset: pd.DataFrame,
    components: tuple[str, ...],
) -> pd.Series:
    available = [component for component in components if component in dataset]
    if not available:
        raise RuntimeError(f"No family components are available: {components}")
    return dataset[available].gt(0).any(axis=1)


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
    config = json.loads(
        (ROOT / "configs" / "pipeline.json").read_text(encoding="utf-8")
    )
    feature_columns = list(audit["feature_columns"])
    base_model = build_models(config)["XGBoost"]

    membership = pd.DataFrame(
        {
            "row_id": np.arange(len(dataset)),
            "record_id": dataset["record_id"],
            "reference_group": dataset["reference_group"].astype(str),
            "composition_fingerprint": dataset["composition_fingerprint"],
        }
    )
    for family, components in FAMILY_COMPONENTS.items():
        membership[family] = family_mask(dataset, components).to_numpy()
    membership["number_of_families"] = membership[
        list(FAMILY_COMPONENTS)
    ].sum(axis=1)
    membership.to_csv(
        RESULTS_DIR / "family_expert_membership.csv",
        index=False,
    )

    split_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    family_counts: dict[str, dict[str, int | float | list[str]]] = {}
    for family, components in FAMILY_COMPONENTS.items():
        subset = dataset.loc[membership[family].to_numpy()].reset_index(
            drop=True
        )
        x = subset[feature_columns].to_numpy(dtype=float)
        y = subset["Tg_C"].to_numpy(dtype=float)
        groups = subset["reference_group"].astype(str).to_numpy()
        fingerprints = subset["composition_fingerprint"].astype(str).to_numpy()
        n_groups = int(pd.Series(groups).nunique())
        n_splits = min(5, n_groups)
        if n_splits < 2:
            raise RuntimeError(
                f"Family {family!r} has fewer than two publication groups."
            )

        family_counts[family] = {
            "rows": int(len(subset)),
            "unique_compositions": int(
                subset["composition_fingerprint"].nunique()
            ),
            "publications": n_groups,
            "target_std_C": float(np.std(y, ddof=1)),
            "membership_components": list(components),
        }

        splitter = GroupKFold(n_splits=n_splits)
        for fold, (train_index, test_index) in enumerate(
            splitter.split(x, y, groups), start=1
        ):
            train_groups = set(groups[train_index])
            test_groups = set(groups[test_index])
            if train_groups.intersection(test_groups):
                raise AssertionError("Publication leakage in family expert.")
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
            for validation, protocol_train in protocols.items():
                fitted = clone(base_model).fit(
                    x[protocol_train],
                    y[protocol_train],
                )
                predicted = fitted.predict(x[test_index])
                train_fingerprints = set(fingerprints[protocol_train])
                composition_overlap = train_fingerprints.intersection(
                    test_fingerprints
                )
                split_rows.append(
                    {
                        "family": family,
                        "validation": validation,
                        "split": fold,
                        "n_train": int(len(protocol_train)),
                        "n_test": int(len(test_index)),
                        "publication_overlap": 0,
                        "composition_overlap": int(len(composition_overlap)),
                        **regression_metrics(y[test_index], predicted),
                    }
                )
                prediction_rows.extend(
                    {
                        "family": family,
                        "validation": validation,
                        "split": fold,
                        "subset_row_id": int(row_index),
                        "record_id": subset.iloc[row_index]["record_id"],
                        "reference_group": groups[row_index],
                        "composition_fingerprint": fingerprints[row_index],
                        "observed_Tg_C": float(y[row_index]),
                        "predicted_Tg_C": float(prediction),
                    }
                    for row_index, prediction in zip(test_index, predicted)
                )

    split_metrics = pd.DataFrame(split_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = (
        split_metrics.groupby(["family", "validation"], as_index=False)
        .agg(
            n_splits=("split", "nunique"),
            n_train_mean=("n_train", "mean"),
            n_test_mean=("n_test", "mean"),
            MAE_mean_C=("MAE_C", "mean"),
            MAE_std_C=("MAE_C", "std"),
            RMSE_mean_C=("RMSE_C", "mean"),
            RMSE_std_C=("RMSE_C", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            publication_overlap_mean=("publication_overlap", "mean"),
            composition_overlap_mean=("composition_overlap", "mean"),
        )
        .sort_values(["family", "validation"])
    )
    pooled_rows: list[dict[str, object]] = []
    for (family, validation), subset in predictions.groupby(
        ["family", "validation"]
    ):
        pooled_rows.append(
            {
                "family": family,
                "validation": validation,
                "n": int(len(subset)),
                **regression_metrics(
                    subset["observed_Tg_C"],
                    subset["predicted_Tg_C"],
                ),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    summary = summary.merge(
        pooled.rename(
            columns={
                "MAE_C": "MAE_pooled_C",
                "RMSE_C": "RMSE_pooled_C",
                "R2": "R2_pooled",
            }
        ),
        on=["family", "validation"],
        how="left",
        validate="one_to_one",
    )

    split_metrics.to_csv(
        RESULTS_DIR / "family_expert_split_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        RESULTS_DIR / "family_expert_predictions.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS_DIR / "family_expert_summary.csv",
        index=False,
    )
    report = {
        "status": "secondary_appendix_analysis",
        "model": "XGBoost",
        "model_configuration": config["models"]["xgboost"],
        "families_are_mutually_exclusive": False,
        "family_counts": family_counts,
        "membership_overlap": {
            str(int(key)): int(value)
            for key, value in membership["number_of_families"]
            .value_counts()
            .sort_index()
            .items()
        },
        "interpretation": (
            "Family-specific R2 values are conditional on target variance "
            "and publication composition in each family. They are an appendix "
            "analysis and must not replace the global primary result."
        ),
    }
    (RESULTS_DIR / "family_expert_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
