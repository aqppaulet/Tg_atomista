"""Compare the Riley--Vienna Tg protocol with leakage-aware validation.

The published model is a partial-specific-coefficient mixture model fitted to
mass fractions without an intercept.  This script:

1. applies the published coefficients to the current reproducible cohort;
2. refits the same model form and calculates analytical leave-one-out PRESS;
3. evaluates the refitted model with random, publication-grouped, and purged
   publication-grouped splits.

The comparison is diagnostic.  It does not treat in-sample R2 or PRESS R2 as
equivalent to source-grouped external generalization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

from tellurite_tg.data import _parse_composition, _safe_identifier


ROOT = Path(__file__).resolve().parents[1]

# Table 1 of PNNL-26760. Concentrations are mass fractions and the mixture
# equation has no intercept: Tg = sum_i(Tg_i * g_i).
RILEY_COEFFICIENTS_C = {
    "Ag2O": 21.67121,
    "AgI": -105.00570,
    "Al2O3": 896.51740,
    "AlN": 1160.29270,
    "As2O3": 1234.91000,
    "B2O3": 731.82353,
    "BaCl2": 207.81348,
    "BaO": 491.05328,
    "CaO": 833.30717,
    "CdCl2": 190.77415,
    "CdF2": 152.73182,
    "CeO2": 787.99950,
    "Cu2O": 216.12003,
    "Er2O3": 661.71152,
    "F": -1189.65300,
    "Fe2O3": 808.56853,
    "Ga2O3": 583.61413,
    "Gd2O3": 703.86665,
    "GeO2": 442.38509,
    "K2O": -146.97170,
    "La2O3": 702.24469,
    "LaF3": 936.08895,
    "Li2O": -488.87860,
    "LiBr": 95.49429,
    "LiCl": 44.07846,
    "MgO": 1287.08520,
    "Na2O": -244.04290,
    "Nb2O5": 672.53679,
    "NiO": 693.54863,
    "P2O5": 544.26603,
    "PbBr2": 178.47706,
    "PbCl2": 208.51647,
    "PbF2": 169.78958,
    "PbO": 272.25955,
    "Rb2O": -25.32185,
    "Sb2O3": 391.47863,
    "Sm2O3": 723.02931,
    "SrO": 792.55629,
    "Ta2O5": 636.81792,
    "TeO2": 304.39678,
    "TiO2": 1041.49940,
    "Tl2O": 14.594724,
    "TlF": -60.23235,
    "V2O5": 255.73055,
    "VN": 214.24536,
    "WO3": 482.29043,
    "Y2O3": 988.87616,
    "Yb2O3": 851.90815,
    "ZnCl2": 233.47489,
    "ZnF2": 245.30147,
    "Other": 376.54881,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE_C": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_C": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_mass_fraction_design(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    composition_path = ROOT / "data" / "raw" / "GlassPy" / "data" / "select_Gcomp.csv.zip"
    compositions = pd.read_csv(composition_path, sep="\t", low_memory=False)
    compositions = compositions.rename(columns={"Kod": "KOD", "GlasNo": "GLASNO"})
    compositions["record_id"] = [
        _safe_identifier(kod, number)
        for kod, number in zip(compositions["KOD"], compositions["GLASNO"])
    ]
    compositions = compositions.drop_duplicates("record_id", keep=False)
    selected = dataset[["record_id"]].merge(
        compositions[["record_id", "Composition"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    if selected["Composition"].isna().any():
        raise RuntimeError("Some processed records have no raw composition.")

    named_terms = set(RILEY_COEFFICIENTS_C).difference({"Other"})
    rows: list[dict[str, float]] = []
    normalized_compositions: list[dict[str, float]] = []
    raw_weight_sums: list[float] = []
    for cell in selected["Composition"]:
        _, weight = _parse_composition(cell)
        total = float(sum(weight.values()))
        if total <= 0:
            raise RuntimeError("A selected composition has no positive mass total.")
        raw_weight_sums.append(total)
        normalized = {
            component: float(value) / total
            for component, value in weight.items()
            if float(value) > 0
        }
        normalized_compositions.append(normalized)
        row = {
            component: float(normalized.get(component, 0.0))
            for component in named_terms
        }
        row["Other"] = (
            sum(
                float(value)
                for component, value in normalized.items()
                if component not in named_terms
            )
        )
        rows.append(row)

    columns = list(RILEY_COEFFICIENTS_C)
    riley_design = pd.DataFrame(rows, columns=columns, dtype=float)

    component_frequency: dict[str, int] = {}
    for composition in normalized_compositions:
        for component in composition:
            component_frequency[component] = component_frequency.get(component, 0) + 1
    retained_components = sorted(
        component
        for component, frequency in component_frequency.items()
        if frequency >= 5
    )
    frequency_rows: list[dict[str, float]] = []
    for composition in normalized_compositions:
        row = {
            component: float(composition.get(component, 0.0))
            for component in retained_components
        }
        row["Other"] = sum(
            float(value)
            for component, value in composition.items()
            if component not in retained_components
        )
        frequency_rows.append(row)
    frequency_design = pd.DataFrame(
        frequency_rows,
        columns=retained_components + ["Other"],
        dtype=float,
    )

    diagnostics = {
        "raw_weight_sum_min": float(np.min(raw_weight_sums)),
        "raw_weight_sum_max": float(np.max(raw_weight_sums)),
        "normalized_row_sum_min": float(riley_design.sum(axis=1).min()),
        "normalized_row_sum_max": float(riley_design.sum(axis=1).max()),
        "frequency_only_feature_count": int(frequency_design.shape[1]),
    }
    return riley_design, frequency_design, diagnostics


def fit_no_intercept(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    return LinearRegression(fit_intercept=False).fit(x, y)


def fit_prior_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    prior: np.ndarray,
) -> np.ndarray:
    """Fit coefficients penalized toward a specified compositional prior."""

    gram = x.T @ x
    right_hand_side = x.T @ y + alpha * prior
    penalized = gram + alpha * np.eye(x.shape[1])
    return np.linalg.solve(penalized, right_hand_side)


def select_prior_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    prior: np.ndarray,
    alphas: tuple[float, ...],
) -> float:
    """Select regularization using only publication-grouped training data."""

    splitter = GroupKFold(n_splits=min(4, len(np.unique(groups))))
    mean_errors: list[tuple[float, float]] = []
    for alpha in alphas:
        fold_errors: list[float] = []
        for train_index, validation_index in splitter.split(x, groups=groups):
            coefficients = fit_prior_ridge(
                x[train_index],
                y[train_index],
                alpha,
                prior,
            )
            prediction = x[validation_index] @ coefficients
            fold_errors.append(
                float(
                    np.sqrt(
                        mean_squared_error(y[validation_index], prediction)
                    )
                )
            )
        mean_errors.append((float(np.mean(fold_errors)), alpha))
    return min(mean_errors)[1]


def evaluate_nested_prior_ridge(
    dataset: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    priors: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Nested publication-grouped evaluation of regularized mixture models."""

    groups = dataset["reference_group"].astype(str).to_numpy()
    fingerprints = dataset["composition_fingerprint"].astype(str).to_numpy()
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    splitter = GroupKFold(n_splits=5)
    rows: list[dict[str, float | int | str]] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, groups=groups), start=1
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
        for validation, protocol_train in protocols.items():
            for model_name, prior in priors.items():
                alpha = select_prior_ridge_alpha(
                    x[protocol_train],
                    y[protocol_train],
                    groups[protocol_train],
                    prior,
                    alphas,
                )
                coefficients = fit_prior_ridge(
                    x[protocol_train],
                    y[protocol_train],
                    alpha,
                    prior,
                )
                prediction = x[test_index] @ coefficients
                rows.append(
                    {
                        "validation": validation,
                        "split": fold,
                        "model": model_name,
                        "selected_alpha": alpha,
                        "n_train": int(len(protocol_train)),
                        "n_test": int(len(test_index)),
                        **metrics(y[test_index], prediction),
                    }
                )
    return pd.DataFrame(rows)


def press_predictions(
    x: np.ndarray, y: np.ndarray, fitted: LinearRegression
) -> tuple[np.ndarray, np.ndarray]:
    prediction = fitted.predict(x)
    residual = y - prediction
    pseudo_inverse = np.linalg.pinv(x)
    leverage = np.sum(x * pseudo_inverse.T, axis=1)
    denominator = np.clip(1.0 - leverage, 1e-12, None)
    press_residual = residual / denominator
    return y - press_residual, leverage


def remove_five_sigma_outliers(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, list[dict[str, float]]]:
    keep = np.ones(len(y), dtype=bool)
    history: list[dict[str, float]] = []
    for iteration in range(1, 11):
        fitted = fit_no_intercept(x[keep], y[keep])
        prediction = fitted.predict(x[keep])
        residual = y[keep] - prediction
        pseudo_inverse = np.linalg.pinv(x[keep])
        leverage = np.sum(x[keep] * pseudo_inverse.T, axis=1)
        rank = int(np.linalg.matrix_rank(x[keep]))
        degrees_freedom = max(int(keep.sum()) - rank, 1)
        residual_sigma = float(np.sqrt(np.sum(residual**2) / degrees_freedom))
        studentized = residual / (
            residual_sigma * np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
        )
        local_remove = np.abs(studentized) > 5.0
        history.append(
            {
                "iteration": iteration,
                "rows_before": int(keep.sum()),
                "removed": int(local_remove.sum()),
                "residual_sigma_C": residual_sigma,
            }
        )
        if not local_remove.any():
            break
        global_indices = np.flatnonzero(keep)
        keep[global_indices[local_remove]] = False
    return keep, history


def evaluate_splits(
    dataset: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    groups = dataset["reference_group"].astype(str).reset_index(drop=True)
    fingerprints = dataset["composition_fingerprint"].astype(str).reset_index(drop=True)
    all_indices = np.arange(len(dataset))
    rows: list[dict[str, float | int | str]] = []

    def record(
        validation: str,
        split: int,
        train_index: np.ndarray,
        test_index: np.ndarray,
        *,
        remove_training_outliers: bool = False,
    ) -> None:
        fit_index = train_index
        training_outliers_removed = 0
        if remove_training_outliers:
            local_keep, _ = remove_five_sigma_outliers(
                x[train_index], y[train_index]
            )
            training_outliers_removed = int((~local_keep).sum())
            fit_index = train_index[local_keep]
        fitted = fit_no_intercept(x[fit_index], y[fit_index])
        prediction = fitted.predict(x[test_index])
        train_groups = set(groups.iloc[fit_index])
        test_groups = set(groups.iloc[test_index])
        train_fingerprints = set(fingerprints.iloc[fit_index])
        test_fingerprints = set(fingerprints.iloc[test_index])
        overlap = train_fingerprints.intersection(test_fingerprints)
        rows.append(
            {
                "validation": validation,
                "split": split,
                "n_train": int(len(fit_index)),
                "n_test": int(len(test_index)),
                "training_outliers_removed": training_outliers_removed,
                "publication_overlap": int(len(train_groups.intersection(test_groups))),
                "composition_overlap": int(len(overlap)),
                **metrics(y[test_index], prediction),
            }
        )

    for seed in (11, 23, 42, 57, 89):
        train_local, test_local = train_test_split(
            all_indices,
            test_size=0.2,
            random_state=seed,
        )
        record(
            "Riley-like random 80/20",
            seed,
            train_local,
            test_local,
        )
        record(
            "Riley-like random 80/20; 5-sigma training only",
            seed,
            train_local,
            test_local,
            remove_training_outliers=True,
        )

    splitter = GroupKFold(n_splits=5)
    for fold, (train_local, test_local) in enumerate(
        splitter.split(all_indices, groups=groups), start=1
    ):
        train_index = train_local
        test_index = test_local
        record(
            "Riley-like GroupKFold by publication",
            fold,
            train_index,
            test_index,
        )
        record(
            "Riley-like GroupKFold; 5-sigma training only",
            fold,
            train_index,
            test_index,
            remove_training_outliers=True,
        )
        test_fingerprints = set(fingerprints.iloc[test_index])
        purged_train = np.asarray(
            [
                index
                for index in train_index
                if fingerprints.iloc[index] not in test_fingerprints
            ],
            dtype=int,
        )
        record(
            "Riley-like purged GroupKFold",
            fold,
            purged_train,
            test_index,
        )
        record(
            "Riley-like purged GroupKFold; 5-sigma training only",
            fold,
            purged_train,
            test_index,
            remove_training_outliers=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    dataset_path = ROOT / "data" / "processed" / "tellurite_tg_dataset.csv"
    results_dir = ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_csv(dataset_path, low_memory=False)
    design, frequency_design, mass_diagnostics = load_mass_fraction_design(dataset)
    x = design.to_numpy(dtype=float)
    y = dataset["Tg_C"].to_numpy(dtype=float)

    coefficient_vector = np.asarray(
        [RILEY_COEFFICIENTS_C[column] for column in design.columns],
        dtype=float,
    )
    published_prediction = x @ coefficient_vector
    published_metrics = metrics(y, published_prediction)

    refitted_all = fit_no_intercept(x, y)
    fitted_prediction = refitted_all.predict(x)
    press_prediction, leverage = press_predictions(x, y, refitted_all)

    keep, outlier_history = remove_five_sigma_outliers(x, y)
    refitted_inliers = fit_no_intercept(x[keep], y[keep])
    inlier_prediction = refitted_inliers.predict(x[keep])
    inlier_press_prediction, inlier_leverage = press_predictions(
        x[keep], y[keep], refitted_inliers
    )

    split_metrics = evaluate_splits(dataset, x, y)
    prior_ridge_metrics = evaluate_nested_prior_ridge(
        dataset,
        x,
        y,
        {
            "Mass-fraction ridge on Riley terms toward zero": np.zeros(
                x.shape[1], dtype=float
            ),
            "Mass-fraction ridge toward Riley coefficients": coefficient_vector,
        },
    )
    frequency_ridge_metrics = evaluate_nested_prior_ridge(
        dataset,
        frequency_design.to_numpy(dtype=float),
        y,
        {
            "Mass-fraction ridge; frequency-only components": np.zeros(
                frequency_design.shape[1], dtype=float
            )
        },
    )
    prior_ridge_metrics = pd.concat(
        [prior_ridge_metrics, frequency_ridge_metrics],
        ignore_index=True,
    )
    prior_ridge_summary = (
        prior_ridge_metrics.groupby(["validation", "model"], as_index=False)
        .agg(
            n_splits=("split", "nunique"),
            selected_alpha_median=("selected_alpha", "median"),
            MAE_mean_C=("MAE_C", "mean"),
            MAE_std_C=("MAE_C", "std"),
            RMSE_mean_C=("RMSE_C", "mean"),
            RMSE_std_C=("RMSE_C", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .sort_values(["validation", "RMSE_mean_C"])
    )
    split_summary = (
        split_metrics.groupby("validation", as_index=False)
        .agg(
            n_splits=("split", "nunique"),
            n_train_mean=("n_train", "mean"),
            n_test_mean=("n_test", "mean"),
            training_outliers_removed_mean=("training_outliers_removed", "mean"),
            publication_overlap_mean=("publication_overlap", "mean"),
            composition_overlap_mean=("composition_overlap", "mean"),
            MAE_mean_C=("MAE_C", "mean"),
            MAE_std_C=("MAE_C", "std"),
            RMSE_mean_C=("RMSE_C", "mean"),
            RMSE_std_C=("RMSE_C", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .sort_values("RMSE_mean_C")
    )

    comparison_rows = [
        {
            "evaluation": "Published Riley coefficients on current cohort",
            "n": int(len(y)),
            **published_metrics,
        },
        {
            "evaluation": "Riley-like refit, apparent training fit",
            "n": int(len(y)),
            **metrics(y, fitted_prediction),
        },
        {
            "evaluation": "Riley-like refit, analytical PRESS",
            "n": int(len(y)),
            **metrics(y, press_prediction),
        },
        {
            "evaluation": "Riley-like refit after 5-sigma removal, apparent fit",
            "n": int(keep.sum()),
            **metrics(y[keep], inlier_prediction),
        },
        {
            "evaluation": "Riley-like refit after 5-sigma removal, PRESS",
            "n": int(keep.sum()),
            **metrics(y[keep], inlier_press_prediction),
        },
    ]
    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(results_dir / "riley_protocol_comparison.csv", index=False)
    split_metrics.to_csv(results_dir / "riley_protocol_split_metrics.csv", index=False)
    split_summary.to_csv(results_dir / "riley_protocol_split_summary.csv", index=False)
    prior_ridge_metrics.to_csv(
        results_dir / "riley_prior_ridge_split_metrics.csv",
        index=False,
    )
    prior_ridge_summary.to_csv(
        results_dir / "riley_prior_ridge_summary.csv",
        index=False,
    )

    report = {
        "source": "Riley and Vienna, PNNL-26760 (2017), Table 1",
        "cohort_rows": int(len(dataset)),
        "model_terms_including_other": int(x.shape[1]),
        "mass_fraction_diagnostics": mass_diagnostics,
        "published_reference": {
            "observations": 1939,
            "R2_apparent": 0.840634,
            "R2_PRESS": 0.828806,
            "RMSE_C": 22.83494,
            "PRESS_RMSE_C": 23.3599096,
        },
        "outlier_removal": {
            "criterion": "absolute internally studentized residual > 5",
            "rows_removed": int((~keep).sum()),
            "history": outlier_history,
        },
        "leverage": {
            "all_rows_max": float(np.max(leverage)),
            "inliers_max": float(np.max(inlier_leverage)),
        },
        "important_non_equivalences": [
            "The present cohort uses GlassPy supporting files rather than SciGlass v7.0.",
            "The published coefficients were fitted on an overlapping literature database.",
            "PRESS leaves out one row but does not exclude its publication or repeated composition.",
            "Grouped and purged validation estimate a harder new-publication use case.",
        ],
    }
    (results_dir / "riley_protocol_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(comparison.to_string(index=False))
    print()
    print(split_summary.to_string(index=False))
    print()
    print(prior_ridge_summary.to_string(index=False))
    print()
    print(f"5-sigma rows removed: {(~keep).sum()}")


if __name__ == "__main__":
    main()
