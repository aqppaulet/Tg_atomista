"""Data preparation for the tellurite-glass Tg study.

The raw files are the public SciGlass-derived files distributed with GlassPy
0.5.3 (Zenodo record 13625981).  Temperatures are retained in degrees Celsius,
which is the unit used in the raw ``TG`` field.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ZENODO_URL = (
    "https://zenodo.org/records/13625981/files/GlassPy_files.zip?download=1"
)
ZENODO_MD5 = "b5821e817bf1d15adc7fef6002b6f9ec"

# SciGlass aggregate/ambiguous labels excluded by GlassPy's own preparation
# logic because they cannot be mapped to unique compounds.
AMBIGUOUS_COMPONENTS = {
    "",
    "Al2O3+Fe2O3",
    "MoO3+WO3",
    "CaO+MgO",
    "FeO+Fe2O3",
    "Li2O+Na2O+K2O",
    "Na2O+K2O",
    "F2O-1",
    "FemOn",
    "HF+H2O",
    "R2O",
    "R2O3",
    "RO",
    "RmOn",
}


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_raw_data(raw_dir: Path) -> tuple[Path, Path]:
    """Download, verify, and extract the GlassPy data archive when necessary."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "GlassPy_files.zip"
    extracted = raw_dir / "GlassPy" / "data"
    props_path = extracted / "select_SciGK.csv.zip"
    comp_path = extracted / "select_Gcomp.csv.zip"

    if not archive.exists():
        urllib.request.urlretrieve(ZENODO_URL, archive)
    checksum = file_md5(archive)
    if checksum.lower() != ZENODO_MD5:
        raise RuntimeError(
            f"Checksum mismatch for {archive}: {checksum}; expected {ZENODO_MD5}."
        )
    if not props_path.exists() or not comp_path.exists():
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(raw_dir / "GlassPy")
    return props_path, comp_path


def _parse_composition(cell: Any) -> tuple[dict[str, float], dict[str, float]]:
    """Parse a GlassPy composition cell into mol% and wt% dictionaries."""

    if not isinstance(cell, str) or not cell:
        return {}, {}
    tokens = cell.strip("\x7f").split("\x7f")
    mol: dict[str, float] = {}
    wt: dict[str, float] = {}
    # Each compound is encoded as formula, molar mass, wt%, mol%.
    for index in range(0, len(tokens) - 3, 4):
        component = tokens[index].strip()
        if component in AMBIGUOUS_COMPONENTS:
            continue
        try:
            weight_value = float(tokens[index + 2])
            mol_value = float(tokens[index + 3])
        except (TypeError, ValueError):
            continue
        if np.isfinite(weight_value) and weight_value != 0:
            wt[component] = wt.get(component, 0.0) + weight_value
        if np.isfinite(mol_value) and mol_value != 0:
            mol[component] = mol.get(component, 0.0) + mol_value
    return mol, wt


def _safe_identifier(kod: Any, glass_number: Any) -> str:
    return f"{str(kod).strip()}::{str(glass_number).strip()}"


def _safe_group(kod: Any) -> str:
    value = str(kod).strip()
    return value if value and value.lower() != "nan" else "UNKNOWN"


def prepare_dataset(
    props_path: Path,
    comp_path: Path,
    processed_dir: Path,
    *,
    tellurite_min_wt_percent: float = 52.0,
    apply_component_domain_filter: bool = True,
    zno_min_mol_percent: float = 10.0,
    zno_max_mol_percent: float = 20.0,
    teo2_min_mol_percent: float = 60.0,
    teo2_max_mol_percent: float = 90.0,
    la2o3_min_mol_percent: float = 1.0,
    la2o3_max_mol_percent: float = 10.0,
    retain_repeated_measurements: bool = True,
    composition_sum_tolerance: float = 1.0,
    minimum_component_frequency: int = 5,
    fingerprint_decimals: int = 3,
    target_min_C: float = 100.0,
    target_max_C: float = 600.0,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Create the leakage-controlled tellurite Tg modeling table.

    The returned table contains metadata followed by molar composition features.
    The modeling domain is restricted to inclusive ZnO and TeO2 molar ranges.
    Repeated measurements are retained by default so their inter-publication
    reproducibility can be evaluated explicitly.
    """

    processed_dir.mkdir(parents=True, exist_ok=True)
    props = pd.read_csv(props_path, sep="\t", low_memory=False)
    compositions = pd.read_csv(comp_path, sep="\t", low_memory=False)

    required_props = {"KOD", "GLASNO", "TG"}
    compositions = compositions.rename(columns={"Kod": "KOD", "GlasNo": "GLASNO"})
    required_comp = {"KOD", "GLASNO", "Composition"}
    if missing := required_props.difference(props.columns):
        raise ValueError(f"Missing property columns: {sorted(missing)}")
    if missing := required_comp.difference(compositions.columns):
        raise ValueError(f"Missing composition columns: {sorted(missing)}")

    props = props.copy()
    compositions = compositions.copy()
    props["record_id"] = [
        _safe_identifier(kod, number)
        for kod, number in zip(props["KOD"], props["GLASNO"])
    ]
    compositions["record_id"] = [
        _safe_identifier(kod, number)
        for kod, number in zip(compositions["KOD"], compositions["GLASNO"])
    ]

    initial_property_rows = int(len(props))
    props["Tg_C"] = pd.to_numeric(props["TG"], errors="coerce")
    props = props.loc[np.isfinite(props["Tg_C"])].copy()
    props = props.drop_duplicates("record_id", keep=False)

    compositions = compositions.drop_duplicates("record_id", keep=False)
    joined = props.merge(
        compositions[["record_id", "Composition"]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )

    parsed = joined["Composition"].map(_parse_composition)
    joined["_mol"] = parsed.map(lambda pair: pair[0])
    joined["_wt"] = parsed.map(lambda pair: pair[1])
    joined["mol_sum_raw"] = joined["_mol"].map(lambda item: float(sum(item.values())))
    joined["wt_sum_raw"] = joined["_wt"].map(lambda item: float(sum(item.values())))
    joined["TeO2_wt_percent"] = joined["_wt"].map(
        lambda item: float(item.get("TeO2", 0.0))
    )

    lower = 100.0 - composition_sum_tolerance
    upper = 100.0 + composition_sum_tolerance
    valid_sum = joined["mol_sum_raw"].between(lower, upper, inclusive="both")
    tellurite = joined["TeO2_wt_percent"] >= tellurite_min_wt_percent
    joined = joined.loc[valid_sum & tellurite].copy()
    rows_after_sum_and_tellurite_filter = int(len(joined))
    joined["reference_group"] = joined["KOD"].map(_safe_group)

    # Normalize mol% after checking the original analytical total.
    joined["_mol"] = [
        {name: value * 100.0 / total for name, value in composition.items()}
        for composition, total in zip(joined["_mol"], joined["mol_sum_raw"])
    ]
    joined["ZnO_mol_percent"] = joined["_mol"].map(
        lambda item: float(item.get("ZnO", 0.0))
    )
    joined["TeO2_mol_percent"] = joined["_mol"].map(
        lambda item: float(item.get("TeO2", 0.0))
    )
    joined["La2O3_mol_percent"] = joined["_mol"].map(
        lambda item: float(item.get("La2O3", 0.0))
    )
    if apply_component_domain_filter:
        selected_domain = joined["ZnO_mol_percent"].between(
            zno_min_mol_percent, zno_max_mol_percent, inclusive="both"
        ) & joined["TeO2_mol_percent"].between(
            teo2_min_mol_percent, teo2_max_mol_percent, inclusive="both"
        ) & joined["La2O3_mol_percent"].between(
            la2o3_min_mol_percent, la2o3_max_mol_percent, inclusive="both"
        )
    else:
        selected_domain = pd.Series(True, index=joined.index)
    joined = joined.loc[selected_domain].copy()

    component_frequency: Counter[str] = Counter()
    for composition in joined["_mol"]:
        component_frequency.update(
            name for name, value in composition.items() if value > 0
        )
    retained_components = sorted(
        name
        for name, count in component_frequency.items()
        if count >= minimum_component_frequency
    )

    feature_rows: list[dict[str, float]] = []
    for composition in joined["_mol"]:
        row = {name: float(composition.get(name, 0.0)) for name in retained_components}
        row["Other"] = float(
            sum(
                value
                for name, value in composition.items()
                if name not in retained_components
            )
        )
        feature_rows.append(row)
    features = pd.DataFrame(feature_rows, index=joined.index).fillna(0.0)
    feature_columns = retained_components + ["Other"]

    def exact_fingerprint(composition: dict[str, float]) -> str:
        entries = []
        for name, value in sorted(composition.items()):
            rounded = round(float(value), fingerprint_decimals)
            if rounded != 0:
                entries.append(
                    f"{name}:{rounded:.{fingerprint_decimals}f}"
                )
        return "|".join(entries)

    joined["composition_fingerprint"] = joined["_mol"].map(exact_fingerprint)

    metadata_columns = [
        "record_id",
        "composition_fingerprint",
        "reference_group",
        "Tg_C",
        "TeO2_wt_percent",
        "ZnO_mol_percent",
        "TeO2_mol_percent",
        "La2O3_mol_percent",
    ]
    for optional in ("Author", "Year", "Analysis", "Glass_Class"):
        if optional in joined.columns:
            metadata_columns.append(optional)
    modeling = pd.concat([joined[metadata_columns], features], axis=1)

    repeated_measurements_before_policy = int(len(modeling))
    if not retain_repeated_measurements:
        aggregations: dict[str, str] = {
            "record_id": "first",
            "Tg_C": "median",
            "TeO2_wt_percent": "median",
            "ZnO_mol_percent": "first",
            "TeO2_mol_percent": "first",
            "La2O3_mol_percent": "first",
        }
        for optional in ("Author", "Year", "Analysis", "Glass_Class"):
            if optional in modeling.columns:
                aggregations[optional] = "first"
        aggregations.update({feature: "first" for feature in feature_columns})
        modeling = (
            modeling.groupby(
                ["reference_group", "composition_fingerprint"], as_index=False
            )
            .agg(aggregations)
            .reset_index(drop=True)
        )

    # Broad, declared plausibility guard. It removes only clearly anomalous
    # values and does not replace them with guessed corrections.
    physical_target = modeling["Tg_C"].between(
        target_min_C, target_max_C, inclusive="both"
    )
    nonphysical_rows = int((~physical_target).sum())
    modeling = modeling.loc[physical_target].reset_index(drop=True)

    publication_level_replicates = (
        modeling.groupby(
            ["composition_fingerprint", "reference_group"], as_index=False
        )
        .agg(
            publication_Tg_C=("Tg_C", "median"),
            measurements_in_publication=("Tg_C", "size"),
        )
    )
    publication_count = publication_level_replicates.groupby(
        "composition_fingerprint"
    )["reference_group"].nunique()
    repeated_fingerprints = publication_count[publication_count > 1].index
    repeated_rows = modeling["composition_fingerprint"].isin(
        repeated_fingerprints
    )
    cross_publication_replicates = modeling.loc[repeated_rows].copy()
    cross_publication_replicates.to_csv(
        processed_dir / "cross_publication_replicates.csv", index=False
    )
    replicate_summary = (
        publication_level_replicates.loc[
            publication_level_replicates["composition_fingerprint"].isin(
                repeated_fingerprints
            )
        ]
        .groupby("composition_fingerprint", as_index=False)
        .agg(
            n_publications=("reference_group", "nunique"),
            n_publication_measurements=("publication_Tg_C", "size"),
            Tg_median_C=("publication_Tg_C", "median"),
            Tg_min_C=("publication_Tg_C", "min"),
            Tg_max_C=("publication_Tg_C", "max"),
            Tg_std_C=("publication_Tg_C", "std"),
        )
    )
    replicate_summary["Tg_range_C"] = (
        replicate_summary["Tg_max_C"] - replicate_summary["Tg_min_C"]
    )
    replicate_summary.to_csv(
        processed_dir / "cross_publication_replicate_summary.csv", index=False
    )

    feature_columns = [
        feature
        for feature in feature_columns
        if feature in modeling.columns and (modeling[feature] > 0).any()
    ]
    modeling.to_csv(processed_dir / "tellurite_tg_dataset.csv", index=False)

    component_table = pd.DataFrame(
        [
            {
                "component": component,
                "raw_frequency": int(component_frequency[component]),
                "retained_as_feature": component in feature_columns,
            }
            for component in sorted(component_frequency)
        ]
    )
    component_table.to_csv(processed_dir / "component_frequencies.csv", index=False)

    audit: dict[str, Any] = {
        "source": {
            "name": "SciGlass-derived GlassPy supporting files",
            "zenodo_record": "13625981",
            "doi": "10.5281/zenodo.13625981",
            "archive_md5": ZENODO_MD5,
            "target_unit": "degrees Celsius (raw SciGlass TG field)",
        },
        "criteria": {
            "tellurite_min_wt_percent": tellurite_min_wt_percent,
            "composition_basis": "mol%",
            "composition_sum_range": [lower, upper],
            "component_domain_filter_enabled": apply_component_domain_filter,
            "zno_mol_percent_range_inclusive": [
                zno_min_mol_percent,
                zno_max_mol_percent,
            ],
            "teo2_mol_percent_range_inclusive": [
                teo2_min_mol_percent,
                teo2_max_mol_percent,
            ],
            "la2o3_mol_percent_range_inclusive": [
                la2o3_min_mol_percent,
                la2o3_max_mol_percent,
            ],
            "retain_repeated_measurements": retain_repeated_measurements,
            "minimum_component_frequency": minimum_component_frequency,
            "fingerprint_decimals": fingerprint_decimals,
            "target_range_C": [target_min_C, target_max_C],
        },
        "counts": {
            "raw_property_rows": initial_property_rows,
            "property_rows_with_numeric_tg_and_unique_id": int(len(props)),
            "joined_unique_rows": int(len(parsed)),
            "rows_after_sum_and_tellurite_filter": (
                rows_after_sum_and_tellurite_filter
            ),
            "rows_after_domain_and_sum_filters": (
                repeated_measurements_before_policy
            ),
            "rows_after_component_domain_filters": (
                repeated_measurements_before_policy
            ),
            "rows_removed_cross_publication_duplicate_composition": 0,
            "rows_before_within_publication_aggregation": (
                repeated_measurements_before_policy
            ),
            "rows_after_within_publication_aggregation": int(
                len(modeling) + nonphysical_rows
            ),
            "repeated_measurements_retained": int(
                len(modeling) + nonphysical_rows
            ),
            "rows_removed_nonphysical_target": nonphysical_rows,
            "final_rows": int(len(modeling)),
            "unique_compositions": int(
                modeling["composition_fingerprint"].nunique()
            ),
            "cross_publication_repeated_compositions": int(
                len(repeated_fingerprints)
            ),
            "rows_in_cross_publication_repeats": int(repeated_rows.sum()),
            "publication_composition_measurements": int(
                len(publication_level_replicates)
            ),
            "reference_groups": int(modeling["reference_group"].nunique()),
            "features": int(len(feature_columns)),
        },
        "target_summary_C": {
            key: float(value)
            for key, value in modeling["Tg_C"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .to_dict()
            .items()
        },
        "feature_columns": feature_columns,
    }
    (processed_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return modeling, feature_columns, audit


def load_external_dataset(
    path: Path,
    feature_columns: list[str],
    *,
    zno_min_mol_percent: float = 10.0,
    zno_max_mol_percent: float = 20.0,
    teo2_min_mol_percent: float = 60.0,
    teo2_max_mol_percent: float = 90.0,
    la2o3_min_mol_percent: float = 1.0,
    la2o3_max_mol_percent: float = 10.0,
    enforce_component_domain: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series]:
    """Load an optional independent-laboratory CSV without imputing chemistry."""

    external = pd.read_csv(path)
    if "sample_id" not in external.columns:
        external.insert(0, "sample_id", [f"external_{i + 1}" for i in range(len(external))])
    feature_frame = pd.DataFrame(0.0, index=external.index, columns=feature_columns)
    known_components = [column for column in feature_columns if column != "Other"]
    for component in known_components:
        if component in external.columns:
            feature_frame[component] = pd.to_numeric(
                external[component], errors="coerce"
            ).fillna(0.0)
    if "Other" in feature_columns:
        supplied_components = [
            column
            for column in external.columns
            if re.fullmatch(r"[A-Z][A-Za-z0-9()+\-]*", str(column))
            and column not in {"Tg_C"}
        ]
        feature_frame["Other"] = external[
            [column for column in supplied_components if column not in known_components]
        ].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    totals = feature_frame.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("Every external row must contain a positive composition total.")
    feature_frame = feature_frame.div(totals, axis=0) * 100.0
    if enforce_component_domain:
        inside_domain = feature_frame["ZnO"].between(
            zno_min_mol_percent, zno_max_mol_percent, inclusive="both"
        ) & feature_frame["TeO2"].between(
            teo2_min_mol_percent, teo2_max_mol_percent, inclusive="both"
        ) & feature_frame["La2O3"].between(
            la2o3_min_mol_percent, la2o3_max_mol_percent, inclusive="both"
        )
        domain_description = (
            f"ZnO {zno_min_mol_percent:g}-{zno_max_mol_percent:g} mol%, "
            f"TeO2 {teo2_min_mol_percent:g}-{teo2_max_mol_percent:g} mol%, "
            f"La2O3 {la2o3_min_mol_percent:g}-{la2o3_max_mol_percent:g} mol%"
        )
    else:
        inside_domain = feature_frame["TeO2"] > 0
        domain_description = "presencia de TeO2"
    if not inside_domain.all():
        invalid_samples = external.loc[~inside_domain, "sample_id"].astype(str)
        raise ValueError(
            "External rows outside the inclusive model domain "
            f"({domain_description}): "
            + ", ".join(invalid_samples.head(10))
        )
    target = (
        pd.to_numeric(external["Tg_C"], errors="coerce")
        if "Tg_C" in external.columns
        else None
    )
    return feature_frame, target, external["sample_id"].astype(str)
