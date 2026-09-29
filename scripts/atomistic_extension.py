"""Preparación de datos y prototipo de evaluación de la extensión AIMD.

No atomistic values are generated here. Descriptors must come from documented
simulations, not from Tg or from the composition-only model.
El protocolo de evaluación y el umbral de 15 composiciones son provisionales;
no constituyen una justificación estadística del tamaño de muestra.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/tellurite_tg_dataset.csv"
OUT = ROOT / "data/atomistic"
DESCRIPTORS = ("cn_te_o", "mean_te_o_angstrom", "mean_te_o_te_deg",
               "fraction_te_o3", "fraction_te_o4", "fraction_te_o5")
META = ("composition_fingerprint", "source_file", "simulation_method")


def read_binary():
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        component_names = reader.fieldnames[reader.fieldnames.index("Ag2O"):]
        other = [c for c in component_names if c not in {"TeO2", "ZnO"}]
        rows = []
        for row in reader:
            if (float(row["TeO2"] or 0) > 0
                    and float(row["ZnO"] or 0) > 0
                    and all(abs(float(row[c] or 0)) <= 1e-8 for c in other)):
                rows.append(row)
    return rows


def inventory(rows):
    by_comp = defaultdict(list)
    for row in rows:
        by_comp[row["composition_fingerprint"]].append(row)
    result = []
    for key, group in by_comp.items():
        result.append({
            "composition_fingerprint": key,
            "TeO2_mol_percent": group[0]["TeO2_mol_percent"],
            "ZnO_mol_percent": group[0]["ZnO_mol_percent"],
            "n_measurements": len(group),
            "n_publications": len({r["reference_group"] for r in group}),
        })
    result.sort(key=lambda r: (float(r["ZnO_mol_percent"]), r["composition_fingerprint"]))
    return result


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    rows = read_binary()
    candidates = inventory(rows)
    write_csv(OUT / "binary_teo2_zno_candidates.csv", tuple(candidates[0]), candidates)
    template = [dict.fromkeys((*META, *DESCRIPTORS), "") for _ in candidates]
    for item, candidate in zip(template, candidates):
        item["composition_fingerprint"] = candidate["composition_fingerprint"]
    # Deliberately do not overwrite a populated descriptor file.
    write_csv(OUT / "descriptor_template.csv", (*META, *DESCRIPTORS), template)
    print(json.dumps({"measurements": len(rows), "compositions": len(candidates),
                      "publications": len({r['reference_group'] for r in rows}),
                      "status": "AIMD descriptors not supplied; no hybrid R2 exists"},
                     ensure_ascii=False, indent=2))


def evaluate(path: Path):
    rows = read_binary()
    candidates = {r["composition_fingerprint"]: r for r in inventory(rows)}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set((*META, *DESCRIPTORS)) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing descriptor columns: {sorted(missing)}")
        supplied = list(reader)
    descriptor_by_key = {}
    for row in supplied:
        key = row["composition_fingerprint"]
        if key not in candidates or key in descriptor_by_key:
            raise ValueError(f"Unknown or duplicate composition: {key}")
        if not row["source_file"].strip() or not row["simulation_method"].strip():
            raise ValueError(f"Simulation provenance is missing for {key}")
        vals = [float(row[c]) for c in DESCRIPTORS]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"Non-finite descriptor for {key}")
        if not all(0 <= vals[i] <= 1 for i in (3, 4, 5)):
            raise ValueError(f"Coordination fractions outside [0, 1] for {key}")
        descriptor_by_key[key] = vals
    if len(descriptor_by_key) < 15:
        raise ValueError("At least 15 distinct simulated compositions are required for evaluation")
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold

    matched = [r for r in rows if r["composition_fingerprint"] in descriptor_by_key]
    groups = np.array([r["reference_group"] for r in matched])
    keys = np.array([r["composition_fingerprint"] for r in matched])
    if len(set(groups)) < 5:
        raise ValueError("At least five distinct publications are required")
    y = np.array([float(r["Tg_C"]) for r in matched])
    # The same matched measurements, folds, and estimator are used for both arms.
    x_comp = np.array([[float(r["TeO2_mol_percent"]),
                        float(r["ZnO_mol_percent"])] for r in matched])
    x_hybrid = np.column_stack((x_comp, np.array([descriptor_by_key[k] for k in keys])))
    output = {"n_measurements": len(matched), "n_compositions": len(descriptor_by_key),
              "n_publications": len(set(groups)), "descriptor_file": str(path),
              "comparisons": {}}
    for protocol in ("publication", "publication_and_unseen_composition"):
        predictions = {}
        for label, x in (("composition_only", x_comp), ("composition_plus_aimd", x_hybrid)):
            pred = np.full(len(y), np.nan)
            for train, test in GroupKFold(n_splits=5).split(x, y, groups):
                if protocol == "publication_and_unseen_composition":
                    train = train[~np.isin(keys[train], keys[test])]
                if len(train) < 10 or len(set(keys[train])) < 5:
                    raise ValueError("Too few training compositions after purging")
                model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                            random_state=42, n_jobs=-1)
                model.fit(x[train], y[train])
                pred[test] = model.predict(x[test])
            predictions[label] = {"r2_oof_pooled": float(r2_score(y, pred)),
                                  "mae_C": float(mean_absolute_error(y, pred)),
                                  "rmse_C": float(mean_squared_error(y, pred) ** 0.5)}
        output["comparisons"][protocol] = predictions
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "hybrid_comparison.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estado", action="store_true",
                        help="Mostrar las etapas implementadas y pendientes del nuevo planteamiento")
    parser.add_argument("--descriptors", type=Path,
                        help="CSV with genuine AIMD descriptors; omit to prepare the inventory")
    args = parser.parse_args()
    if args.estado:
        plan = json.loads((ROOT / "configs/atomistic_plan.json").read_text(encoding="utf-8"))
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        evaluate(args.descriptors) if args.descriptors else prepare()
