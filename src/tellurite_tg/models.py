"""Fixed, publication-ready model definitions."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def build_models(config: dict[str, Any]) -> dict[str, Any]:
    """Instantiate models from a preregistered, non-tuned configuration."""

    seed = int(config["seed"])
    ridge_cfg = config["models"]["ridge"]
    forest_cfg = config["models"]["random_forest"]
    xgb_cfg = config["models"]["xgboost"]
    return {
        "Ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=float(ridge_cfg["alpha"]))),
            ]
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=int(forest_cfg["n_estimators"]),
            max_features=float(forest_cfg["max_features"]),
            min_samples_leaf=int(forest_cfg["min_samples_leaf"]),
            random_state=seed,
            n_jobs=int(forest_cfg["n_jobs"]),
        ),
        "XGBoost": XGBRegressor(
            n_estimators=int(xgb_cfg["n_estimators"]),
            max_depth=int(xgb_cfg["max_depth"]),
            learning_rate=float(xgb_cfg["learning_rate"]),
            subsample=float(xgb_cfg["subsample"]),
            colsample_bytree=float(xgb_cfg["colsample_bytree"]),
            min_child_weight=float(xgb_cfg["min_child_weight"]),
            reg_lambda=float(xgb_cfg["reg_lambda"]),
            objective=str(xgb_cfg["objective"]),
            random_state=seed,
            n_jobs=int(xgb_cfg["n_jobs"]),
            tree_method="hist",
        ),
    }
