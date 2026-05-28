from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from radar.ensemble.base_models import predict_proba, train_base_model
from radar.models.supervised.calibration import calibrate_probabilities


def train_meta_learner(
    meta_features: np.ndarray,
    y: np.ndarray,
    meta_model_name: str = "lightgbm",
    seed: int = 42,
    feature_names: Optional[list[str]] = None,
    *,
    fast: bool = False,
) -> tuple[Any, Any]:
    """Train stacking meta-learner on OOS base model predictions."""
    n = len(meta_features)
    split = int(n * 0.85)
    X_train, X_val = meta_features[:split], meta_features[split:]
    y_train, y_val = y[:split], y[split:]

    model = train_base_model(
        meta_model_name,
        X_train,
        y_train,
        seed=seed,
        feature_names=feature_names,
        fast=fast,
    )
    val_probs = predict_proba(model, X_val)
    cal_probs, calibrator = calibrate_probabilities(val_probs, y_val)
    return model, calibrator


def build_meta_features(base_preds: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Stack base model probabilities as meta-learner input."""
    names = sorted(base_preds.keys())
    cols = [base_preds[name] for name in names]
    feature_names = [f"p_{name}" for name in names]
    return np.column_stack(cols), feature_names


def apply_meta_learner(
    model: Any,
    calibrator: Any,
    base_preds: dict[str, np.ndarray],
) -> np.ndarray:
    meta_X, _ = build_meta_features(base_preds)
    raw = predict_proba(model, meta_X)
    return calibrator.transform(raw)


def ensemble_predictions_df(
    panel: pd.DataFrame,
    base_preds: dict[str, np.ndarray],
    meta_probs: np.ndarray,
) -> pd.DataFrame:
    out = panel[["date", "symbol", "y_direction", "next_return"]].copy()
    for name, probs in base_preds.items():
        out[f"p_{name}"] = probs
    out["p_ensemble"] = meta_probs
    out["p_down"] = 1.0 - meta_probs
    return out
