from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from radar.ensemble.live_scorer import (
    _patch_missing_features,
    load_ensemble_bundle,
)
from radar.ensemble.base_models import impute_features, predict_proba
from radar.ensemble.meta_learner import apply_meta_learner
from radar.features.pipeline import get_feature_columns, load_feature_panel


def score_symbol_at_date(
    settings,
    symbol: str,
    as_of: pd.Timestamp,
) -> Optional[dict[str, Any]]:
    """
    Score the latest feature row on or before as_of (walk-forward safe for backtests).
    """
    bundle = load_ensemble_bundle(settings)
    if bundle is None:
        return None

    panel = load_feature_panel(settings)
    panel["date"] = pd.to_datetime(panel["date"])
    as_of = pd.Timestamp(as_of).normalize()

    sym = panel[
        (panel["symbol"] == symbol.upper()) & (panel["date"] <= as_of)
    ].sort_values("date")
    if sym.empty:
        return None

    row = sym.iloc[-1]
    feature_cols = bundle.get("feature_cols") or get_feature_columns(settings, panel)
    fill_values = np.asarray(bundle["fill_values"])

    missing = [col for col in feature_cols if col not in row.index or pd.isna(row[col])]
    if missing:
        row = _patch_missing_features(row, missing, settings, symbol, fill_values, feature_cols)

    X = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
    X = impute_features(X, fill_values)

    base_preds: dict[str, np.ndarray] = {}
    for model_name in bundle.get("base_models", []):
        model = bundle["base"].get(model_name)
        if model is None:
            continue
        base_preds[model_name] = predict_proba(model, X)

    if not base_preds:
        return None

    meta_probs = apply_meta_learner(bundle["meta"], bundle["calibrator"], base_preds)
    p_up = float(meta_probs[0])

    result: dict[str, Any] = {
        "date": pd.Timestamp(row["date"]),
        "symbol": symbol.upper(),
        "p_up": p_up,
        "p_ensemble": p_up,
        "source": "historical",
    }
    if "return_model" in bundle and bundle["return_model"] is not None:
        result["predicted_return_1d"] = float(bundle["return_model"].predict(X)[0])

    return result
