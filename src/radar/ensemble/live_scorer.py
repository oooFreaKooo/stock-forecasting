from __future__ import annotations

from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.ensemble.base_models import impute_features, predict_proba
from radar.ensemble.meta_learner import apply_meta_learner, build_meta_features
from radar.ensemble.multi_horizon import apply_agreement_filter
from radar.features.pipeline import get_feature_columns, load_feature_panel
from radar.nlp.fusion.memory_enricher import SENTIMENT_FEATURE_COLUMNS, sentiment_values_from_cache

logger = structlog.get_logger(__name__)

ENSEMBLE_BUNDLE_NAME = "ensemble.joblib"


def ensemble_bundle_path(settings: Settings):
    from pathlib import Path

    return Path(settings.paths.models_dir) / "ensemble" / ENSEMBLE_BUNDLE_NAME


def load_ensemble_bundle(settings: Settings) -> Optional[dict[str, Any]]:
    path = ensemble_bundle_path(settings)
    if not path.exists():
        return None
    return joblib.load(path)


def _patch_missing_features(
    row: pd.Series,
    missing: list[str],
    settings: Settings,
    symbol: str,
    fill_values: np.ndarray,
    feature_cols: list[str],
) -> pd.Series:
    patched = row.copy()
    nlp_missing = [c for c in missing if c in SENTIMENT_FEATURE_COLUMNS]
    if nlp_missing:
        live = sentiment_values_from_cache(settings, symbol)
        for col in nlp_missing:
            if col in live:
                patched[col] = live[col]
            else:
                idx = feature_cols.index(col) if col in feature_cols else None
                patched[col] = float(fill_values[idx]) if idx is not None else 0.0
        logger.info("live_score_nlp_imputed", symbol=symbol, cols=nlp_missing)

    still_missing = [c for c in missing if c not in patched.index or pd.isna(patched[c])]
    for col in still_missing:
        if col in feature_cols:
            idx = feature_cols.index(col)
            patched[col] = float(fill_values[idx])
        else:
            patched[col] = 0.0
    return patched


def score_live_symbol(settings: Settings, symbol: str) -> Optional[dict[str, Any]]:
    """Score the latest feature row with the saved ensemble bundle."""
    bundle = load_ensemble_bundle(settings)
    if bundle is None:
        return None

    panel = load_feature_panel(settings)
    panel["date"] = pd.to_datetime(panel["date"])
    sym = panel[panel["symbol"] == symbol.upper()].sort_values("date")
    if sym.empty:
        return None

    row = sym.iloc[-1]
    feature_cols = bundle.get("feature_cols")
    if not feature_cols:
        feature_cols = get_feature_columns(settings, panel)

    missing = [col for col in feature_cols if col not in row.index or pd.isna(row[col])]
    if missing:
        fill_values = np.asarray(bundle["fill_values"])
        row = _patch_missing_features(row, missing, settings, symbol, fill_values, feature_cols)
        still_missing = [col for col in feature_cols if col not in row.index]
        if still_missing:
            logger.warning("live_score_missing_features", symbol=symbol, count=len(still_missing))
            return None

    fill_values = np.asarray(bundle["fill_values"])
    X = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
    X = impute_features(X, fill_values)

    base_preds: dict[str, np.ndarray] = {}
    for model_name in bundle.get("base_models", settings.ensemble.base_models):
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
        "source": "live",
    }
    for model_name, probs in base_preds.items():
        result[f"p_{model_name}"] = float(probs[0])

    horizon_models = bundle.get("horizon_models", {})
    for horizon_key, model in horizon_models.items():
        if model is not None and hasattr(model, "predict_proba"):
            result[horizon_key] = float(model.predict_proba(X)[0, 1])

    if "return_model" in bundle and bundle["return_model"] is not None:
        result["predicted_return_1d"] = float(bundle["return_model"].predict(X)[0])

    trade_frame = pd.DataFrame([{**result, **row.to_dict()}])
    trade_frame = apply_agreement_filter(trade_frame, settings.ensemble)
    result["trade_allowed"] = bool(trade_frame.iloc[0]["trade_allowed"])

    return result


def score_live_universe(settings: Settings) -> dict[str, dict[str, Any]]:
    """Score all traded symbols from the latest feature panel rows."""
    scores: dict[str, dict[str, Any]] = {}
    for symbol in settings.universe.traded:
        live = score_live_symbol(settings, symbol)
        if live is not None:
            scores[symbol] = live
    return scores
