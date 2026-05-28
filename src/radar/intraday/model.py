from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


MODEL_NAME = "intraday_5m.joblib"


def model_path(settings) -> Path:
    return Path(settings.paths.models_dir) / "intraday" / MODEL_NAME


@dataclass
class IntradayModelBundle:
    model_mu: Any
    model_abs: Any
    feature_cols: list[str]
    symbol_map: dict[str, int]


def save_bundle(settings, bundle: IntradayModelBundle) -> Path:
    path = model_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_mu": bundle.model_mu,
            "model_abs": bundle.model_abs,
            "feature_cols": bundle.feature_cols,
            "symbol_map": bundle.symbol_map,
        },
        path,
    )
    return path


def load_bundle(settings) -> Optional[IntradayModelBundle]:
    path = model_path(settings)
    if not path.exists():
        return None
    raw = joblib.load(path)
    if "model_mu" not in raw and "model" in raw:
        raw["model_mu"] = raw["model"]
        raw["model_abs"] = None
    return IntradayModelBundle(
        model_mu=raw["model_mu"],
        model_abs=raw.get("model_abs"),
        feature_cols=list(raw["feature_cols"]),
        symbol_map=dict(raw.get("symbol_map", {})),
    )


def fit_intraday_model(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> IntradayModelBundle:
    X = X.copy()
    y = y.astype(float)

    # Encode symbol as integer categorical.
    symbols = sorted(set(X["symbol"].astype(str)))
    sym_map = {s: i for i, s in enumerate(symbols)}
    X["symbol_code"] = X["symbol"].astype(str).map(sym_map).astype(int)
    X = X.drop(columns=["symbol"])

    feature_cols = list(X.columns)

    model_mu = LGBMRegressor(
        objective="huber",
        n_estimators=3000,
        learning_rate=0.015,
        num_leaves=47,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model_abs = LGBMRegressor(
        objective="regression",
        n_estimators=1500,
        learning_rate=0.025,
        num_leaves=47,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None:
        Xv = X_val.copy()
        yv = y_val.astype(float)
        Xv["symbol_code"] = Xv["symbol"].astype(str).map(sym_map).fillna(-1).astype(int)
        Xv = Xv.drop(columns=["symbol"])
        fit_kwargs = {
            "eval_set": [(Xv[feature_cols], yv.values)],
            "eval_metric": "l1",
            "callbacks": [],
        }
        try:
            from lightgbm import early_stopping

            fit_kwargs["callbacks"] = [early_stopping(80, verbose=False)]
        except ImportError:
            pass

    model_mu.fit(X[feature_cols], y.values, **fit_kwargs)
    model_abs.fit(X[feature_cols], np.abs(y.values.astype(float)))

    return IntradayModelBundle(
        model_mu=model_mu,
        model_abs=model_abs,
        feature_cols=feature_cols,
        symbol_map=sym_map,
    )


def predict_next_return(
    bundle: IntradayModelBundle,
    features: pd.DataFrame,
) -> float:
    X = features.copy()
    sym = str(X["symbol"].iloc[0])
    code = bundle.symbol_map.get(sym)
    if code is None:
        # Unseen symbol: fall back to 0 drift.
        return 0.0
    X["symbol_code"] = code
    X = X.drop(columns=["symbol"])
    X = X[bundle.feature_cols]
    pred = float(bundle.model_mu.predict(X.to_numpy(dtype=float))[0])
    if not np.isfinite(pred):
        return 0.0
    return float(np.clip(pred, -0.05, 0.05))


def predict_next_abs_return(
    bundle: IntradayModelBundle,
    features: pd.DataFrame,
) -> float:
    if bundle.model_abs is None:
        return 0.0
    X = features.copy()
    sym = str(X["symbol"].iloc[0])
    code = bundle.symbol_map.get(sym)
    if code is None:
        return 0.0
    X["symbol_code"] = code
    X = X.drop(columns=["symbol"])
    X = X[bundle.feature_cols]
    pred = float(bundle.model_abs.predict(X.to_numpy(dtype=float))[0])
    if not np.isfinite(pred):
        return 0.0
    return float(np.clip(pred, 0.0, 0.05))


def encode_features_for_bundle(
    bundle: IntradayModelBundle,
    X: pd.DataFrame,
) -> pd.DataFrame:
    work = X.copy()
    sym = work["symbol"].astype(str)
    work["symbol_code"] = sym.map(bundle.symbol_map).fillna(-1).astype(int)
    work = work.drop(columns=["symbol"])
    return work[bundle.feature_cols]

