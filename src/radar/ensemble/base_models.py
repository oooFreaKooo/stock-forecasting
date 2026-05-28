from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class BaseClassifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def fit_imputer(X_train: np.ndarray) -> np.ndarray:
    """Column medians from training data for NaN imputation."""
    clean = np.where(np.isinf(X_train), np.nan, X_train)
    medians = np.nanmedian(clean, axis=0)
    return np.where(np.isnan(medians), 0.0, medians)


def impute_features(X: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64)
    out = np.where(np.isinf(out), np.nan, out)
    for j in range(out.shape[1]):
        mask = np.isnan(out[:, j])
        if mask.any():
            out[mask, j] = fill_values[j]
    return out


def _default_feature_names(n_features: int) -> list[str]:
    return [f"feature_{idx}" for idx in range(n_features)]


def _resolve_feature_names(
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> list[str]:
    if feature_names is not None:
        return list(feature_names)
    return _default_feature_names(X.shape[1])


def _to_model_frame(X: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(X, dtype=np.float64), columns=list(feature_names))


def _prepare_matrix(
    X: np.ndarray,
    fill_values: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    fill = fit_imputer(X) if fill_values is None else fill_values
    return impute_features(X, fill), fill


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    feature_names: Optional[Sequence[str]] = None,
    *,
    fast: bool = False,
) -> Any:
    from radar.models.supervised.lightgbm_model import train_classifier

    X_clean, fill = _prepare_matrix(X_train)
    names = _resolve_feature_names(X_clean, feature_names)
    n = len(X_clean)
    split = int(n * 0.9)
    model = train_classifier(
        X_clean[:split],
        y_train[:split],
        X_clean[split:],
        y_train[split:],
        seed=seed,
        feature_names=names,
        fast=fast,
    )
    model._radar_impute_fill = fill  # type: ignore[attr-defined]
    model._radar_feature_names = names  # type: ignore[attr-defined]
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    feature_names: Optional[Sequence[str]] = None,
) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost required for ensemble. pip install xgboost") from exc

    X_clean, fill = _prepare_matrix(X_train)
    names = _resolve_feature_names(X_clean, feature_names)
    n = len(X_clean)
    split = int(n * 0.9)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(_to_model_frame(X_clean[:split], names), y_train[:split])
    model._radar_impute_fill = fill  # type: ignore[attr-defined]
    model._radar_feature_names = names  # type: ignore[attr-defined]
    return model


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    feature_names: Optional[Sequence[str]] = None,
) -> LogisticRegression:
    X_clean, fill = _prepare_matrix(X_train)
    names = _resolve_feature_names(X_clean, feature_names)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed, solver="liblinear")
    model.fit(X_scaled, y_train)
    model._radar_impute_fill = fill  # type: ignore[attr-defined]
    model._radar_scaler = scaler  # type: ignore[attr-defined]
    model._radar_feature_names = names  # type: ignore[attr-defined]
    return model


TRAINERS = {
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
    "logistic": train_logistic,
}


def train_base_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    feature_names: Optional[Sequence[str]] = None,
    *,
    fast: bool = False,
) -> Any:
    if name not in TRAINERS:
        raise ValueError(f"Unknown base model: {name}. Choose from {list(TRAINERS)}")
    trainer = TRAINERS[name]
    if name == "lightgbm":
        return trainer(X, y, seed=seed, feature_names=feature_names, fast=fast)
    return trainer(X, y, seed=seed, feature_names=feature_names)


def _logistic_predict_proba(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Stable batched logistic predict (avoids BLAS matmul warnings on large batches)."""
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    probs: list[np.ndarray] = []
    chunk_size = 8
    for start in range(0, len(X), chunk_size):
        chunk = np.asarray(X[start:start + chunk_size], dtype=np.float64)
        logits = np.clip(chunk @ coef.T + intercept, -500.0, 500.0)
        probs.append((1.0 / (1.0 + np.exp(-logits))).ravel())
    return np.concatenate(probs) if probs else np.array([], dtype=np.float64)


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    X_clean = np.asarray(X, dtype=np.float64)
    if hasattr(model, "_radar_impute_fill"):
        X_clean = impute_features(X_clean, model._radar_impute_fill)
    if hasattr(model, "_radar_scaler"):
        X_clean = model._radar_scaler.transform(X_clean)

    if isinstance(model, LogisticRegression):
        probs = _logistic_predict_proba(model, X_clean)
        return np.clip(probs, 0.0, 1.0)

    feature_names = getattr(model, "_radar_feature_names", None)
    if feature_names is not None:
        model_input: np.ndarray | pd.DataFrame = _to_model_frame(X_clean, feature_names)
    elif hasattr(model, "feature_names_in_"):
        model_input = _to_model_frame(X_clean, list(model.feature_names_in_))
    else:
        model_input = X_clean

    probs = model.predict_proba(model_input)[:, 1]
    return np.clip(probs, 0.0, 1.0)
