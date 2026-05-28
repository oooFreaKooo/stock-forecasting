from __future__ import annotations

from typing import Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd


def build_lgbm_classifier(seed: int = 42, *, fast: bool = False) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=120 if fast else 500,
        learning_rate=0.05,
        num_leaves=31 if not fast else 15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )


def _as_frame(X: np.ndarray, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    names = list(feature_names) if feature_names is not None else [f"feature_{idx}" for idx in range(X.shape[1])]
    return pd.DataFrame(np.asarray(X, dtype=np.float64), columns=names)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    feature_names: Optional[Sequence[str]] = None,
    *,
    fast: bool = False,
) -> lgb.LGBMClassifier:
    model = build_lgbm_classifier(seed=seed, fast=fast)
    train_frame = _as_frame(X_train, feature_names)
    val_frame = _as_frame(X_val, feature_names or list(train_frame.columns))
    model.fit(
        train_frame,
        y_train,
        eval_set=[(val_frame, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20 if fast else 50, verbose=False)],
    )
    return model
