from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from radar.config.settings import Settings
from radar.models.supervised.calibration import calibrate_probabilities
from radar.models.supervised.lightgbm_model import train_classifier
from radar.validation.metrics import compute_classification_metrics


def _time_based_val_split(
    df: pd.DataFrame,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date")
    n = len(df)
    split_idx = int(n * (1 - val_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    settings: Settings,
    fold_id: int,
) -> dict[str, Any]:
    """Train one walk-forward fold and produce OOS predictions."""
    inner_train, inner_val = _time_based_val_split(train_df, settings.model.val_fraction)

    X_train = inner_train[feature_cols].values
    y_train = inner_train["y_direction"].values.astype(int)
    X_val = inner_val[feature_cols].values
    y_val = inner_val["y_direction"].values.astype(int)

    model = train_classifier(
        X_train, y_train, X_val, y_val,
        seed=settings.model.random_seed,
        feature_names=feature_cols,
    )

    val_frame = pd.DataFrame(X_val, columns=feature_cols)
    val_probs = model.predict_proba(val_frame)[:, 1]
    cal_probs, calibrator = calibrate_probabilities(val_probs, y_val)

    X_test = test_df[feature_cols].values
    y_test = test_df["y_direction"].values.astype(int)
    test_frame = pd.DataFrame(X_test, columns=feature_cols)
    raw_test_probs = model.predict_proba(test_frame)[:, 1]
    test_probs = calibrator.transform(raw_test_probs)

    metrics = compute_classification_metrics(y_test, test_probs)

    predictions = test_df[["date", "symbol", "close", "next_return", "y_direction"]].copy()
    predictions["p_up"] = test_probs
    predictions["p_down"] = 1.0 - test_probs
    predictions["y_pred"] = (test_probs >= settings.model.direction_threshold).astype(int)

    return {
        "model": model,
        "calibrator": calibrator,
        "metrics": metrics,
        "predictions": predictions,
        "fold_id": fold_id,
    }
