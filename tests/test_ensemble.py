from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.config.schemas import EnsembleConfig
from radar.ensemble.multi_horizon import add_multi_horizon_labels, apply_agreement_filter
from radar.ensemble.meta_learner import build_meta_features


def test_multi_horizon_labels():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=30),
        "symbol": "AAPL",
        "close": 100 + np.arange(30) * 0.5,
    })
    out = add_multi_horizon_labels(df, horizons=[1, 5])
    assert "y_direction_1d" in out.columns
    assert "y_direction_5d" in out.columns


def test_build_meta_features():
    base = {
        "lightgbm": np.array([0.6, 0.7]),
        "logistic": np.array([0.55, 0.65]),
    }
    meta = build_meta_features(base)
    assert meta[0].shape == (2, 2)
    assert meta[1] == ["p_lightgbm", "p_logistic"]


def test_agreement_filter():
    config = EnsembleConfig(horizons=[1, 5], uncertainty_threshold=0.1)
    preds = pd.DataFrame({
        "p_ensemble": [0.8, 0.51],
        "y_direction_1d": [1, 1],
        "y_direction_5d": [1, 0],
    })
    out = apply_agreement_filter(preds, config)
    assert out["trade_allowed"].dtype == bool
    assert out["trade_allowed"].iloc[0]
    assert not out["trade_allowed"].iloc[1]
