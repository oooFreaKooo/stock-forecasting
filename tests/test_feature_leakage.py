from __future__ import annotations

import numpy as np
import pandas as pd

from radar.features.leakage import shift_features


def test_shift_features_prevents_same_bar_values():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "rsi": [10, 20, 30, 40, 50],
        "y_direction": [0, 1, 0, 1, 0],
    })
    shifted = shift_features(df, ["rsi"], periods=1)
    assert pd.isna(shifted["rsi"].iloc[0])
    assert shifted["rsi"].iloc[1] == 10
    assert shifted["rsi"].iloc[4] == 40


def test_labels_use_future_returns():
    from radar.config.schemas import LabelsConfig
    from radar.features.labels import add_labels

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=4),
        "close": [100.0, 101.0, 99.0, 102.0],
    })
    labeled = add_labels(df, LabelsConfig(direction_min_move_pct=0.001))
    # Day 0: next return positive -> y_direction=1
    assert labeled.loc[0, "y_direction"] == 1
    # Last row has no next day
    assert pd.isna(labeled.loc[3, "next_return"])
