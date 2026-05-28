from __future__ import annotations

import pandas as pd

from radar.config.schemas import LabelsConfig
from radar.features.labels import add_labels


def test_direction_label_threshold():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "close": [100.0, 100.05, 100.0],
    })
    config = LabelsConfig(direction_min_move_pct=0.001)
    labeled = add_labels(df, config)
    # 0.05% move is below 0.1% threshold -> NaN
    assert pd.isna(labeled.loc[0, "y_direction"])
