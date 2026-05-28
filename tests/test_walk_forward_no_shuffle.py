from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from radar.config.schemas import WalkForwardConfig
from radar.validation.splits import generate_splits


def test_generate_splits_anchored_expanding():
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="B")
    config = WalkForwardConfig(
        min_train_days=504,
        test_window="monthly",
        step="monthly",
        purge_days=1,
    )
    splits = generate_splits(dates, config, data_start=date(2018, 1, 1))

    assert len(splits) > 0
    # Train always starts at data_start
    for split in splits:
        assert split.train_start == date(2018, 1, 1)
        assert split.train_end < split.test_start
        assert split.test_start <= split.test_end


def test_splits_never_shuffle():
    """Splits must be strictly chronological with no overlap in test windows."""
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    config = WalkForwardConfig(min_train_days=252, test_window="monthly", step="monthly")
    splits = generate_splits(dates, config)

    for i in range(1, len(splits)):
        prev = splits[i - 1]
        curr = splits[i]
        assert curr.train_end >= prev.train_end
        assert curr.test_start > prev.test_start
