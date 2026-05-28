from __future__ import annotations

import numpy as np
import pandas as pd

from radar.api.chart_validation import build_daily_validation, build_intraday_validation


def test_daily_validation_no_lookahead():
    dates = pd.bdate_range("2024-01-01", periods=80)
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, len(dates))), index=dates)

    val_points, metrics = build_daily_validation(close, validation_days=20, context_days=30)
    assert len(val_points) == 19
    assert metrics["n_points"] == 19
    assert metrics["mae"] is not None
    assert 0.0 <= metrics["direction_accuracy"] <= 1.0


def test_daily_validation_spans_display_window():
    dates = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(2)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.3, len(dates))), index=dates)

    val_points, metrics = build_daily_validation(
        close,
        validation_days=None,
        context_days=60,
        validation_context_days=30,
    )
    assert metrics["n_points"] >= 160
    assert val_points[0]["date"] < val_points[-1]["date"]


def test_intraday_validation_covers_most_of_history():
    dates = pd.date_range("2026-05-20 13:30", periods=120, freq="5min")
    rng = np.random.default_rng(1)
    close = 200 + np.cumsum(rng.normal(0, 0.15, len(dates)))
    frame = pd.DataFrame({"date": dates, "close": close})

    val_points, metrics = build_intraday_validation(frame, "5m", symbol="TEST")
    assert metrics["n_points"] > 0
    assert len(val_points) == metrics["n_points"]
    assert metrics["coverage_pct"] >= 0.75
    assert metrics.get("segments", 0) >= 2
