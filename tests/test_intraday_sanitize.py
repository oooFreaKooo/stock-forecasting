from __future__ import annotations

import pandas as pd

from radar.forecast.intraday_sanitize import sanitize_intraday_closes


def test_sanitize_removes_single_bar_spike():
    frame = pd.DataFrame({
        "date": pd.date_range("2026-05-26 20:50", periods=5, freq="5min"),
        "close": [265.0, 265.0, 279.29, 265.03, 265.0],
        "symbol": ["AMZN"] * 5,
    })
    cleaned = sanitize_intraday_closes(frame, "5m")
    assert cleaned["close"].iloc[2] == 265.015
    assert cleaned["close"].max() < 270


def test_sanitize_keeps_real_trend():
    frame = pd.DataFrame({
        "date": pd.date_range("2026-05-26 10:00", periods=4, freq="5min"),
        "close": [100.0, 101.0, 102.0, 103.0],
        "symbol": ["AAPL"] * 4,
    })
    cleaned = sanitize_intraday_closes(frame, "5m")
    pd.testing.assert_series_equal(cleaned["close"], frame["close"], check_names=False)
