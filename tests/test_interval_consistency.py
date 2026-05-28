from __future__ import annotations

import pandas as pd

from radar.forecast.chart_paths import (
    resample_chart_points_to_1h,
    resample_intraday_chart_to_1h,
)
from radar.forecast.market_hours import project_trading_timestamps_until


def test_resample_chart_points_to_1h_uses_last_close_in_hour():
    points = [
        {"date": "2026-05-28T14:55:00Z", "close": 270.0},
        {"date": "2026-05-28T15:00:00Z", "close": 271.0},
        {"date": "2026-05-28T15:05:00Z", "close": 272.0},
        {"date": "2026-05-28T15:55:00Z", "close": 273.0},
    ]
    out = resample_chart_points_to_1h(points)
    assert len(out) == 2
    assert out[0]["close"] == 270.0  # 14:xx hour → last bar 14:55
    assert out[1]["close"] == 273.0  # 15:xx hour → last bar 15:55


def test_resample_intraday_chart_to_1h_uses_extended_history():
    chart_5m = {
        "symbol": "AAPL",
        "interval": "5m",
        "points": [{"date": "2026-05-28T15:00:00Z", "close": 100.0}],
        "model": {"engine": "intraday_lgbm", "points": [], "backtest_bars": 0, "forward_bars": 0},
        "forecast": {"engine": "intraday_lgbm", "horizon_bars": 1, "points": [{"date": "2026-05-28T15:05:00Z", "close": 101.0}]},
        "validation": {"engine": "intraday_lgbm", "points": [], "metrics": {}},
        "meta": {"source": "yfinance", "rows": 1, "note": "5m"},
    }
    long_history = [
        {"date": "2026-04-01T14:00:00Z", "close": 90.0},
        {"date": "2026-05-28T15:00:00Z", "close": 100.0},
    ]
    out = resample_intraday_chart_to_1h(chart_5m, history_points=long_history)
    assert out["interval"] == "1h"
    assert len(out["points"]) == 2
    assert out["forecast"]["engine"] == "intraday_lgbm"
    assert "30d" in out["meta"]["note"] or "5M" in out["meta"]["note"]


def test_project_trading_timestamps_until_covers_next_day():
    last = pd.Timestamp("2026-05-28T15:30:00")
    end = pd.Timestamp("2026-05-29T23:00:00")
    sched = project_trading_timestamps_until(last, "5m", end)
    assert len(sched) > 64
    assert sched[-1] <= end
