import numpy as np
import pandas as pd
import pytest

from radar.forecast.alphavantage_forecast import (
    _intraday_eod_markers,
    build_alphavantage_comparison,
)


def test_intraday_eod_markers_raw_av_only():
    history = [
        {"date": "2026-05-26T18:00:00Z", "close": 999.0},
        {"date": "2026-05-27T18:00:00Z", "close": 888.0},
    ]
    av = pd.Series(
        [410.0, 405.0],
        index=pd.to_datetime(["2026-05-26", "2026-05-27"]),
    )
    pts = _intraday_eod_markers(history, av)
    assert len(pts) == 2
    assert pts[0]["close"] == 410.0
    assert pts[1]["close"] == 405.0
    assert pts[0]["close"] != history[0]["close"]


def test_intraday_comparison_is_markers_not_line(monkeypatch):
    dates = pd.bdate_range("2026-05-20", periods=8)
    av = pd.Series(np.linspace(400.0, 420.0, len(dates)), index=dates)
    history = [
        {"date": d.strftime("%Y-%m-%dT18:00:00Z"), "close": 500.0}
        for d in dates[-3:]
    ]
    future = pd.date_range("2026-05-28T19:00:00Z", periods=5, freq="5min")

    monkeypatch.setattr("radar.forecast.alphavantage_forecast.is_configured", lambda: True)

    out = build_alphavantage_comparison(
        "MSFT",
        interval="5m",
        anchor_price=500.0,
        anchor_ts=pd.Timestamp(history[0]["date"]),
        future_dates=future,
        daily_closes=av,
        history_points=history,
    )
    assert out is not None
    assert out["display"] == "markers"
    assert out["forward_bars"] == 0
    assert len(out["points"]) <= 3
    assert all(p["close"] < 450 for p in out["points"])
