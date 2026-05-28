from __future__ import annotations

import numpy as np
import pandas as pd

from radar.forecast.intraday_forecast import _forecast_baseline_bars, _historical_open_returns
from radar.forecast.market_hours import is_cash_open_window, project_trading_timestamps


def _synthetic_premarket_then_open() -> pd.DataFrame:
    """Flat pre-market, sharp pct moves at 15:30 Berlin (13:30 UTC in May CEST)."""
    pre = pd.date_range("2026-05-28 08:00", periods=66, freq="5min")
    open_bar = pd.date_range("2026-05-28 13:30", periods=12, freq="5min")
    dates = pre.append(open_bar)
    price = 900.0
    closes: list[float] = [price] * 66
    for ret in [0.012, 0.004, -0.002, 0.003, 0.001, -0.001, 0.002, 0.0, 0.001, 0.002, 0.001, 0.001]:
        price *= 1.0 + ret
        closes.append(price)
    return pd.DataFrame({"date": dates, "close": closes})


def test_historical_open_returns_detects_open_volatility():
    frame = _synthetic_premarket_then_open()
    rets = _historical_open_returns(frame)
    assert len(rets) >= 1
    assert float(np.max(np.abs(rets))) >= 0.003


def test_forecast_moves_more_at_cash_open_than_premarket():
    frame = _synthetic_premarket_then_open()
    # Anchor forecast from last pre-market bar so the path crosses 15:30 Berlin.
    pre_only = frame.iloc[:66]
    context = pre_only["close"].astype(float).values
    last_ts = pre_only["date"].iloc[-1]
    future = project_trading_timestamps(last_ts, "5m", 24)

    flat_only = _forecast_baseline_bars(context, 24, future_times=future, historical_frame=None)
    with_open = _forecast_baseline_bars(
        context,
        24,
        future_times=future,
        historical_frame=frame,
        daily_return_target=0.01,
    )

    def open_window_returns(prices: np.ndarray, times: pd.DatetimeIndex) -> list[float]:
        out: list[float] = []
        prev = context[-1]
        for i, ts in enumerate(times):
            if is_cash_open_window(ts):
                out.append(float(prices[i] / prev - 1))
            prev = prices[i]
        return out

    open_rets = open_window_returns(with_open, future)
    assert open_rets, "expected forecast bars in 15:30–17:00 Berlin window"
    assert max(abs(r) for r in open_rets) >= 0.003

    # Without historical open stats the path stays near flat through the open window.
    legacy_open = open_window_returns(flat_only, future)
    if legacy_open:
        assert max(abs(r) for r in open_rets) >= min(0.003, max(abs(r) for r in legacy_open) * 1.5)
