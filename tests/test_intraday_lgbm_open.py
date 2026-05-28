from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from radar.forecast.intraday_forecast import (
    _apply_cash_open_overlay,
    _cash_open_step_return,
    _open_regime_stats,
    forecast_intraday_series,
)
from radar.forecast.market_hours import is_cash_open_window, project_trading_timestamps


def _synthetic_premarket_then_open() -> pd.DataFrame:
    pre = pd.date_range("2026-05-28 08:00", periods=66, freq="5min")
    open_bar = pd.date_range("2026-05-28 13:30", periods=12, freq="5min")
    dates = pre.append(open_bar)
    price = 900.0
    closes: list[float] = [price] * 66
    for ret in [0.012, 0.004, -0.002, 0.003, 0.001, -0.001, 0.002, 0.0, 0.001, 0.002, 0.001, 0.001]:
        price *= 1.0 + ret
        closes.append(price)
    return pd.DataFrame({"date": dates, "close": closes, "symbol": "AMZN"})


def test_cash_open_prefers_model_step_at_session_open():
    stats = _open_regime_stats(np.array([100.0, 100.1, 100.2]), None)
    model_step = 0.0012
    step = _cash_open_step_return(stats, at_session_open=True, model_step=model_step)
    assert abs(step - model_step) < 0.003


def test_apply_cash_open_overlay_moves_at_1530():
    frame = _synthetic_premarket_then_open()
    pre = frame.iloc[:66]
    last_ts = pre["date"].iloc[-1]
    future = project_trading_timestamps(last_ts, "5m", 24)
    anchor = float(pre["close"].iloc[-1])
    flat = np.full(24, anchor * 1.001, dtype=float)

    adjusted = _apply_cash_open_overlay(
        flat,
        future,
        anchor,
        frame,
        pre["close"].astype(float).values,
        daily_return_target=-0.01,
        p_up=0.4,
    )

    open_steps: list[float] = []
    prev = anchor
    for ts, price in zip(future, adjusted):
        if is_cash_open_window(ts):
            open_steps.append(float(price / prev - 1.0))
        prev = float(price)

    assert open_steps
    assert max(abs(s) for s in open_steps) >= 0.0025


@patch("radar.forecast.intraday_forecast.load_bundle")
@patch("radar.forecast.intraday_forecast.predict_next_return", return_value=0.0001)
def test_lgbm_forecast_uses_open_at_cash_session(mock_mu, mock_bundle):
    mock_bundle.return_value = MagicMock()
    frame = _synthetic_premarket_then_open()
    pre = frame.iloc[:66]

    result = forecast_intraday_series(pre, "5m")

    assert "lgbm" in result.engine
    assert len(result.points) == 64

    anchor = float(pre["close"].iloc[-1])
    prev = anchor
    open_moves: list[float] = []
    for pt in result.points:
        ts = pd.Timestamp(str(pt["date"]).replace("Z", ""))
        if is_cash_open_window(ts):
            price = float(pt["close"])
            open_moves.append(price / prev - 1.0)
        prev = float(pt["close"])

    assert open_moves
    assert max(abs(m) for m in open_moves) >= 0.003
