from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from radar.config.settings import get_settings
from radar.forecast.market_hours import (
    _snap_to_berlin_premarket,
    is_valid_trading_time,
    project_trading_timestamps,
    to_utc_iso,
)


@dataclass
class IntradayForecastResult:
    points: list[dict[str, Any]]
    engine: str
    horizon_bars: int


def _forecast_baseline_bars(close: np.ndarray, horizon: int) -> np.ndarray:
    """Bar-level forecast using damped recent return patterns."""
    if len(close) < 10:
        last = float(close[-1])
        return np.full(horizon, last, dtype=float)

    returns = np.diff(close) / np.maximum(close[:-1], 1e-6)
    recent = returns[-min(40, len(returns)) :]
    momentum = float(np.mean(recent[-8:])) if len(recent) >= 8 else float(np.mean(recent))
    vol = float(np.std(recent)) if len(recent) > 1 else 0.001
    cap = max(0.002, min(0.012, vol * 2.5))

    prices: list[float] = []
    price = float(close[-1])
    for i in range(horizon):
        pattern = float(recent[i % len(recent)])
        blended = 0.55 * pattern + 0.45 * momentum
        blended = float(np.clip(blended, -cap, cap))
        price *= 1.0 + blended
        prices.append(price)
    return np.array(prices, dtype=float)


def forecast_intraday_series(
    frame: pd.DataFrame,
    interval: str,
    config_dir: str = "config",
) -> IntradayForecastResult:
    """Forecast next intraday bars from recent price action."""
    interval = interval.lower()
    if interval not in ("5m", "1h"):
        raise ValueError(f"Unsupported interval '{interval}'")

    settings = get_settings(config_dir)
    fc = settings.forecast
    if interval == "5m":
        context_bars = fc.intraday_context_bars_5m
        horizon_bars = fc.intraday_horizon_bars_5m
    else:
        context_bars = fc.intraday_context_bars_1h
        horizon_bars = fc.intraday_horizon_bars_1h

    if frame.empty or len(frame) < 20:
        return IntradayForecastResult(points=[], engine="none", horizon_bars=0)

    work = frame.dropna(subset=["close"]).copy()
    work["date"] = pd.to_datetime(work["date"])
    closes = work["close"].astype(float).values
    context = closes[-min(context_bars, len(closes)) :]
    last_ts = work["date"].iloc[-1]
    if not is_valid_trading_time(last_ts):
        last_ts = _snap_to_berlin_premarket(last_ts)

    forecast_values = _forecast_baseline_bars(context, horizon_bars)
    future_dates = project_trading_timestamps(last_ts, interval, horizon_bars)
    points = [
        {
            "date": to_utc_iso(ts),
            "close": round(float(price), 4),
        }
        for ts, price in zip(future_dates, forecast_values)
    ]

    return IntradayForecastResult(
        points=points,
        engine="baseline_bars",
        horizon_bars=horizon_bars,
    )
