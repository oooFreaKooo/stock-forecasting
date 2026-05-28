from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    prices: np.ndarray
    engine: str
    horizon_days: int


def forecast_baseline(
    close: pd.Series,
    horizon_days: int = 5,
    context_days: int = 120,
) -> ForecastResult:
    """
    Statistical forecast using recent return patterns + mean reversion.

    Unlike a fixed daily drift (which draws a straight line), this replays
    damped recent daily moves so the path can rise, fall, and flatten.
    """
    series = close.dropna().astype(float)
    if len(series) < 20:
        raise ValueError("Need at least 20 price points for baseline forecast")

    history = series.iloc[-context_days:]
    last_date = history.index[-1]
    last_price = float(history.iloc[-1])

    returns = history.pct_change().dropna()
    if returns.empty:
        returns = pd.Series([0.0])

    ema_fast = history.ewm(span=10, adjust=False).mean().iloc[-1]
    ema_slow = history.ewm(span=30, adjust=False).mean().iloc[-1]
    long_ema = history.ewm(span=60, adjust=False).mean().iloc[-1]
    momentum = float((ema_fast - ema_slow) / max(last_price, 1e-6))
    momentum = float(np.clip(momentum, -0.025, 0.025))

    recent = returns.iloc[-min(20, len(returns)) :].values
    vol = float(returns.std())
    if np.isnan(vol) or vol <= 0:
        vol = 0.012

    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon_days)
    prices: list[float] = []
    price = last_price

    for i in range(horizon_days):
        pattern_ret = float(recent[i % len(recent)])
        reversion = 0.2 * (long_ema - price) / max(price, 1e-6)
        blended = 0.5 * pattern_ret + 0.3 * momentum + 0.2 * reversion
        blended = float(np.clip(blended, -3.5 * vol, 3.5 * vol))
        price = price * (1.0 + blended)
        prices.append(price)

    return ForecastResult(
        dates=future_dates,
        prices=np.array(prices, dtype=float),
        engine="baseline",
        horizon_days=horizon_days,
    )


def forecast_return_1d(result: ForecastResult, last_price: float) -> float:
    if len(result.prices) == 0:
        return 0.0
    return float(result.prices[0] / last_price - 1.0)
