from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from radar.config.settings import get_settings
from radar.forecast.market_hours import (
    NYSE_CASH_OPEN_BERLIN,
    _snap_to_berlin_premarket,
    berlin_calendar_date,
    berlin_time,
    is_cash_open_window,
    is_valid_trading_time,
    project_trading_timestamps,
    to_utc_iso,
)

# US regular session ≈ 6.5h × 12 five-minute bars/hour
_RTH_5M_BARS_PER_SESSION = 78


@dataclass
class IntradayForecastResult:
    points: list[dict[str, Any]]
    engine: str
    horizon_bars: int


def _historical_open_returns(frame: pd.DataFrame, *, post_open_bars: int = 6) -> np.ndarray:
    """Median 5m returns in the first ~30 minutes after 15:30 Berlin on prior sessions."""
    work = frame.dropna(subset=["close"]).copy()
    if len(work) < 20:
        return np.array([])

    work["date"] = pd.to_datetime(work["date"])
    work["_bt"] = work["date"].apply(berlin_time)
    work["_bd"] = work["date"].apply(berlin_calendar_date)

    returns: list[float] = []
    for _, day in work.groupby("_bd"):
        after_open = day[day["_bt"] >= NYSE_CASH_OPEN_BERLIN].head(post_open_bars + 1)
        if len(after_open) < 2:
            continue
        closes = after_open["close"].astype(float).values
        rets = np.diff(closes) / np.maximum(closes[:-1], 1e-6)
        returns.extend(float(r) for r in rets)

    return np.array(returns, dtype=float)


def _recent_shock_momentum(returns: np.ndarray) -> float:
    """Propagate large last-bar moves (e.g. afternoon selloff) into the next few bars."""
    if len(returns) < 4:
        return 0.0
    baseline_vol = float(np.std(returns[:-2])) if len(returns) > 3 else float(np.std(returns))
    if baseline_vol < 1e-6:
        baseline_vol = 0.001
    last = float(returns[-1])
    threshold = max(0.006, baseline_vol * 2.2)
    if abs(last) > threshold:
        return float(np.clip(last * 0.65, -0.025, 0.025))
    return 0.0


def _rth_recent_returns(frame: pd.DataFrame, n_bars: int = 8) -> np.ndarray:
    """Recent returns from regular-session bars only (avoids flat pre-market momentum)."""
    work = frame.dropna(subset=["close"]).copy()
    work["date"] = pd.to_datetime(work["date"])
    rth = work[work["date"].apply(lambda d: berlin_time(d) >= NYSE_CASH_OPEN_BERLIN)]
    if len(rth) < 3:
        closes = work["close"].astype(float).values
        if len(closes) < 2:
            return np.array([])
        return np.diff(closes) / np.maximum(closes[:-1], 1e-6)

    closes = rth["close"].astype(float).values
    segment = closes[-(n_bars + 1) :]
    return np.diff(segment) / np.maximum(segment[:-1], 1e-6)


def _session_target_return(
    daily_return_target: float,
    p_up: float,
    horizon: int,
    interval: str,
) -> float:
    bars_per_session = _RTH_5M_BARS_PER_SESSION if interval == "5m" else max(1, _RTH_5M_BARS_PER_SESSION // 12)
    session_frac = min(1.0, horizon / bars_per_session)
    target = float(daily_return_target) * session_frac
    if p_up >= 0.55:
        target = max(target, 0.002 * session_frac)
    elif p_up <= 0.45:
        target = min(target, -0.002 * session_frac)
    return target


def _drift_path_prices(last_close: float, horizon: int, target_total: float) -> np.ndarray:
    """Linear path from last_close toward session-scaled ensemble/baseline return."""
    if horizon <= 0:
        return np.array([], dtype=float)
    prices: list[float] = []
    for i in range(horizon):
        frac = (i + 1) / horizon
        prices.append(last_close * (1.0 + target_total * frac))
    return np.array(prices, dtype=float)


def _blend_forecast_paths(
    baseline: np.ndarray,
    drift: np.ndarray,
    blend: float,
) -> np.ndarray:
    if len(baseline) == 0:
        return drift
    if len(drift) == 0:
        return baseline
    n = min(len(baseline), len(drift))
    weight = float(np.clip(blend, 0.0, 1.0))
    return weight * drift[:n] + (1.0 - weight) * baseline[:n]


def _forecast_baseline_bars(
    close: np.ndarray,
    horizon: int,
    *,
    future_times: Optional[pd.DatetimeIndex] = None,
    historical_frame: Optional[pd.DataFrame] = None,
    daily_return_target: Optional[float] = None,
    interval: str = "5m",
) -> np.ndarray:
    """
    Bar-level forecast with session-aware open volatility.

    Pre-market bars often have near-zero returns; without an open regime the path
    stays flat through 15:30 Berlin. We boost caps/momentum in the cash-open window
    using historical open-bar stats and optionally tilt toward the daily forecast.
    """
    if len(close) < 10:
        last = float(close[-1])
        return np.full(horizon, last, dtype=float)

    returns = np.diff(close) / np.maximum(close[:-1], 1e-6)
    recent = returns[-min(40, len(returns)) :]

    if historical_frame is not None and not historical_frame.empty:
        rth_recent = _rth_recent_returns(historical_frame, n_bars=8)
        if len(rth_recent) >= 3:
            momentum = float(np.mean(rth_recent[-8:]))
            vol = float(np.std(rth_recent)) if len(rth_recent) > 1 else 0.001
        else:
            momentum = float(np.mean(recent[-8:])) if len(recent) >= 8 else float(np.mean(recent))
            vol = float(np.std(recent)) if len(recent) > 1 else 0.001
    else:
        momentum = float(np.mean(recent[-8:])) if len(recent) >= 8 else float(np.mean(recent))
        vol = float(np.std(recent)) if len(recent) > 1 else 0.001

    cap = max(0.002, min(0.012, vol * 2.5))
    shock = _recent_shock_momentum(recent)

    open_returns = (
        _historical_open_returns(historical_frame)
        if historical_frame is not None
        else np.array([])
    )
    if len(open_returns) > 0:
        # Opening bar is usually the largest move; median pre-market returns are ~0.
        peak_idx = int(np.argmax(np.abs(open_returns)))
        open_momentum = float(open_returns[peak_idx])
        open_typical = float(np.percentile(np.abs(open_returns), 75))
        open_vol = float(np.std(open_returns)) if len(open_returns) > 1 else open_typical
        open_cap = max(0.008, min(0.035, max(open_vol * 2.5, open_typical * 1.5)))
    else:
        open_momentum = momentum
        open_typical = max(0.006, abs(momentum))
        open_cap = max(cap, 0.01)

    prices: list[float] = []
    step_returns: list[float] = []
    price = float(close[-1])

    for i in range(horizon):
        in_open = (
            future_times is not None
            and i < len(future_times)
            and is_cash_open_window(future_times[i])
        )
        pattern = float(recent[i % len(recent)])
        if in_open:
            # Cash open: follow historical open-bar behavior, not flat pre-market drift.
            blended = float(np.clip(open_momentum, -open_cap, open_cap))
            if abs(blended) < open_typical * 0.5:
                sign = float(np.sign(open_momentum or momentum or 1.0))
                blended = sign * min(open_cap, max(open_typical, 0.005))
            cap_use = open_cap
            momentum_use = open_momentum
            price *= 1.0 + blended
            prices.append(price)
            step_returns.append(blended)
            continue
        else:
            momentum_use = momentum
            cap_use = cap

        shock_term = shock if i < 3 else 0.0
        blended = 0.55 * pattern + 0.45 * momentum_use + shock_term
        blended = float(np.clip(blended, -cap_use, cap_use))
        price *= 1.0 + blended
        prices.append(price)
        step_returns.append(blended)

    if daily_return_target is not None and horizon > 0:
        bars_per_session = _RTH_5M_BARS_PER_SESSION if interval == "5m" else max(1, _RTH_5M_BARS_PER_SESSION // 12)
        session_frac = min(1.0, horizon / bars_per_session)
        target_total = float(daily_return_target) * session_frac
        start = float(close[-1])
        current_total = prices[-1] / start - 1.0
        gap = target_total - current_total
        # Partial blend so daily signal tilts the path without overriding open dynamics.
        adjust = gap * 0.45
        non_open_idx = [
            i
            for i in range(horizon)
            if not (
                future_times is not None
                and i < len(future_times)
                and is_cash_open_window(future_times[i])
            )
        ]
        per_bar = adjust / max(len(non_open_idx), 1)
        price = start
        prices = []
        for i in range(horizon):
            blended = step_returns[i]
            if i in non_open_idx:
                blended += per_bar
            price *= 1.0 + blended
            prices.append(price)

    return np.array(prices, dtype=float)


def forecast_intraday_series(
    frame: pd.DataFrame,
    interval: str,
    config_dir: str = "config",
    daily_return_target: Optional[float] = None,
    p_up: Optional[float] = None,
    horizon_bars_override: Optional[int] = None,
    future_dates_override: Optional[pd.DatetimeIndex] = None,
) -> IntradayForecastResult:
    """Forecast next intraday bars (baseline + ensemble daily drift blend)."""
    interval = interval.lower()
    if interval not in ("5m", "1h"):
        raise ValueError(f"Unsupported interval '{interval}'")

    settings = get_settings(config_dir)
    fc = settings.forecast
    if interval == "5m":
        context_bars = fc.intraday_context_bars_5m
        horizon_bars = horizon_bars_override or fc.intraday_horizon_bars_5m
    else:
        context_bars = fc.intraday_context_bars_1h
        horizon_bars = horizon_bars_override or fc.intraday_horizon_bars_1h

    if frame.empty or len(frame) < 20:
        return IntradayForecastResult(points=[], engine="none", horizon_bars=0)

    work = frame.dropna(subset=["close"]).copy()
    work["date"] = pd.to_datetime(work["date"])
    closes = work["close"].astype(float).values
    context = closes[-min(context_bars, len(closes)) :]
    last_close = float(closes[-1])
    last_ts = work["date"].iloc[-1]
    if not is_valid_trading_time(last_ts):
        last_ts = _snap_to_berlin_premarket(last_ts)

    prob_up = 0.5 if p_up is None else float(p_up)
    if future_dates_override is not None and len(future_dates_override) > 0:
        future_dates = pd.DatetimeIndex(future_dates_override[:horizon_bars])
        horizon_bars = len(future_dates)
    else:
        future_dates = project_trading_timestamps(last_ts, interval, horizon_bars)
    baseline_values = _forecast_baseline_bars(
        context,
        horizon_bars,
        future_times=future_dates,
        historical_frame=work,
        daily_return_target=None,
        interval=interval,
    )

    engine = "baseline_bars"
    if daily_return_target is not None:
        target_total = _session_target_return(
            daily_return_target, prob_up, horizon_bars, interval
        )
        drift_values = _drift_path_prices(last_close, horizon_bars, target_total)
        forecast_values = _blend_forecast_paths(
            baseline_values,
            drift_values,
            fc.intraday_ensemble_blend,
        )
        engine = "hybrid_bars"
    else:
        forecast_values = baseline_values

    points = [
        {
            "date": to_utc_iso(ts),
            "close": round(float(price), 4),
        }
        for ts, price in zip(future_dates, forecast_values)
    ]

    return IntradayForecastResult(
        points=points,
        engine=engine,
        horizon_bars=horizon_bars,
    )
