from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from radar.config.settings import get_settings
from radar.intraday.features import build_intraday_feature_frame
from radar.intraday.model import load_bundle, predict_next_return
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
class _OpenRegimeStats:
    open_momentum: float
    open_typical: float
    open_cap: float
    session_momentum: float
    session_vol: float


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


def _recency_weighted_mean(values: np.ndarray, tail: int = 12) -> float:
    segment = values[-min(tail, len(values)) :]
    if len(segment) == 0:
        return 0.0
    weights = np.exp(np.linspace(-1.5, 0.0, len(segment)))
    weights /= weights.sum()
    return float(np.average(segment, weights=weights))


def _reanchor_price_path(prices: np.ndarray, anchor_price: float) -> np.ndarray:
    """Scale path so the first point matches anchor_price (preserves shape)."""
    if len(prices) == 0:
        return prices
    base = float(prices[0])
    if abs(base) < 1e-6:
        return prices
    return np.asarray(prices, dtype=float) * (float(anchor_price) / base)


def _session_momentum_vol(
    context: np.ndarray,
    historical_frame: Optional[pd.DataFrame],
) -> tuple[float, float]:
    returns = np.diff(context) / np.maximum(context[:-1], 1e-6)
    recent = returns[-min(40, len(returns)) :]
    if historical_frame is not None and not historical_frame.empty:
        rth_recent = _rth_recent_returns(historical_frame, n_bars=12)
        if len(rth_recent) >= 3:
            momentum = _recency_weighted_mean(rth_recent, tail=12)
            vol = float(np.std(rth_recent)) if len(rth_recent) > 1 else 0.001
            return momentum, vol
    momentum = _recency_weighted_mean(recent, tail=12)
    vol = float(np.std(recent)) if len(recent) > 1 else 0.001
    return momentum, vol


def _open_regime_stats(
    context: np.ndarray,
    historical_frame: Optional[pd.DataFrame],
) -> _OpenRegimeStats:
    momentum, vol = _session_momentum_vol(context, historical_frame)
    cap = max(0.002, min(0.012, vol * 2.5))
    open_returns = (
        _historical_open_returns(historical_frame)
        if historical_frame is not None
        else np.array([])
    )
    if len(open_returns) > 0:
        peak_idx = int(np.argmax(np.abs(open_returns)))
        open_momentum = float(open_returns[peak_idx])
        open_typical = float(np.percentile(np.abs(open_returns), 75))
        open_vol = float(np.std(open_returns)) if len(open_returns) > 1 else open_typical
        open_cap = max(0.008, min(0.035, max(open_vol * 2.5, open_typical * 1.5)))
    else:
        open_momentum = momentum
        open_typical = max(0.006, abs(momentum))
        open_cap = max(cap, 0.01)
    return _OpenRegimeStats(
        open_momentum=open_momentum,
        open_typical=open_typical,
        open_cap=open_cap,
        session_momentum=momentum,
        session_vol=vol,
    )


def _daily_direction_sign(
    daily_return_target: Optional[float],
    p_up: float,
) -> float:
    if daily_return_target is not None and abs(daily_return_target) > 1e-8:
        return float(np.sign(daily_return_target))
    if p_up >= 0.55:
        return 1.0
    if p_up <= 0.45:
        return -1.0
    return 1.0


def _is_entering_cash_open(i: int, future_dates: pd.DatetimeIndex) -> bool:
    if i >= len(future_dates) or not is_cash_open_window(future_dates[i]):
        return False
    if i == 0:
        return True
    return not is_cash_open_window(future_dates[i - 1])


def _cash_open_step_return(
    stats: _OpenRegimeStats,
    *,
    at_session_open: bool,
    model_step: float = 0.0,
    daily_return_target: Optional[float] = None,
    p_up: float = 0.5,
) -> float:
    """Step return for bars in the US cash-open window (15:30–17:00 Berlin)."""
    hist_open = float(np.clip(stats.open_momentum, -stats.open_cap, stats.open_cap))

    # When the ML model produced a step, prefer it (accuracy at 15:30 matters more than forced drama).
    if abs(model_step) > 1e-6:
        hist_weight = 0.2 if at_session_open else 0.15
        blended = (1.0 - hist_weight) * model_step + hist_weight * hist_open
        if daily_return_target is not None:
            direction = _daily_direction_sign(daily_return_target, p_up)
            if np.sign(blended) != direction and abs(blended) < abs(hist_open):
                blended = direction * max(abs(blended), abs(hist_open) * 0.5)
        return float(np.clip(blended, -stats.open_cap, stats.open_cap))

    direction = _daily_direction_sign(daily_return_target, p_up)
    if at_session_open:
        blended = hist_open
        sign = float(np.sign(stats.open_momentum or stats.session_momentum or direction))
        if abs(blended) < stats.open_typical * 0.35:
            blended = sign * min(stats.open_cap, max(stats.open_typical * 0.5, 0.002))
        if daily_return_target is not None and np.sign(blended) != direction:
            blended = direction * max(abs(blended), stats.open_typical * 0.35)
        return float(np.clip(blended, -stats.open_cap, stats.open_cap))

    open_step = float(np.clip(stats.open_momentum * 0.65, -stats.open_cap, stats.open_cap))
    return float(np.clip(open_step, -stats.open_cap, stats.open_cap))


def _rebuild_path_from_steps(anchor_price: float, steps: list[float]) -> np.ndarray:
    prices: list[float] = []
    prev = float(anchor_price)
    for step in steps:
        prev *= 1.0 + float(step)
        prices.append(prev)
    return np.array(prices, dtype=float)


def _apply_cash_open_overlay(
    prices: np.ndarray,
    future_dates: pd.DatetimeIndex,
    anchor_price: float,
    historical_frame: pd.DataFrame,
    context: np.ndarray,
    *,
    daily_return_target: Optional[float] = None,
    p_up: float = 0.5,
) -> np.ndarray:
    """Ensure projected paths move at 15:30 Berlin, not only after the open window."""
    if len(prices) == 0 or len(future_dates) == 0:
        return prices

    stats = _open_regime_stats(context, historical_frame)
    steps: list[float] = []
    prev = float(anchor_price)
    for price in prices:
        steps.append(float(price / prev - 1.0) if prev else 0.0)
        prev = float(price)

    for i in range(len(steps)):
        if i >= len(future_dates):
            break
        if not is_cash_open_window(future_dates[i]):
            continue
        model_step = steps[i]
        steps[i] = _cash_open_step_return(
            stats,
            at_session_open=_is_entering_cash_open(i, future_dates),
            model_step=model_step,
            daily_return_target=daily_return_target,
            p_up=p_up,
        )

    return _rebuild_path_from_steps(anchor_price, steps)


def _scale_path_to_total_return(
    prices: np.ndarray,
    anchor_price: float,
    target_total_return: float,
    future_dates: Optional[pd.DatetimeIndex] = None,
) -> np.ndarray:
    """
    Rescale step returns so final price matches anchor*(1+target_total_return).

    This keeps the *shape* (up/down pattern) but prevents interval-specific
    compounding from producing wildly different totals (e.g. 5m vs 1h).
    """
    if len(prices) == 0:
        return prices
    anchor = float(anchor_price)
    if anchor <= 0:
        return prices

    current_last = float(prices[-1])
    if current_last <= 0:
        return prices

    # Extract original step returns (preserve up/down pattern).
    orig_prev = anchor
    steps: list[float] = []
    for p in prices:
        step = float(p / orig_prev - 1.0) if orig_prev else 0.0
        steps.append(step)
        orig_prev = float(p)

    desired_total = float(target_total_return)
    if abs(desired_total) < 1e-10:
        # Keep volatility but remove net drift.
        desired_total = 0.0

    # Find a constant drift adjustment per step so product matches desired_total,
    # while keeping per-step volatility shape intact.
    target_log = float(np.log1p(desired_total))

    def total_log(add: float) -> float:
        s = 0.0
        for r in steps:
            rr = float(np.clip(r + add, -0.15, 0.15))
            s += float(np.log1p(rr))
        return s

    # Binary search drift add in a conservative range.
    lo, hi = -0.02, 0.02
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if total_log(mid) < target_log:
            lo = mid
        else:
            hi = mid
    add = (lo + hi) / 2.0

    out: list[float] = []
    new_prev = anchor
    for i, r in enumerate(steps):
        if future_dates is not None and i < len(future_dates) and _is_entering_cash_open(i, future_dates):
            rr = float(np.clip(r, -0.15, 0.15))
        elif future_dates is not None and i < len(future_dates) and is_cash_open_window(future_dates[i]):
            rr = float(np.clip(r + add * 0.2, -0.15, 0.15))
        else:
            rr = float(np.clip(r + add, -0.15, 0.15))
        new_prev = new_prev * (1.0 + rr)
        out.append(new_prev)

    return np.array(out, dtype=float)


def _boost_path_volatility(
    prices: np.ndarray,
    anchor_price: float,
    target_step_vol: float,
    *,
    max_boost: float = 6.0,
) -> np.ndarray:
    """
    Increase forecast step volatility without changing overall level.

    Intraday ML models often regress toward 0 for next-bar returns; this boosts the
    per-step return std toward the recent realized volatility, preserving the
    up/down sign pattern.
    """
    if len(prices) < 3:
        return prices
    anchor = float(anchor_price)
    if anchor <= 0:
        return prices

    orig_prev = anchor
    steps: list[float] = []
    for p in prices:
        step = float(p / orig_prev - 1.0) if orig_prev else 0.0
        steps.append(step)
        orig_prev = float(p)

    step_std = float(np.std(steps))
    if not np.isfinite(step_std) or step_std < 1e-8:
        return prices

    target = float(target_step_vol)
    if not np.isfinite(target) or target <= 0:
        return prices

    boost = float(np.clip(target / step_std, 1.0, max_boost))
    if boost <= 1.001:
        return prices

    out: list[float] = []
    prev = anchor
    for r in steps:
        rr = float(np.clip(r * boost, -0.15, 0.15))
        prev = prev * (1.0 + rr)
        out.append(prev)
    return np.array(out, dtype=float)


def _blend_lgbm_with_session_shape(
    lgbm_prices: np.ndarray,
    anchor_price: float,
    context: np.ndarray,
    work: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    interval: str,
    *,
    shape_weight: float = 0.55,
    daily_return_target: Optional[float] = None,
) -> np.ndarray:
    """
    LGBM next-bar returns are often near-flat; blend in session-shaped baseline
    moves so the forward segment looks like a market path, keeping LGBM terminal.
    """
    horizon = len(lgbm_prices)
    if horizon < 2:
        return lgbm_prices

    shaped = _forecast_baseline_bars(
        context,
        horizon,
        future_times=future_dates,
        historical_frame=work,
        daily_return_target=daily_return_target,
        interval=interval,
    )
    if len(shaped) != horizon:
        return lgbm_prices

    anchor = float(anchor_price)
    lgbm_path = np.r_[anchor, lgbm_prices]
    shape_path = np.r_[anchor, shaped]
    lgbm_steps = np.diff(lgbm_path) / np.maximum(lgbm_path[:-1], 1e-6)
    shape_steps = np.diff(shape_path) / np.maximum(shape_path[:-1], 1e-6)
    blended_steps = (1.0 - shape_weight) * lgbm_steps + shape_weight * shape_steps
    out = _rebuild_path_from_steps(anchor, blended_steps.tolist())
    terminal = float(lgbm_prices[-1])
    if out[-1] > 0 and terminal > 0:
        out = out * (terminal / float(out[-1]))
    return out


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
    stats = _open_regime_stats(close, historical_frame)
    momentum = stats.session_momentum
    vol = stats.session_vol
    cap = max(0.002, min(0.012, vol * 2.5))
    shock = _recent_shock_momentum(recent)
    open_momentum = stats.open_momentum
    open_typical = stats.open_typical
    open_cap = stats.open_cap

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
        if in_open and future_times is not None:
            blended = _cash_open_step_return(
                stats,
                at_session_open=_is_entering_cash_open(i, future_times),
                model_step=momentum,
                daily_return_target=daily_return_target,
                p_up=0.5,
            )
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
    horizon_bars_override: Optional[int] = None,
    future_dates_override: Optional[pd.DatetimeIndex] = None,
    *,
    daily_return_target: Optional[float] = None,
    p_up: Optional[float] = None,
) -> IntradayForecastResult:
    """Next-bar intraday path from the trained 5m LGBM model (fallback: session-aware baseline)."""
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

    if future_dates_override is not None and len(future_dates_override) > 0:
        future_dates = pd.DatetimeIndex(future_dates_override)
        if horizon_bars_override is not None:
            future_dates = future_dates[:horizon_bars_override]
        horizon_bars = len(future_dates)
    else:
        future_dates = project_trading_timestamps(last_ts, interval, horizon_bars)
    # Intraday trained model (5m) when available; fallback to baseline bars.
    baseline_values: np.ndarray
    engine = "baseline_bars"
    open_stats = _open_regime_stats(context, work)
    bundle = load_bundle(settings) if interval == "5m" else None
    if bundle is not None:
        # Iterative next-bar return predictions.
        sim = work[["date", "close"]].copy()
        price = float(sim["close"].iloc[-1])
        preds: list[float] = []
        symbol = str(frame["symbol"].iloc[-1]).upper() if "symbol" in frame.columns else ""
        for i in range(horizon_bars):
            next_ts = future_dates[i]
            model_step = 0.0
            feat = build_intraday_feature_frame(
                sim,
                symbol=symbol,
                horizon_bars=1,
                settings=settings,
                predict_for_ts=next_ts,
            )
            if feat is not None and not feat.X.empty:
                last_row = feat.X.tail(1).copy()
                mu = predict_next_return(bundle, last_row)
                model_step = float(np.clip(mu, -0.08, 0.08))

            if is_cash_open_window(next_ts):
                step = _cash_open_step_return(
                    open_stats,
                    at_session_open=_is_entering_cash_open(i, future_dates),
                    model_step=model_step,
                    daily_return_target=daily_return_target,
                    p_up=p_up if p_up is not None else 0.5,
                )
            else:
                step = model_step

            price *= 1.0 + step
            preds.append(price)
            sim = pd.concat(
                [
                    sim,
                    pd.DataFrame({"date": [next_ts], "close": [price]}),
                ],
                ignore_index=True,
            )
        baseline_values = np.array(preds, dtype=float)
        engine = "intraday_lgbm"
    else:
        baseline_values = _forecast_baseline_bars(
            context,
            horizon_bars,
            future_times=future_dates,
            historical_frame=work,
            daily_return_target=daily_return_target,
            p_up=p_up if p_up is not None else 0.5,
            interval=interval,
        )

    forecast_values = baseline_values

    if "lgbm" in engine and len(forecast_values) > 2:
        terminal = float(baseline_values[-1])
        forecast_values = _blend_lgbm_with_session_shape(
            forecast_values,
            last_close,
            context,
            work,
            future_dates,
            interval,
            daily_return_target=daily_return_target,
        )

    # Do not rescale the trained LGBM path to the daily return — that flattens steps into
    # a near-linear ramp. Baseline fallback still nudges toward the daily target in-loop.
    if daily_return_target is not None and len(forecast_values) > 0 and "lgbm" not in engine:
        bars_per_session = _RTH_5M_BARS_PER_SESSION if interval == "5m" else max(1, _RTH_5M_BARS_PER_SESSION // 12)
        session_frac = min(1.0, len(forecast_values) / bars_per_session)
        target_total = float(daily_return_target) * session_frac
        forecast_values = _scale_path_to_total_return(
            forecast_values,
            last_close,
            target_total,
            future_dates,
        )

    # Baseline-only paths: apply open overlay once. LGBM already adjusts each bar in-loop.
    if interval == "5m" and len(forecast_values) > 0 and "lgbm" not in engine:
        forecast_values = _apply_cash_open_overlay(
            forecast_values,
            future_dates,
            last_close,
            work,
            context,
        )

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
