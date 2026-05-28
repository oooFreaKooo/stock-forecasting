"""Alpha Vantage comparison — only what the API actually provides."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from radar.data.adapters.alphavantage import fetch_daily_closes, is_configured
from radar.forecast.market_hours import to_utc_iso

TREND_LOOKBACK = 10
MAX_DAILY_FORWARD_RETURN = 0.05

ComparisonDisplay = Literal["markers", "daily_line"]


def _daily_trend_return_1d(closes: np.ndarray) -> float:
    n = len(closes)
    if n < 5:
        return 0.0
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, closes.astype(float), 1)
    last = float(closes[-1])
    if last <= 0:
        return 0.0
    next_day = float(slope * n + intercept)
    return next_day / last - 1.0


def _blend_trend_return_1d(closes: np.ndarray, *, lookback: int = TREND_LOOKBACK) -> float:
    if len(closes) < 2:
        return 0.0
    last_day = float(closes[-1] / closes[-2] - 1.0)
    window = closes[-lookback:] if len(closes) >= lookback else closes
    fit = _daily_trend_return_1d(window)
    return float(0.4 * last_day + 0.6 * fit)


def _us_trading_date(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.tz_convert("America/New_York").strftime("%Y-%m-%d")


def _av_day_key(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.hour == 0 and t.minute == 0 and t.second == 0 and t.microsecond == 0:
        return t.strftime("%Y-%m-%d")
    return _us_trading_date(t)


def _av_by_us_day(series: pd.Series) -> dict[str, float]:
    return {_av_day_key(pd.Timestamp(idx)): float(val) for idx, val in series.items()}


def _last_bar_per_us_day(
    history_points: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pt in history_points:
        day = _us_trading_date(pd.Timestamp(str(pt["date"]).replace("Z", "")))
        out[day] = pt
    return out


def _intraday_eod_markers(
    history_points: list[dict[str, Any]],
    av_series: pd.Series,
) -> list[dict[str, Any]]:
    """
    One marker per US session: raw AV ``TIME_SERIES_DAILY`` close at that day's last bar.

    No intraday synthesis, no scaling to yfinance, no forward segment.
    """
    if not history_points or av_series.empty:
        return []
    by_day = _av_by_us_day(av_series)
    last_bars = _last_bar_per_us_day(history_points)
    points: list[dict[str, Any]] = []
    for day in sorted(last_bars):
        av_close = by_day.get(day)
        if av_close is None:
            continue
        bar = last_bars[day]
        points.append({
            "date": to_utc_iso(pd.Timestamp(str(bar["date"]).replace("Z", ""))),
            "close": round(av_close, 4),
        })
    return points


def _daily_chart_points(
    history_points: list[dict[str, Any]],
    av_series: pd.Series,
) -> list[dict[str, Any]]:
    """Raw AV close on each daily chart bar."""
    if not history_points or av_series.empty:
        return []
    by_day = _av_by_us_day(av_series)
    out: list[dict[str, Any]] = []
    for pt in history_points:
        day = _us_trading_date(pd.Timestamp(str(pt["date"]).replace("Z", "")))
        close = by_day.get(day)
        if close is None:
            continue
        out.append({
            "date": to_utc_iso(pd.Timestamp(str(pt["date"]).replace("Z", ""))),
            "close": round(close, 4),
        })
    return out


def _horizon_fraction_1d(future_dates: pd.DatetimeIndex) -> float:
    return float(max(1, len(future_dates)))


def _effective_forward_return_1d(daily_ret: float, horizon_days: float) -> float:
    scaled = daily_ret * horizon_days
    return float(np.clip(scaled, -MAX_DAILY_FORWARD_RETURN, MAX_DAILY_FORWARD_RETURN))


def _spread_return_across_bars(anchor: float, total_return: float, n_bars: int) -> np.ndarray:
    if n_bars <= 0:
        return np.array([], dtype=float)
    if abs(total_return) < 1e-12:
        return np.full(n_bars, anchor, dtype=float)
    per_bar = (1.0 + total_return) ** (1.0 / n_bars) - 1.0
    out = np.empty(n_bars, dtype=float)
    price = anchor
    for i in range(n_bars):
        price *= 1.0 + per_bar
        out[i] = price
    return out


def _forward_1d_points(
    av_last: float,
    future_dates: pd.DatetimeIndex,
    effective_ret: float,
) -> list[dict[str, Any]]:
    preds = _spread_return_across_bars(av_last, effective_ret, len(future_dates))
    return [
        {"date": to_utc_iso(pd.Timestamp(ts)), "close": round(float(price), 4)}
        for ts, price in zip(future_dates, preds)
    ]


def _dedupe_sorted_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ts: dict[str, dict[str, Any]] = {}
    for pt in points:
        by_ts[pt["date"]] = pt
    return [by_ts[k] for k in sorted(by_ts)]


def _return_1d(anchor: float, end: float) -> Optional[float]:
    if anchor <= 0:
        return None
    return float(end / anchor - 1.0)


def build_alphavantage_comparison(
    symbol: str,
    *,
    interval: str,
    anchor_price: float,
    anchor_ts: pd.Timestamp,
    future_dates: pd.DatetimeIndex,
    daily_closes: Optional[pd.Series] = None,
    history_points: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    """
    Alpha Vantage overlay honest to the free daily API.

    - 5M / 1H: scatter markers at published daily closes (EOD), not a synthetic line.
    - 1D: daily line from AV + short forward trend on daily bars only.
    """
    del anchor_price, anchor_ts

    if not is_configured() or len(future_dates) == 0:
        return None

    series = daily_closes if daily_closes is not None else fetch_daily_closes(symbol)
    if series is None or series.empty:
        return None

    av_last = float(series.iloc[-1])
    av_last_day = _av_day_key(pd.Timestamp(series.index[-1]))
    history = history_points or []
    interval_l = interval.lower()

    if interval_l in ("5m", "1h"):
        hist_pts = _intraday_eod_markers(history, series)
        forward: list[dict[str, Any]] = []
        display: ComparisonDisplay = "markers"
        ret_1d = (
            _return_1d(float(series.iloc[-2]), av_last)
            if len(series) >= 2
            else None
        )
    else:
        hist_pts = _daily_chart_points(history, series)
        daily_ret = _blend_trend_return_1d(series.values.astype(float))
        horizon = _horizon_fraction_1d(future_dates)
        effective_ret = _effective_forward_return_1d(daily_ret, horizon)
        forward = _forward_1d_points(av_last, future_dates, effective_ret)
        display = "daily_line"
        ret_1d = _return_1d(av_last, float(forward[-1]["close"])) if forward else None

    path = _dedupe_sorted_points(hist_pts + forward)
    if not path:
        return None

    return {
        "engine": "alphavantage_daily",
        "display": display,
        "points": path,
        "forward_bars": len(forward),
        "return_1d": ret_1d,
        "last_av_date": av_last_day,
        "note": (
            "EOD markers from TIME_SERIES_DAILY (no intraday feed on free tier)."
            if display == "markers"
            else "Daily AV closes; forward is a short AV trend extrapolation."
        ),
    }
