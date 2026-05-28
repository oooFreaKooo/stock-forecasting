from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from radar.forecast.market_hours import NYSE_CASH_OPEN_BERLIN, berlin_calendar_date, berlin_time


@dataclass
class IntradayTimingResult:
    entry_quality: float
    vwap_distance: float
    opening_range_breakout: float
    first_hour_momentum: float
    volume_zscore: float
    aligned_with_daily: bool


def _compute_vwap(frame: pd.DataFrame) -> float:
    if "volume" not in frame.columns or frame["volume"].sum() <= 0:
        return float(frame["close"].iloc[-1])
    vol = frame["volume"].astype(float).clip(lower=0)
    return float((frame["close"].astype(float) * vol).sum() / vol.sum())


def compute_intraday_timing(
    bars: pd.DataFrame,
    daily_signal: int = 0,
    daily_forecast_return: float = 0.0,
) -> IntradayTimingResult:
    """
    Score intraday entry quality from recent 5m/1h bars.

    Only meaningful when daily_signal == 1; still returns a score for display.
    """
    if bars.empty or len(bars) < 5:
        return IntradayTimingResult(
            entry_quality=0.0,
            vwap_distance=0.0,
            opening_range_breakout=0.0,
            first_hour_momentum=0.0,
            volume_zscore=0.0,
            aligned_with_daily=False,
        )

    frame = bars.copy().sort_values("date")
    close = frame["close"].astype(float)
    last = float(close.iloc[-1])

    vwap = _compute_vwap(frame)
    vwap_distance = (last - vwap) / vwap if vwap else 0.0

    frame["date"] = pd.to_datetime(frame["date"])
    latest_day = berlin_calendar_date(frame["date"].iloc[-1])
    today = frame[frame["date"].apply(lambda d: berlin_calendar_date(d) == latest_day)]
    after_open = today[today["date"].apply(lambda d: berlin_time(d) >= NYSE_CASH_OPEN_BERLIN)]

    if not after_open.empty:
        session_open = float(after_open["close"].iloc[0])
        session_high = float(after_open["close"].max())
        session_low = float(after_open["close"].min())
        open_closes = after_open["close"].astype(float)
        split = min(12, max(3, len(open_closes) // 4))
        first_hour_momentum = (
            float(open_closes.iloc[split - 1] / open_closes.iloc[0] - 1)
            if len(open_closes) >= split
            else 0.0
        )
    else:
        session_open = float(close.iloc[-1])
        session_high = float(close.max())
        session_low = float(close.min())
        first_hour_momentum = 0.0

    opening_range = max(session_high - session_low, 1e-6)
    opening_range_breakout = (last - session_open) / opening_range

    if "volume" in frame.columns:
        vol = frame["volume"].astype(float)
        vol_mean = vol.mean() or 1.0
        vol_std = vol.std() or 1.0
        volume_zscore = float((vol.iloc[-1] - vol_mean) / max(vol_std, 1e-6))
    else:
        volume_zscore = 0.0

    bullish_micro = (
        0.35 * np.clip(vwap_distance * 20, -1, 1)
        + 0.25 * np.clip(opening_range_breakout, -1, 1)
        + 0.25 * np.clip(first_hour_momentum * 50, -1, 1)
        + 0.15 * np.clip(volume_zscore / 3, -1, 1)
    )
    entry_quality = float(np.clip((bullish_micro + 1) / 2, 0, 1))

    aligned = daily_signal == 1 and daily_forecast_return > 0 and entry_quality >= 0.55
    if daily_signal != 1:
        entry_quality *= 0.5

    return IntradayTimingResult(
        entry_quality=entry_quality,
        vwap_distance=float(vwap_distance),
        opening_range_breakout=float(opening_range_breakout),
        first_hour_momentum=float(first_hour_momentum),
        volume_zscore=float(volume_zscore),
        aligned_with_daily=aligned,
    )
