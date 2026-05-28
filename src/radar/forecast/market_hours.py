from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

BERLIN = ZoneInfo("Europe/Berlin")
UTC = ZoneInfo("UTC")

# TradingView DE: US extended hours in Europe/Berlin — active until 02:00, resumes 10:00.
OVERNIGHT_GAP_START = time(2, 0)
OVERNIGHT_GAP_END = time(10, 0)

# US cash session open on German brokers (09:30 ET → 15:30 Europe/Berlin).
NYSE_CASH_OPEN_BERLIN = time(15, 30)
NYSE_OPEN_WINDOW_END_BERLIN = time(17, 0)

STEP_DELTAS = {
    "5m": pd.Timedelta(minutes=5),
    "1h": pd.Timedelta(hours=1),
}


def to_utc_naive(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert(UTC).tz_localize(None)


def to_utc_iso(ts: pd.Timestamp) -> str:
    return to_utc_naive(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def berlin_time(ts_utc: pd.Timestamp) -> time:
    return to_utc_naive(ts_utc).tz_localize(UTC).tz_convert(BERLIN).time()


def is_valid_trading_time(ts_utc: pd.Timestamp) -> bool:
    """True during pre/regular/post market in Berlin: 10:00–24:00 and 00:00–02:00."""
    t = berlin_time(ts_utc)
    return t >= OVERNIGHT_GAP_END or t < OVERNIGHT_GAP_START


def is_cash_open_window(ts_utc: pd.Timestamp) -> bool:
    """First ~90 minutes of US regular session (15:30–17:00 Europe/Berlin)."""
    t = berlin_time(ts_utc)
    return NYSE_CASH_OPEN_BERLIN <= t < NYSE_OPEN_WINDOW_END_BERLIN


def berlin_calendar_date(ts_utc: pd.Timestamp):
    return to_utc_naive(ts_utc).tz_localize(UTC).tz_convert(BERLIN).date()


def _snap_to_berlin_premarket(ts_utc: pd.Timestamp) -> pd.Timestamp:
    berlin = to_utc_naive(ts_utc).tz_localize(UTC).tz_convert(BERLIN)
    snapped = berlin.replace(
        hour=OVERNIGHT_GAP_END.hour,
        minute=OVERNIGHT_GAP_END.minute,
        second=0,
        microsecond=0,
    )
    return to_utc_naive(snapped.tz_convert(UTC))


def advance_trading_timestamp(last_ts: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Advance one bar, never landing inside 02:00–10:00 Europe/Berlin."""
    delta = STEP_DELTAS[interval]
    nxt = to_utc_naive(last_ts) + delta
    if not is_valid_trading_time(nxt):
        nxt = _snap_to_berlin_premarket(nxt)
    return nxt


def project_trading_timestamps(last_ts: pd.Timestamp, interval: str, horizon: int) -> pd.DatetimeIndex:
    dates: list[pd.Timestamp] = []
    current = to_utc_naive(last_ts)
    for _ in range(horizon):
        current = advance_trading_timestamp(current, interval)
        if not is_valid_trading_time(current):
            current = _snap_to_berlin_premarket(current)
        dates.append(current)
    return pd.DatetimeIndex(dates)


def filter_trading_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = frame["date"].apply(lambda d: is_valid_trading_time(pd.Timestamp(d)))
    return frame.loc[mask].reset_index(drop=True)
