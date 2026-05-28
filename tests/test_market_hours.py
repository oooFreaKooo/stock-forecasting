from __future__ import annotations

import pandas as pd

from radar.forecast.market_hours import (
    advance_trading_timestamp,
    filter_trading_frame,
    is_valid_trading_time,
    project_trading_timestamps,
    to_utc_iso,
)


def test_to_utc_iso_appends_z():
    assert to_utc_iso(pd.Timestamp("2026-05-27T19:55:00")) == "2026-05-27T19:55:00Z"


def test_invalid_berlin_overnight_hours():
    # 03:00 Berlin = 01:00 UTC (CEST)
    assert not is_valid_trading_time(pd.Timestamp("2026-05-28T01:00:00"))
    # 10:00 Berlin = 08:00 UTC
    assert is_valid_trading_time(pd.Timestamp("2026-05-28T08:00:00"))
    # 01:00 Berlin = 23:00 UTC previous day
    assert is_valid_trading_time(pd.Timestamp("2026-05-27T23:00:00"))


def test_advance_skips_berlin_overnight_gap():
    last = pd.Timestamp("2026-05-27T23:55:00")
    nxt = advance_trading_timestamp(last, "5m")
    assert nxt == pd.Timestamp("2026-05-28T08:00:00")


def test_project_timestamps_never_in_gap():
    last = pd.Timestamp("2026-05-27T23:55:00")
    dates = project_trading_timestamps(last, "5m", 20)
    for ts in dates:
        assert is_valid_trading_time(ts), f"Invalid forecast time: {ts}"


def test_filter_trading_frame_drops_gap_rows():
    dates = pd.to_datetime([
        "2026-05-27T23:00:00",
        "2026-05-28T01:00:00",
        "2026-05-28T08:00:00",
    ])
    frame = pd.DataFrame({"date": dates, "close": [1.0, 2.0, 3.0]})
    out = filter_trading_frame(frame)
    assert len(out) == 2
    assert out.iloc[0]["close"] == 1.0
    assert out.iloc[1]["close"] == 3.0
