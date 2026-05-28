from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from radar.config.settings import Settings
from radar.features.events import load_events_calendar

INTRADAY_EVENT_COLUMNS = [
    "is_event_day",
    "is_fomc_day",
    "is_cpi_day",
    "is_nfp_day",
    "is_post_event_day",
    "days_to_next_event",
    "days_since_last_event",
    "geo_risk_flag",
    "conflict_intensity",
]


def ensure_events_calendar(settings: Settings) -> pd.DataFrame:
    path = Path(settings.paths.processed_dir) / "events.parquet"
    if path.exists():
        return load_events_calendar(settings.paths.processed_dir)

    from radar.events.calendar_builder import build_event_calendar

    return build_event_calendar(settings)


def attach_event_features(
    frame: pd.DataFrame,
    settings: Settings,
    events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join daily event flags onto intraday bars by calendar date."""
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["_cal_date"] = work["date"].dt.normalize()

    try:
        events = events if events is not None else ensure_events_calendar(settings)
    except FileNotFoundError:
        for col in INTRADAY_EVENT_COLUMNS:
            work[col] = 0.0
        return work.drop(columns=["_cal_date"], errors="ignore")

    ev = events.copy()
    ev["_event_date"] = pd.to_datetime(ev["date"]).dt.normalize()
    cols = ["_event_date"] + [c for c in INTRADAY_EVENT_COLUMNS if c in ev.columns]
    ev = ev[cols]

    merged = work.merge(ev, left_on="_cal_date", right_on="_event_date", how="left")

    for col in INTRADAY_EVENT_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)

    return merged.drop(columns=["_cal_date", "_event_date"], errors="ignore")
