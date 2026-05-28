from __future__ import annotations

from pathlib import Path

import pandas as pd

EVENT_FEATURE_COLUMNS = [
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


def load_events_calendar(processed_dir: str | Path) -> pd.DataFrame:
    path = Path(processed_dir) / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Events calendar not found at {path}. Run build_event_calendar first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_event_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Join market-wide event flags onto symbol panel."""
    cols = ["date"] + [c for c in EVENT_FEATURE_COLUMNS if c in events.columns]
    return df.merge(events[cols], on="date", how="left")


def get_event_feature_columns() -> list[str]:
    return list(EVENT_FEATURE_COLUMNS)
