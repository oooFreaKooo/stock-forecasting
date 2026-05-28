from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_gdelt_daily(seed_path: str | Path) -> pd.DataFrame:
    """
    Load GDELT-style geopolitical aggregates from seed CSV.

    Phase 7+ can replace with live GDELT API pulls.
    """
    path = Path(seed_path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "geo_risk_flag", "conflict_intensity"])

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def expand_gdelt_to_daily(gdelt: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    daily = pd.DataFrame({"date": dates})
    gdelt = gdelt.sort_values("date")
    merged = pd.merge_asof(
        daily.sort_values("date"),
        gdelt,
        on="date",
        direction="backward",
    )
    merged["geo_risk_flag"] = merged["geo_risk_flag"].fillna(0)
    merged["conflict_intensity"] = merged["conflict_intensity"].fillna(0)
    return merged
