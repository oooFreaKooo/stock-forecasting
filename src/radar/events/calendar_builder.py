from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from radar.config.settings import Settings

logger = structlog.get_logger(__name__)


def load_macro_seed(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_geo_seed(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _expand_geo_to_daily(geo: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill geopolitical flags to daily grid."""
    daily = pd.DataFrame({"date": dates})
    geo = geo.sort_values("date")
    merged = pd.merge_asof(
        daily.sort_values("date"),
        geo.sort_values("date"),
        on="date",
        direction="backward",
    )
    merged["geo_risk_flag"] = merged["geo_risk_flag"].fillna(0)
    merged["conflict_intensity"] = merged["conflict_intensity"].fillna(0)
    return merged


def build_event_calendar(settings: Settings) -> pd.DataFrame:
    """
    Build unified events parquet with macro + geo flags per date.

    Macro events use prior-day awareness for pre-event positioning.
    """
    macro_path = Path(settings.events.seed_path)
    geo_path = Path(settings.events.geo_seed_path)

    if not macro_path.exists():
        raise FileNotFoundError(f"Macro seed not found: {macro_path}")

    macro = load_macro_seed(macro_path)
    event_dates = macro.groupby("date")["event_type"].apply(lambda x: "|".join(sorted(set(x)))).reset_index()
    event_dates.columns = ["date", "event_types"]

    start = pd.Timestamp(settings.data.start_date)
    end = pd.Timestamp.now().normalize()
    dates = pd.bdate_range(start, end)

    calendar = pd.DataFrame({"date": dates})
    calendar = calendar.merge(event_dates, on="date", how="left")
    calendar["is_event_day"] = calendar["event_types"].notna().astype(int)
    calendar["is_fomc_day"] = calendar["event_types"].fillna("").str.contains("FOMC").astype(int)
    calendar["is_cpi_day"] = calendar["event_types"].fillna("").str.contains("CPI").astype(int)
    calendar["is_nfp_day"] = calendar["event_types"].fillna("").str.contains("NFP").astype(int)

    calendar["is_post_event_day"] = calendar["is_event_day"].shift(1).fillna(0).astype(int)

    calendar["days_to_next_event"] = _days_to_next_event(calendar["date"], macro["date"])
    calendar["days_since_last_event"] = _days_since_last_event(calendar["date"], macro["date"])

    if geo_path.exists():
        geo = load_geo_seed(geo_path)
        geo_daily = _expand_geo_to_daily(geo, pd.DatetimeIndex(calendar["date"]))
        calendar = calendar.merge(geo_daily, on="date", how="left")
    else:
        calendar["geo_risk_flag"] = 0
        calendar["conflict_intensity"] = 0.0

    out_path = Path(settings.paths.processed_dir) / "events.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(out_path, index=False)
    logger.info("built_event_calendar", path=str(out_path), rows=len(calendar))
    return calendar


def _days_to_next_event(dates: pd.Series, event_dates: pd.Series) -> pd.Series:
    event_sorted = sorted(pd.to_datetime(event_dates.unique()))
    result = []
    for d in pd.to_datetime(dates):
        future = [e for e in event_sorted if e > d]
        result.append((future[0] - d).days if future else 999)
    return pd.Series(result, index=dates.index)


def _days_since_last_event(dates: pd.Series, event_dates: pd.Series) -> pd.Series:
    event_sorted = sorted(pd.to_datetime(event_dates.unique()))
    result = []
    for d in pd.to_datetime(dates):
        past = [e for e in event_sorted if e <= d]
        result.append((d - past[-1]).days if past else 999)
    return pd.Series(result, index=dates.index)
