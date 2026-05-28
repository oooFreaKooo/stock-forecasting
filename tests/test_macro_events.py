from __future__ import annotations

import pandas as pd
import pytest

from radar.config.schemas import MacroParamsConfig
from radar.events.calendar_builder import build_event_calendar
from radar.features.events import add_event_features, EVENT_FEATURE_COLUMNS
from radar.features.macro import add_macro_features, MACRO_FEATURE_COLUMNS


def _ohlcv(dates: pd.DatetimeIndex, base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates,
        "open": base,
        "high": base * 1.01,
        "low": base * 0.99,
        "close": base,
        "volume": 1_000_000,
    })


def test_add_macro_features():
    dates = pd.bdate_range("2020-01-01", periods=60)
    traded = _ohlcv(dates)
    traded["symbol"] = "AAPL"
    macro_frames = {
        "^TNX": _ohlcv(dates, 2.0),
        "^IRX": _ohlcv(dates, 0.5),
        "HYG": _ohlcv(dates, 80.0),
        "LQD": _ohlcv(dates, 110.0),
        "UUP": _ohlcv(dates, 25.0),
        "^VVIX": _ohlcv(dates, 90.0),
    }
    out = add_macro_features(traded, macro_frames, MacroParamsConfig())
    for col in ["yield_curve_slope", "credit_stress_hyg_lqd", "usd_trend_20d"]:
        assert col in out.columns


def test_event_calendar_build(tmp_path):
    from radar.config.settings import Settings

    seed = tmp_path / "macro_dates.csv"
    seed.write_text(
        "date,event_type,description\n2020-03-15,FOMC,Emergency cut\n2020-06-10,FOMC,Statement\n"
    )
    geo = tmp_path / "geo.csv"
    geo.write_text("date,geo_risk_flag,conflict_intensity\n2020-03-01,1,0.8\n")

    settings = Settings.load(config_dir="config")
    settings.events.seed_path = str(seed)
    settings.events.geo_seed_path = str(geo)
    settings.paths.processed_dir = str(tmp_path / "processed")
    settings.data.start_date = "2020-01-01"

    cal = build_event_calendar(settings)
    assert "is_fomc_day" in cal.columns
    assert cal["is_fomc_day"].sum() >= 1


def test_add_event_features():
    dates = pd.bdate_range("2020-01-01", periods=10)
    panel = pd.DataFrame({"date": dates, "symbol": "AAPL", "close": 100.0})
    events = pd.DataFrame({
        "date": dates,
        "is_event_day": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "is_fomc_day": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "days_to_next_event": [5] * 10,
        "days_since_last_event": [10] * 10,
    })
    out = add_event_features(panel, events)
    assert out.loc[1, "is_event_day"] == 1
