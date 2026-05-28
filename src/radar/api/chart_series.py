from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import yfinance as yf

from radar.forecast.intraday_forecast import forecast_intraday_series
from radar.forecast.intraday_sanitize import sanitize_intraday_closes
from radar.forecast.market_hours import filter_trading_frame, is_valid_trading_time, to_utc_iso

SUPPORTED_INTERVALS = ("5m", "1h")

INTERVAL_META: dict[str, dict[str, Any]] = {
    "5m": {
        "source": "yfinance",
        "period": "5d",
        "default_limit": 390,
        "note": "5-minute close incl. pre/post-market + short-term forecast.",
    },
    "1h": {
        "source": "yfinance",
        "period": "30d",
        "default_limit": 160,
        "note": "Hourly close incl. pre/post-market + short-term forecast.",
    },
}


def _round_price(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _normalize_intraday(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "close", "symbol"])

    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    rename = {
        "Date": "date",
        "Datetime": "date",
        "Close": "close",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    frame["symbol"] = symbol.upper()

    if "close" not in frame.columns:
        frame["close"] = pd.NA

    out = frame[["date", "close", "symbol"]]
    return out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)


def _row_to_point(row: pd.Series) -> dict[str, Any]:
    ts = row["date"]
    return {
        "date": to_utc_iso(pd.Timestamp(ts)),
        "close": _round_price(row["close"]),
    }


def get_chart_series(
    symbol: str,
    interval: str = "5m",
    limit: Optional[int] = None,
    config_dir: str = "config",
) -> dict[str, Any]:
    interval = interval.lower()
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Use 5m or 1h.")

    meta = INTERVAL_META[interval]
    symbol = symbol.upper().replace("^", "")
    period = meta["period"]
    lim = limit or meta.get("default_limit")

    df = yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        auto_adjust=True,
        prepost=True,
    )
    normalized = _normalize_intraday(df, symbol)
    normalized = sanitize_intraday_closes(normalized, interval)
    normalized = filter_trading_frame(normalized)
    if lim:
        normalized = normalized.tail(int(lim))

    points = [_row_to_point(row) for _, row in normalized.iterrows() if pd.notna(row["close"])]
    forecast = forecast_intraday_series(normalized, interval, config_dir=config_dir)
    forecast.points = [
        p for p in forecast.points
        if is_valid_trading_time(pd.Timestamp(p["date"].replace("Z", "")))
    ]

    return {
        "symbol": symbol,
        "interval": interval,
        "points": points,
        "forecast": {
            "engine": forecast.engine,
            "horizon_bars": forecast.horizon_bars,
            "points": forecast.points,
        },
        "meta": {
            "source": meta["source"],
            "period": period,
            "rows": len(points),
            "limit": lim,
            "forecast_engine": forecast.engine,
            "forecast_bars": len(forecast.points),
            "display_timezone": "Europe/Berlin",
            "note": meta["note"],
        },
    }
