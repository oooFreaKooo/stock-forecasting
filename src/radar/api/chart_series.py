from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import yfinance as yf

from radar.api.chart_validation import build_daily_validation, build_intraday_validation
from radar.config.settings import get_settings
from radar.data.store import ParquetStore
from radar.ensemble.live_scorer import score_live_symbol
from radar.forecast.baseline import forecast_baseline, forecast_return_1d
from radar.forecast.intraday_forecast import forecast_intraday_series
from radar.forecast.intraday_sanitize import sanitize_intraday_closes
from radar.forecast.market_hours import filter_trading_frame, is_valid_trading_time, to_utc_iso

SUPPORTED_INTERVALS = ("5m", "1h", "1d")

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
    "1d": {
        "source": "parquet",
        "default_limit": 120,
        "validation_days": 30,
        "forward_horizon_days": 5,
        "note": "Daily close with 30-day backtest overlay and forward forecast.",
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


def _daily_return_target(
    settings,
    store: ParquetStore,
    symbol: str,
    last_close: float,
) -> Optional[float]:
    try:
        if not store.exists(symbol):
            return None
        raw = store.read(symbol)
        raw["date"] = pd.to_datetime(raw["date"])
        daily_close = raw.set_index("date").sort_index()["close"].astype(float)
        if len(daily_close) < 20:
            return None
        fc = forecast_baseline(
            daily_close,
            horizon_days=settings.forecast.horizon_days,
            context_days=settings.forecast.context_days,
        )
        return forecast_return_1d(fc, last_close)
    except Exception:
        return None


def _get_daily_chart_series(
    symbol: str,
    limit: Optional[int],
    config_dir: str,
) -> dict[str, Any]:
    meta = INTERVAL_META["1d"]
    settings = get_settings(config_dir)
    store = ParquetStore(settings.paths.raw_dir)
    if not store.exists(symbol):
        raise ValueError(f"No stored daily data for {symbol}. Run fetch_data first.")

    raw = store.read(symbol)
    raw["date"] = pd.to_datetime(raw["date"])
    close = raw.set_index("date").sort_index()["close"].astype(float)
    lim = limit or meta.get("default_limit")
    close = close.tail(int(lim))

    points = [
        {"date": to_utc_iso(pd.Timestamp(idx)), "close": _round_price(val)}
        for idx, val in close.items()
        if pd.notna(val)
    ]

    validation_days = int(meta.get("validation_days", 30))
    val_points, val_metrics = build_daily_validation(
        close,
        validation_days=validation_days,
        horizon_days=int(meta.get("forward_horizon_days", 5)),
        context_days=settings.forecast.context_days,
    )

    forward_points: list[dict[str, Any]] = []
    engine = "baseline"
    horizon = int(meta.get("forward_horizon_days", 5))
    if len(close) >= 20:
        fc = forecast_baseline(
            close,
            horizon_days=horizon,
            context_days=settings.forecast.context_days,
        )
        engine = fc.engine
        last_ts = pd.Timestamp(close.index[-1])
        for i, price in enumerate(fc.prices):
            ts = last_ts + pd.offsets.BDay(i + 1)
            forward_points.append({
                "date": to_utc_iso(ts),
                "close": round(float(price), 4),
            })

    return {
        "symbol": symbol,
        "interval": "1d",
        "points": points,
        "forecast": {
            "engine": engine,
            "horizon_bars": len(forward_points),
            "points": forward_points,
        },
        "validation": {
            "engine": engine,
            "points": val_points,
            "metrics": val_metrics,
        },
        "meta": {
            "source": meta["source"],
            "rows": len(points),
            "limit": lim,
            "forecast_engine": engine,
            "forecast_bars": len(forward_points),
            "validation_bars": len(val_points),
            "display_timezone": "Europe/Berlin",
            "note": meta["note"],
        },
    }


def get_chart_series(
    symbol: str,
    interval: str = "5m",
    limit: Optional[int] = None,
    config_dir: str = "config",
) -> dict[str, Any]:
    interval = interval.lower()
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Use 5m, 1h, or 1d.")

    symbol = symbol.upper().replace("^", "")

    if interval == "1d":
        return _get_daily_chart_series(symbol, limit, config_dir)

    meta = INTERVAL_META[interval]
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
        normalized = normalized.tail(int(lim)).reset_index(drop=True)

    points = [_row_to_point(row) for _, row in normalized.iterrows() if pd.notna(row["close"])]

    settings = get_settings(config_dir)
    store = ParquetStore(settings.paths.raw_dir)
    last_close = float(normalized["close"].dropna().iloc[-1]) if len(normalized) else 0.0
    live_scores = score_live_symbol(settings, symbol)
    daily_return_target = _daily_return_target(settings, store, symbol, last_close)
    if live_scores is not None:
        ret = live_scores.get("predicted_return_1d")
        if ret is not None and not pd.isna(ret):
            daily_return_target = float(ret)

    p_up = float(live_scores.get("p_up", 0.5)) if live_scores else 0.5

    forecast = forecast_intraday_series(
        normalized,
        interval,
        config_dir=config_dir,
        daily_return_target=daily_return_target,
        p_up=p_up,
    )
    forecast.points = [
        p for p in forecast.points
        if is_valid_trading_time(pd.Timestamp(p["date"].replace("Z", "")))
    ]

    val_points, val_metrics = build_intraday_validation(
        normalized,
        interval,
        symbol=symbol,
        config_dir=config_dir,
        live_scores=live_scores,
    )

    return {
        "symbol": symbol,
        "interval": interval,
        "points": points,
        "forecast": {
            "engine": forecast.engine,
            "horizon_bars": forecast.horizon_bars,
            "points": forecast.points,
        },
        "validation": {
            "engine": forecast.engine,
            "points": val_points,
            "metrics": val_metrics,
        },
        "meta": {
            "source": meta["source"],
            "period": period,
            "rows": len(points),
            "limit": lim,
            "forecast_engine": forecast.engine,
            "forecast_bars": len(forecast.points),
            "validation_bars": len(val_points),
            "display_timezone": "Europe/Berlin",
            "note": meta["note"],
        },
    }
