from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import yfinance as yf

from radar.api.chart_validation import build_daily_validation, build_intraday_validation
from radar.cache.artifacts import (
    CHART_CACHE_TTL_SECONDS,
    is_stale,
    load_chart_bundle_cache,
    save_chart_bundle_cache,
)
from radar.config.settings import get_settings
from radar.data.store import ParquetStore
from radar.ensemble.live_scorer import score_live_symbol
from radar.forecast.baseline import forecast_baseline, forecast_return_1d
from radar.forecast.chart_paths import (
    build_unified_model_path,
    resample_chart_points_to_1h,
    resample_intraday_chart_to_1h,
)
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
        "default_limit": 480,
        "note": "Hourly close (~30d). AI line resampled from 5m model (same run as 5M tab).",
    },
    "1d": {
        "source": "parquet",
        "forward_horizon_days": 5,
        "note": "Daily close with full-window walk-forward backtest and forward forecast.",
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


def prepare_intraday_frame(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """Normalize yfinance OHLCV, sanitize closes, and drop non-trading hours."""
    normalized = _normalize_intraday(df, symbol)
    return filter_trading_frame(sanitize_intraday_closes(normalized, interval))


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
    lim = limit or settings.forecast.chart_history_days
    close = close.tail(int(lim))

    points = [
        {"date": to_utc_iso(pd.Timestamp(idx)), "close": _round_price(val)}
        for idx, val in close.items()
        if pd.notna(val)
    ]

    val_points, val_metrics = build_daily_validation(
        close,
        validation_days=None,
        horizon_days=int(meta.get("forward_horizon_days", 5)),
        context_days=settings.forecast.context_days,
        validation_context_days=settings.forecast.daily_validation_context_days,
        symbol=symbol,
        settings=settings,
    )

    forward_points: list[dict[str, Any]] = []
    engine = str(val_metrics.get("engine", "baseline"))
    horizon = int(meta.get("forward_horizon_days", 5))
    live_scores = score_live_symbol(settings, symbol)
    daily_return_target = _daily_return_target(settings, store, symbol, float(close.iloc[-1]))
    if live_scores is not None:
        ret = live_scores.get("predicted_return_1d")
        if ret is not None and not pd.isna(ret):
            daily_return_target = float(ret)
    p_up = float(live_scores.get("p_up", 0.5)) if live_scores else 0.5

    if len(close) >= 20:
        fc = forecast_baseline(
            close,
            horizon_days=horizon,
            context_days=settings.forecast.context_days,
        )
        engine = fc.engine
        last_p = float(close.iloc[-1])
        if daily_return_target is not None and live_scores is not None:
            blend = settings.forecast.daily_validation_blend
            baseline_ret = float(fc.prices[0] / last_p - 1.0)
            blended_ret = (1.0 - blend) * baseline_ret + blend * daily_return_target
            fc.prices[0] = last_p * (1.0 + blended_ret)
            engine = "hybrid_daily"
        last_ts = pd.Timestamp(close.index[-1])
        for i, price in enumerate(fc.prices):
            ts = last_ts + pd.offsets.BDay(i + 1)
            forward_points.append({
                "date": to_utc_iso(ts),
                "close": round(float(price), 4),
            })

    model_points = build_unified_model_path(points, val_points, forward_points)

    return {
        "symbol": symbol,
        "interval": "1d",
        "points": points,
        "model": {
            "engine": engine,
            "points": model_points,
            "backtest_bars": max(0, len(model_points) - len(forward_points) - 1),
            "forward_bars": len(forward_points),
        },
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
            "ai_p_up": round(float(p_up), 6),
            "ai_daily_return_target_1d": round(float(daily_return_target), 6) if daily_return_target is not None else None,
            "ai_daily_target_price_1d": round(float(close.iloc[-1]) * (1.0 + float(daily_return_target)), 4)
            if daily_return_target is not None
            else None,
        },
    }

def _load_intraday_5m_bars(
    symbol: str,
    settings,
    *,
    period: str,
) -> pd.DataFrame:
    """Read 5m bars from local store when fresh; otherwise incremental yfinance upsert."""
    from radar.data.incremental_fetch import fetch_intraday_incremental
    from radar.data.intraday_store import IntradayBarStore

    store = IntradayBarStore(settings.paths.processed_dir)
    max_age = settings.jobs.intraday_max_age_minutes
    if not store.is_fresh(symbol, max_age):
        fetch_intraday_incremental(settings, symbols=[symbol], period=period)

    if store.exists(symbol):
        return prepare_intraday_frame(store.read(symbol), symbol, "5m")

    df5 = yf.Ticker(symbol).history(
        period=period,
        interval="5m",
        auto_adjust=True,
        prepost=True,
    )
    normalized = prepare_intraday_frame(df5, symbol, "5m")
    if not df5.empty:
        from radar.data.adapters.yfinance_source import YFinanceSource

        raw = YFinanceSource().fetch_period(symbol, period=period, interval="5m", prepost=True)
        if not raw.empty:
            store.upsert(symbol, raw)
    return normalized


def _build_intraday_5m_chart(
    symbol: str,
    limit: Optional[int] = None,
    config_dir: str = "config",
) -> dict[str, Any]:
    """Canonical intraday chart: one 5M fetch, one AI forecast, one walk-forward backtest."""
    meta = INTERVAL_META["5m"]
    period = meta["period"]
    lim = limit or meta.get("default_limit")

    settings = get_settings(config_dir)
    normalized = _load_intraday_5m_bars(symbol, settings, period=period)
    if lim:
        normalized = normalized.tail(int(lim)).reset_index(drop=True)

    points = [_row_to_point(row) for _, row in normalized.iterrows() if pd.notna(row["close"])]

    store = ParquetStore(settings.paths.raw_dir)
    last_close = float(normalized["close"].dropna().iloc[-1]) if len(normalized) else 0.0
    live_scores = score_live_symbol(settings, symbol)
    daily_return_target = _daily_return_target(settings, store, symbol, last_close)
    if live_scores is not None:
        ret = live_scores.get("predicted_return_1d")
        if ret is not None and not pd.isna(ret):
            daily_return_target = float(ret)

    p_up = float(live_scores.get("p_up", 0.5)) if live_scores else 0.5

    forecast = forecast_intraday_series(normalized, "5m", config_dir=config_dir)
    forecast_points = [
        p for p in forecast.points
        if is_valid_trading_time(pd.Timestamp(str(p["date"]).replace("Z", "")))
    ]
    forecast_engine = forecast.engine
    forecast_horizon = len(forecast_points)

    val_points, val_metrics = build_intraday_validation(
        normalized,
        "5m",
        symbol=symbol,
        config_dir=config_dir,
        max_history_bars=lim,
    )

    model_points = build_unified_model_path(points, val_points, forecast_points)

    return {
        "symbol": symbol,
        "interval": "5m",
        "points": points,
        "model": {
            "engine": forecast_engine,
            "points": model_points,
            "backtest_bars": max(0, len(model_points) - len(forecast_points) - 1),
            "forward_bars": len(forecast_points),
        },
        "forecast": {
            "engine": forecast_engine,
            "horizon_bars": forecast_horizon,
            "points": forecast_points,
        },
        "validation": {
            "engine": forecast_engine,
            "points": val_points,
            "metrics": val_metrics,
        },
        "meta": {
            "source": meta["source"],
            "period": period,
            "rows": len(points),
            "limit": lim,
            "forecast_engine": forecast_engine,
            "forecast_bars": len(forecast_points),
            "validation_bars": len(val_points),
            "display_timezone": "Europe/Berlin",
            "note": meta["note"],
            "ai_p_up": round(float(p_up), 6),
            "ai_daily_return_target_1d": round(float(daily_return_target), 6) if daily_return_target is not None else None,
            "ai_daily_target_price_1d": round(float(last_close) * (1.0 + float(daily_return_target)), 4)
            if daily_return_target is not None
            else None,
        },
    }


def _fetch_intraday_history_points(
    symbol: str,
    interval: str,
    *,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    meta = INTERVAL_META[interval]
    df = yf.Ticker(symbol).history(
        period=meta["period"],
        interval=interval,
        auto_adjust=True,
        prepost=True,
    )
    normalized = prepare_intraday_frame(df, symbol, interval)
    lim = limit or meta.get("default_limit")
    if lim:
        normalized = normalized.tail(int(lim)).reset_index(drop=True)
    return [_row_to_point(row) for _, row in normalized.iterrows() if pd.notna(row["close"])]


def _build_intraday_1h_view(
    chart_5m: dict[str, Any],
    symbol: str,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """~30d hourly actuals; AI line resampled from the canonical 5m bundle (no second model run)."""
    history_1h = _fetch_intraday_history_points(symbol, "1h", limit=limit)
    return resample_intraday_chart_to_1h(chart_5m, history_points=history_1h)


def get_chart_bundle(
    symbol: str,
    limit: Optional[int] = None,
    config_dir: str = "config",
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Single payload for the dashboard: canonical 5M intraday + 1H view + daily series.

    1H uses ~30d of hourly actuals; AI overlay is resampled from the 5M run only.
    """
    symbol = symbol.upper().replace("^", "")
    settings = get_settings(config_dir)
    if use_cache:
        cached = load_chart_bundle_cache(settings, symbol)
        if cached and not is_stale(cached.get("cached_at"), CHART_CACHE_TTL_SECONDS):
            return {k: v for k, v in cached.items() if k != "cached_at"}

    chart_5m = _build_intraday_5m_chart(symbol, limit, config_dir)
    bundle = {
        "symbol": symbol,
        "intraday": chart_5m,
        "intraday_1h": _build_intraday_1h_view(chart_5m, symbol, limit=limit),
        "daily": _get_daily_chart_series(symbol, limit, config_dir),
    }
    save_chart_bundle_cache(settings, symbol, bundle)
    return bundle


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

    chart_5m = _build_intraday_5m_chart(symbol, limit, config_dir)
    if interval == "5m":
        return chart_5m
    return _build_intraday_1h_view(chart_5m, symbol, limit=limit)
