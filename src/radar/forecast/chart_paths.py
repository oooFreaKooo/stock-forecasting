"""Shared helpers for intraday chart series (unified AI path, 1h resample)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from radar.forecast.market_hours import to_utc_iso


def round_chart_price(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def parse_chart_ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(str(value).replace("Z", ""))


def build_unified_model_path(
    history: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    forward: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Assemble the single AI line returned as ``model.points`` (API only).

    Backtest segment + last actual close + forward forecast. The chart reads
    ``model.points`` only — not validation/forecast separately.
    """
    if not validation and not forward:
        return []
    if not history:
        return list(validation) + list(forward)

    last_point = history[-1]
    last_ts = parse_chart_ts(last_point["date"])

    backtest = [p for p in validation if parse_chart_ts(p["date"]) < last_ts]
    future = [p for p in forward if parse_chart_ts(p["date"]) > last_ts]
    return backtest + [last_point] + future


def _close_series_from_points(points: list[dict[str, Any]]) -> pd.Series:
    if not points:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(points)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["close"]).sort_values("date")
    if frame.empty:
        return pd.Series(dtype=float)
    series = frame.set_index("date")["close"].astype(float)
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_convert("UTC").tz_localize(None)
    return series


def resample_chart_points_to_1h(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last 5m close in each UTC hour (display-only aggregation)."""
    series = _close_series_from_points(points)
    if series.empty:
        return []
    hourly = series.resample("1h").last().dropna()
    return [
        {"date": to_utc_iso(pd.Timestamp(ts)), "close": round_chart_price(float(val))}
        for ts, val in hourly.items()
    ]


def resample_intraday_chart_to_1h(
    chart_5m: dict[str, Any],
    *,
    history_points: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Build 1H chart view from canonical 5M AI (no re-run).

    ``history_points``: optional longer 1H actuals (e.g. 30d yfinance). When set, only
    the AI overlay is resampled from 5M; actual prices use the wider window.
    """
    points = history_points if history_points is not None else resample_chart_points_to_1h(
        chart_5m.get("points", []),
    )
    validation = chart_5m.get("validation") or {}
    forecast = chart_5m.get("forecast") or {}
    val_points = resample_chart_points_to_1h(validation.get("points", []))
    fwd_points = resample_chart_points_to_1h(forecast.get("points", []))
    model_5m = (chart_5m.get("model") or {}).get("points") or []
    if model_5m:
        model_points = resample_chart_points_to_1h(model_5m)
    else:
        model_points = build_unified_model_path(points, val_points, fwd_points)
    meta = dict(chart_5m.get("meta") or {})
    meta["rows"] = len(points)
    meta["validation_bars"] = len(val_points)
    meta["forecast_bars"] = len(fwd_points)
    if history_points is not None:
        meta["note"] = (
            "1H prices: ~30d history. AI path (backtest + forward) from canonical 5M forecast."
        )
    else:
        meta["note"] = "1H view resampled from canonical 5M AI forecast (no second model run)."
    return {
        **chart_5m,
        "interval": "1h",
        "points": points,
        "model": {
            "engine": (chart_5m.get("model") or {}).get("engine", "none"),
            "points": model_points,
            "backtest_bars": max(0, len(model_points) - len(fwd_points) - 1),
            "forward_bars": len(fwd_points),
        },
        "forecast": {
            "engine": forecast.get("engine", "none"),
            "horizon_bars": len(fwd_points),
            "points": fwd_points,
        },
        "validation": {
            "engine": validation.get("engine", "none"),
            "points": val_points,
            "metrics": validation.get("metrics", {}),
        },
        "meta": meta,
    }
