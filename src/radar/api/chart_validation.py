from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from radar.config.settings import get_settings
from radar.forecast.baseline import forecast_baseline, forecast_return_1d
from radar.forecast.bar_alignment import align_forecast_points
from radar.forecast.intraday_forecast import forecast_intraday_series
from radar.forecast.intraday_targets import resolve_intraday_targets
from radar.forecast.market_hours import to_utc_iso

_MIN_INTRADAY_CONTEXT = 20


def _validation_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    prior_close: np.ndarray,
) -> dict[str, Any]:
    if len(predicted) == 0:
        return {
            "mae": None,
            "mape": None,
            "rmse": None,
            "direction_accuracy": None,
            "n_points": 0,
        }

    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(actual), 1e-6)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    pred_dir = predicted > prior_close
    actual_dir = actual > prior_close
    direction_accuracy = float(np.mean(pred_dir == actual_dir))

    return {
        "mae": round(mae, 4),
        "mape": round(mape, 6),
        "rmse": round(rmse, 4),
        "direction_accuracy": round(direction_accuracy, 4),
        "n_points": int(len(predicted)),
    }


def build_daily_validation(
    close: pd.Series,
    *,
    validation_days: int = 30,
    horizon_days: int = 5,
    context_days: int = 120,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Warren-style holdout: for each day in the validation window, predict the next
    close using only history up to that day (no look-ahead).
    """
    series = close.dropna().astype(float).sort_index()
    if len(series) < validation_days + 25:
        validation_days = max(5, min(validation_days, len(series) - 25))

    val_points: list[dict[str, Any]] = []
    predicted: list[float] = []
    actual: list[float] = []
    prior: list[float] = []

    start = len(series) - validation_days
    for i in range(start, len(series) - 1):
        history = series.iloc[: i + 1]
        last_p = float(history.iloc[-1])
        try:
            fc = forecast_baseline(
                history,
                horizon_days=horizon_days,
                context_days=context_days,
            )
            pred = float(fc.prices[0])
        except ValueError:
            continue

        act = float(series.iloc[i + 1])
        ts = pd.Timestamp(series.index[i + 1])
        val_points.append({
            "date": to_utc_iso(ts),
            "close": round(pred, 4),
        })
        predicted.append(pred)
        actual.append(act)
        prior.append(last_p)

    metrics = _validation_metrics(
        np.array(predicted, dtype=float),
        np.array(actual, dtype=float),
        np.array(prior, dtype=float),
    )
    metrics["validation_days"] = validation_days
    return val_points, metrics


def build_intraday_validation(
    frame: pd.DataFrame,
    interval: str,
    *,
    symbol: str = "",
    config_dir: str = "config",
    live_scores: Optional[dict] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Walk-forward intraday backtest: replay the forecast from successive anchors so the
    purple line spans most of the chart (not just the last ~1–3 days).

    Each segment uses only bars before its anchor (no look-ahead), same engine as the
    live forward forecast.
    """
    settings = get_settings(config_dir)
    fc = settings.forecast
    if interval == "5m":
        segment_horizon = fc.intraday_validation_horizon_5m
    else:
        segment_horizon = fc.intraday_validation_horizon_1h

    work = frame.dropna(subset=["close"]).copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.reset_index(drop=True)
    n = len(work)
    if n < _MIN_INTRADAY_CONTEXT + 2:
        return [], {"n_points": 0}

    # Latest walk-forward segment wins when timestamps overlap.
    by_date: dict[str, tuple[float, float, float, dict[str, Any]]] = {}
    segments = 0

    anchor_idx = _MIN_INTRADAY_CONTEXT - 1
    first_anchor_date: Optional[str] = None

    while anchor_idx < n - 1:
        anchor_frame = work.iloc[: anchor_idx + 1]
        actual_future = work.iloc[anchor_idx + 1 :]
        if actual_future.empty:
            break

        anchor_ts = anchor_frame["date"].iloc[-1]
        use_live = live_scores if anchor_idx >= n - segment_horizon - 1 else None
        daily_ret, p_up = resolve_intraday_targets(
            settings,
            symbol,
            anchor_ts,
            live_scores=use_live,
        )

        future_schedule = pd.DatetimeIndex(actual_future["date"].iloc[:segment_horizon])
        result = forecast_intraday_series(
            anchor_frame,
            interval,
            config_dir=config_dir,
            daily_return_target=daily_ret,
            p_up=p_up,
            horizon_bars_override=segment_horizon,
            future_dates_override=future_schedule,
        )
        if not result.points:
            break

        seg_points, seg_pred, seg_act, seg_prior = align_forecast_points(
            actual_future,
            result.points,
            interval,
        )
        if not seg_points:
            # Weekend/holiday: projected bars fall on days with no actual data.
            if actual_future.empty:
                break
            anchor_idx = int(actual_future.index[0])
            continue

        if first_anchor_date is None:
            first_anchor_date = to_utc_iso(pd.Timestamp(anchor_frame["date"].iloc[-1]))

        for pt, pred, act, pri in zip(seg_points, seg_pred, seg_act, seg_prior):
            by_date[pt["date"]] = (pred, act, pri, pt)

        segments += 1
        last_ts = pd.Timestamp(str(seg_points[-1]["date"]).replace("Z", ""))
        covered = work[work["date"] <= last_ts]
        anchor_idx = int(covered.index[-1]) if not covered.empty else anchor_idx + len(seg_points)

    ordered_dates = sorted(by_date.keys())
    val_points = [by_date[d][3] for d in ordered_dates]
    predicted = [by_date[d][0] for d in ordered_dates]
    actual = [by_date[d][1] for d in ordered_dates]
    prior = [by_date[d][2] for d in ordered_dates]

    metrics = _validation_metrics(
        np.array(predicted, dtype=float),
        np.array(actual, dtype=float),
        np.array(prior, dtype=float),
    )
    metrics["validation_bars"] = len(val_points)
    metrics["segments"] = segments
    metrics["anchor_date"] = first_anchor_date
    metrics["coverage_pct"] = round(
        len(val_points) / max(n - _MIN_INTRADAY_CONTEXT, 1),
        4,
    )
    return val_points, metrics
