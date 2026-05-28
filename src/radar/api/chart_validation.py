from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from radar.config.settings import Settings, get_settings
from radar.forecast.baseline import forecast_baseline, forecast_return_1d
from radar.forecast.intraday_targets import load_oos_scores
from radar.forecast.bar_alignment import align_forecast_points
from radar.forecast.intraday_forecast import IntradayForecastResult, forecast_intraday_series
from radar.forecast.market_hours import to_utc_iso

_MIN_INTRADAY_CONTEXT = 20


def _run_segment_forecast(
    anchor_frame: pd.DataFrame,
    future_schedule: pd.DatetimeIndex,
    *,
    config_dir: str,
) -> IntradayForecastResult:
    """Walk-forward segment forecast on the canonical 5m grid."""
    return forecast_intraday_series(
        anchor_frame,
        "5m",
        config_dir=config_dir,
        horizon_bars_override=len(future_schedule),
        future_dates_override=future_schedule,
    )


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

    act_step = np.abs(np.diff(actual))
    pred_step = np.abs(np.diff(predicted))
    denom = float(np.mean(act_step)) if len(act_step) else 0.0
    vol_ratio = float(np.mean(pred_step) / denom) if denom > 1e-12 else None

    return {
        "mae": round(mae, 4),
        "mape": round(mape, 6),
        "rmse": round(rmse, 4),
        "direction_accuracy": round(direction_accuracy, 4),
        "vol_ratio": round(vol_ratio, 4) if vol_ratio is not None else None,
        "n_points": int(len(predicted)),
    }


def build_daily_validation(
    close: pd.Series,
    *,
    validation_days: Optional[int] = None,
    horizon_days: int = 5,
    context_days: int = 120,
    validation_context_days: Optional[int] = None,
    symbol: str = "",
    settings: Optional[Settings] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Walk-forward daily backtest: predict next close from history up to each day.

    When OOS ensemble scores exist, blend baseline with a probability-tilted return.
    """
    series = close.dropna().astype(float).sort_index()
    val_context = validation_context_days if validation_context_days is not None else context_days
    val_context = max(10, min(val_context, context_days))
    min_history = val_context + 5
    if len(series) < min_history + 2:
        return [], {"n_points": 0}

    if validation_days is None:
        validation_days = max(5, len(series) - val_context - 1)
    else:
        validation_days = max(5, min(validation_days, len(series) - min_history))

    blend = 0.0
    sym_oos: Optional[pd.DataFrame] = None
    if settings is not None and symbol:
        blend = float(settings.forecast.daily_validation_blend)
        oos = load_oos_scores(settings)
        if oos is not None:
            sym_oos = oos[oos["symbol"] == symbol.upper()].copy()
            sym_oos["date"] = pd.to_datetime(sym_oos["date"]).dt.normalize()

    val_points: list[dict[str, Any]] = []
    predicted: list[float] = []
    actual: list[float] = []
    prior: list[float] = []
    engine = "baseline"

    start = len(series) - validation_days
    for i in range(start, len(series) - 1):
        history = series.iloc[: i + 1]
        last_p = float(history.iloc[-1])
        try:
            fc = forecast_baseline(
                history,
                horizon_days=horizon_days,
                context_days=min(context_days, max(val_context, len(history) - 5)),
            )
            baseline_pred = float(fc.prices[0])
        except ValueError:
            continue

        pred = baseline_pred
        ts = pd.Timestamp(series.index[i])
        if sym_oos is not None and blend > 0:
            day = ts.normalize()
            rows = sym_oos[sym_oos["date"] == day]
            if not rows.empty:
                p_up = float(rows.iloc[-1].get("p_ensemble", rows.iloc[-1].get("p_up", 0.5)))
                vol = float(history.pct_change().dropna().tail(60).std())
                if np.isnan(vol) or vol <= 0:
                    vol = 0.012
                oos_ret = (p_up - 0.5) * 2.0 * vol
                baseline_ret = baseline_pred / last_p - 1.0
                blended_ret = (1.0 - blend) * baseline_ret + blend * oos_ret
                pred = last_p * (1.0 + blended_ret)
                engine = "hybrid_daily"

        act = float(series.iloc[i + 1])
        next_ts = pd.Timestamp(series.index[i + 1])
        val_points.append({
            "date": to_utc_iso(next_ts),
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
    metrics["engine"] = engine
    return val_points, metrics


def build_intraday_validation(
    frame: pd.DataFrame,
    interval: str,
    *,
    symbol: str = "",
    config_dir: str = "config",
    max_history_bars: Optional[int] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Walk-forward intraday backtest: replay the 5m LGBM forecast from successive anchors.

    Each segment uses only bars before its anchor (no look-ahead, no daily ensemble blend).
    """
    if interval != "5m":
        raise ValueError(
            f"Intraday walk-forward validation only runs on 5m bars (got {interval!r}). "
            "Resample chart series for 1h display."
        )

    settings = get_settings(config_dir)
    segment_horizon = settings.forecast.intraday_validation_horizon_5m

    work = frame.dropna(subset=["close"]).copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.reset_index(drop=True)
    if max_history_bars is not None and len(work) > int(max_history_bars):
        work = work.tail(int(max_history_bars)).reset_index(drop=True)
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

        future_schedule = pd.DatetimeIndex(actual_future["date"].iloc[:segment_horizon])
        result = _run_segment_forecast(
            anchor_frame,
            future_schedule,
            config_dir=config_dir,
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
