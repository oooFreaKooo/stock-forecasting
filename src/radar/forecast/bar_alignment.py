from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from radar.forecast.market_hours import STEP_DELTAS, to_utc_iso

_BAR_TOLERANCE_FACTOR = 0.51


def bar_tolerance(interval: str) -> pd.Timedelta:
    delta = STEP_DELTAS[interval]
    return delta * _BAR_TOLERANCE_FACTOR


def nearest_bar_row(
    frame: pd.DataFrame,
    target_ts: pd.Timestamp,
    interval: str,
) -> Optional[pd.Series]:
    """Return the closest actual bar to a forecast timestamp within half a bar width."""
    if frame.empty:
        return None

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    target = pd.Timestamp(target_ts)
    tol = bar_tolerance(interval)

    deltas = (work["date"] - target).abs()
    idx = int(deltas.idxmin())
    if deltas.loc[idx] > tol:
        return None
    return work.loc[idx]


def align_forecast_points(
    frame: pd.DataFrame,
    forecast_points: list[dict[str, Any]],
    interval: str,
) -> tuple[list[dict[str, Any]], list[float], list[float], list[float]]:
    """
    Pair each forecast timestamp with the nearest actual bar.

    Chart points use forecast timestamps (correct x position). Metrics use
    matched actual closes at those times.
    """
    val_points: list[dict[str, Any]] = []
    predicted: list[float] = []
    actual: list[float] = []
    prior: list[float] = []

    last_actual: Optional[float] = None
    for fpt in forecast_points:
        f_ts = pd.Timestamp(str(fpt["date"]).replace("Z", ""))
        pred = float(fpt["close"])
        row = nearest_bar_row(frame, f_ts, interval)
        if row is None:
            continue

        act = float(row["close"])
        prior_close = last_actual if last_actual is not None else act

        val_points.append({
            "date": to_utc_iso(f_ts),
            "close": round(pred, 4),
        })
        predicted.append(pred)
        actual.append(act)
        prior.append(prior_close)
        last_actual = act

    return val_points, predicted, actual, prior
