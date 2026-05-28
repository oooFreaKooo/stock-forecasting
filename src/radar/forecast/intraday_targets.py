from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from radar.config.settings import Settings
from radar.data.store import ParquetStore
from radar.forecast.baseline import forecast_baseline, forecast_return_1d


def _load_oos_scores(settings: Settings) -> Optional[pd.DataFrame]:
    path = Path(settings.paths.processed_dir) / "ensemble_oos.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def resolve_intraday_targets(
    settings: Settings,
    symbol: str,
    anchor_ts: pd.Timestamp,
    *,
    live_scores: Optional[dict] = None,
) -> Tuple[Optional[float], float]:
    """
    Daily return hint + P(up) for intraday path, using only data available at anchor_ts.

    live_scores: optional precomputed score_live_symbol() for the latest bar only.
    """
    symbol = symbol.upper()
    anchor_ts = pd.Timestamp(anchor_ts)
    anchor_day = anchor_ts.normalize()

    p_up = 0.5
    if live_scores is not None:
        p_up = float(live_scores.get("p_ensemble", live_scores.get("p_up", 0.5)))
        ret = live_scores.get("predicted_return_1d")
        if ret is not None and not pd.isna(ret):
            return float(ret), p_up

    oos = _load_oos_scores(settings)
    if oos is not None:
        sym_oos = oos[(oos["symbol"] == symbol) & (oos["date"] <= anchor_day)]
        if not sym_oos.empty:
            row = sym_oos.iloc[-1]
            p_up = float(row.get("p_ensemble", row.get("p_up", 0.5)))

    store = ParquetStore(settings.paths.raw_dir)
    if not store.exists(symbol):
        return None, p_up

    raw = store.read(symbol)
    raw["date"] = pd.to_datetime(raw["date"])
    hist = raw[raw["date"] <= anchor_day].set_index("date")["close"].astype(float)
    if len(hist) < 20:
        return None, p_up

    fc = forecast_baseline(
        hist,
        horizon_days=settings.forecast.horizon_days,
        context_days=settings.forecast.context_days,
    )
    daily_ret = forecast_return_1d(fc, float(hist.iloc[-1]))

    if p_up >= 0.55 and daily_ret < 0:
        daily_ret = abs(daily_ret) * 0.5
    elif p_up <= 0.45 and daily_ret > 0:
        daily_ret = -abs(daily_ret) * 0.5

    return daily_ret, p_up
