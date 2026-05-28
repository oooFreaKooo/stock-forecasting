from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from radar.config.schemas import MemoryConfig

REGIME_VECTOR_COLUMNS = [
    "vix_level_norm",
    "vix_zscore",
    "spy_trend_20d",
    "soxx_spy_spread",
    "qqq_spy_spread",
    "avg_stock_corr",
    "vol_dispersion",
    "vol_cluster",
    "yield_curve_slope",
    "credit_stress_hyg_lqd",
    "usd_trend_20d",
]


def _pairwise_corr_matrix(returns: pd.DataFrame, window: int) -> pd.Series:
    """Rolling average pairwise correlation across columns."""
    corrs = []
    cols = list(returns.columns)
    for a, b in combinations(cols, 2):
        corrs.append(returns[a].rolling(window).corr(returns[b]))
    if not corrs:
        return pd.Series(index=returns.index, dtype=float)
    stacked = pd.concat(corrs, axis=1)
    return stacked.mean(axis=1)


def build_daily_regime_frame(
    raw_frames: dict[str, pd.DataFrame],
    traded_symbols: list[str],
    memory_config: MemoryConfig,
) -> pd.DataFrame:
    """
    Build daily market-level regime vectors from raw OHLCV.

    Each row represents macro conditions at end-of-day t using data <= t only.
    """
    window = memory_config.correlation_window
    vix_window = memory_config.vix_zscore_window
    cluster_window = memory_config.vol_cluster_window

    closes: dict[str, pd.Series] = {}
    for symbol, df in raw_frames.items():
        key = symbol.replace("^", "").upper()
        series = df.set_index("date")["close"].sort_index()
        closes[key] = series

    dates = sorted(set.intersection(*[set(s.index) for s in closes.values()]))
    if not dates:
        raise ValueError("No overlapping dates across symbols for regime encoding")

    idx = pd.DatetimeIndex(dates)
    traded_keys = [s.replace("^", "").upper() for s in traded_symbols]

    stock_returns = pd.DataFrame(
        {k: closes[k].pct_change() for k in traded_keys if k in closes},
        index=idx,
    )

    regime = pd.DataFrame(index=idx)
    regime.index.name = "date"

    if "VIX" in closes:
        vix = closes["VIX"].reindex(idx)
        regime["vix_level_norm"] = vix / vix.rolling(252, min_periods=20).mean()
        vix_mean = vix.rolling(vix_window, min_periods=5).mean()
        vix_std = vix.rolling(vix_window, min_periods=5).std()
        regime["vix_zscore"] = (vix - vix_mean) / vix_std.replace(0, np.nan)

    if "SPY" in closes:
        spy = closes["SPY"].reindex(idx)
        regime["spy_trend_20d"] = spy / spy.shift(20) - 1

    if "SOXX" in closes and "SPY" in closes:
        soxx = closes["SOXX"].reindex(idx)
        spy = closes["SPY"].reindex(idx)
        regime["soxx_spy_spread"] = soxx.pct_change(20) - spy.pct_change(20)

    if "QQQ" in closes and "SPY" in closes:
        qqq = closes["QQQ"].reindex(idx)
        spy = closes["SPY"].reindex(idx)
        regime["qqq_spy_spread"] = qqq.pct_change(20) - spy.pct_change(20)

    if "TNX" in closes and "IRX" in closes:
        tnx = closes["TNX"].reindex(idx)
        irx = closes["IRX"].reindex(idx)
        regime["yield_curve_slope"] = tnx - irx

    if "HYG" in closes and "LQD" in closes:
        hyg = closes["HYG"].reindex(idx)
        lqd = closes["LQD"].reindex(idx)
        regime["credit_stress_hyg_lqd"] = hyg.pct_change(5) - lqd.pct_change(5)

    if "UUP" in closes:
        uup = closes["UUP"].reindex(idx)
        regime["usd_trend_20d"] = uup.pct_change(20)

    regime["avg_stock_corr"] = _pairwise_corr_matrix(stock_returns, window)

    rolling_vols = stock_returns.rolling(window).std()
    regime["vol_dispersion"] = rolling_vols.std(axis=1)

    if "VIX" in closes:
        vix = closes["VIX"].reindex(idx)
        vix_pct = vix.rolling(cluster_window, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False,
        )
        regime["vol_cluster"] = pd.cut(
            vix_pct,
            bins=[-np.inf, 0.33, 0.66, np.inf],
            labels=[0, 1, 2],
        ).astype(float)

    regime = regime.reset_index()
    for col in REGIME_VECTOR_COLUMNS:
        if col not in regime.columns:
            regime[col] = np.nan

    regime = regime[["date"] + REGIME_VECTOR_COLUMNS]
    regime[REGIME_VECTOR_COLUMNS] = regime[REGIME_VECTOR_COLUMNS].fillna(0.0)
    return regime.dropna(subset=["date"])


def regime_vectors_as_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return normalized regime matrix and date id strings."""
    cols = [c for c in REGIME_VECTOR_COLUMNS if c in df.columns]
    matrix = df[cols].values.astype(float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = matrix / norms
    date_ids = df["date"].dt.strftime("%Y-%m-%d").tolist()
    return normalized, date_ids
