from __future__ import annotations

import numpy as np
import pandas as pd

from radar.config.schemas import FeatureParamsConfig


def add_context_features(
    traded_df: pd.DataFrame,
    context_frames: dict[str, pd.DataFrame],
    params: FeatureParamsConfig,
) -> pd.DataFrame:
    """Add market context features from SPY, QQQ, SOXX, VIX."""
    out = traded_df.copy()
    trend_window = params.context.get("trend_window", 20)
    beta_window = params.context.get("beta_window", 60)
    vix_change_window = params.context.get("vix_change_window", 5)
    corr_window = params.context.get("correlation_window", 20)

    traded_close = out["close"]
    traded_ret = traded_close.pct_change()

    for name, ctx_df in context_frames.items():
        key = name.replace("^", "").upper()
        ctx = ctx_df[["date", "close"]].rename(columns={"close": f"{key}_close"})
        out = out.merge(ctx, on="date", how="left")
        out[f"{key}_close"] = out[f"{key}_close"].ffill()

        out[f"{key}_trend_{trend_window}d"] = (
            out[f"{key}_close"] / out[f"{key}_close"].shift(trend_window) - 1
        )
        out[f"{key}_return_1d"] = out[f"{key}_close"].pct_change()

    if "SPY_close" in out.columns:
        spy_ret = out["SPY_close"].pct_change()
        cov = traded_ret.rolling(beta_window).cov(spy_ret)
        var = spy_ret.rolling(beta_window).var()
        out["beta_spy"] = cov / var.replace(0, np.nan)

    if "VIX_close" in out.columns:
        out["vix_level"] = out["VIX_close"]
        out["vix_change"] = out["VIX_close"].pct_change(vix_change_window)

    if "SOXX_close" in out.columns and "SPY_close" in out.columns:
        out["soxx_spy_spread"] = (
            out["SOXX_close"].pct_change(trend_window) - out["SPY_close"].pct_change(trend_window)
        )

    if "SOXX_close" in out.columns:
        soxx_ret = out["SOXX_close"].pct_change()
        out["corr_soxx"] = traded_ret.rolling(corr_window).corr(soxx_ret)

    return out


def get_context_feature_columns() -> list[str]:
    return [
        "SPY_trend_20d", "SPY_return_1d",
        "QQQ_trend_20d", "QQQ_return_1d",
        "SOXX_trend_20d", "SOXX_return_1d",
        "beta_spy", "vix_level", "vix_change", "soxx_spy_spread", "corr_soxx",
    ]
