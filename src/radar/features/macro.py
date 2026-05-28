from __future__ import annotations

import numpy as np
import pandas as pd

from radar.config.schemas import MacroParamsConfig

MACRO_FEATURE_COLUMNS = [
    "yield_curve_slope",
    "tnx_change_20d",
    "credit_stress_hyg_lqd",
    "tlt_return_20d",
    "usd_trend_20d",
    "usd_return_5d",
    "vvix_level",
    "vix_vvix_ratio",
    "xlk_spy_spread",
    "smh_spy_spread",
]


def add_macro_features(
    df: pd.DataFrame,
    macro_frames: dict[str, pd.DataFrame],
    params: MacroParamsConfig,
) -> pd.DataFrame:
    """Add rates, credit, USD, vol structure, and sector rotation features."""
    out = df.copy()
    trend_window = params.trend_window

    closes: dict[str, pd.Series] = {}
    for name, mdf in macro_frames.items():
        key = name.replace("^", "").upper()
        series = mdf.set_index("date")["close"].sort_index()
        closes[key] = series
        ctx = mdf[["date", "close"]].rename(columns={"close": f"macro_{key}_close"})
        out = out.merge(ctx, on="date", how="left")
        out[f"macro_{key}_close"] = out[f"macro_{key}_close"].ffill()

    if "TNX" in closes and "IRX" in closes:
        tnx = out.get("macro_TNX_close", out.get("macro_TNX_close"))
        irx = out.get("macro_IRX_close")
        if tnx is not None and irx is not None:
            out["yield_curve_slope"] = tnx - irx
            out["tnx_change_20d"] = tnx.pct_change(trend_window)

    if "HYG" in closes and "LQD" in closes:
        hyg = out["macro_HYG_close"].pct_change(5)
        lqd = out["macro_LQD_close"].pct_change(5)
        out["credit_stress_hyg_lqd"] = hyg - lqd

    if "TLT" in closes:
        out["tlt_return_20d"] = out["macro_TLT_close"].pct_change(trend_window)

    if "UUP" in closes:
        out["usd_trend_20d"] = out["macro_UUP_close"].pct_change(trend_window)
        out["usd_return_5d"] = out["macro_UUP_close"].pct_change(5)

    if "VVIX" in closes:
        out["vvix_level"] = out["macro_VVIX_close"]
        if "vix_level" in out.columns:
            out["vix_vvix_ratio"] = out["vix_level"] / out["macro_VVIX_close"].replace(0, np.nan)

    if "XLK" in closes and "SPY_trend_20d" in out.columns:
        out["xlk_spy_spread"] = (
            out["macro_XLK_close"].pct_change(trend_window) - out.get("SPY_close", out["close"]).pct_change(trend_window)
        )
    elif "XLK" in closes:
        out["xlk_spy_spread"] = out["macro_XLK_close"].pct_change(trend_window)

    if "SMH" in closes:
        out["smh_spy_spread"] = out["macro_SMH_close"].pct_change(trend_window)
        if "SPY_close" in out.columns:
            out["smh_spy_spread"] = (
                out["macro_SMH_close"].pct_change(trend_window) - out["SPY_close"].pct_change(trend_window)
            )

    return out


def get_macro_feature_columns() -> list[str]:
    return list(MACRO_FEATURE_COLUMNS)
