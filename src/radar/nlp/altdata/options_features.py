from __future__ import annotations

import numpy as np
import pandas as pd


def add_options_iv_proxy(df: pd.DataFrame, vix_col: str = "vix_level") -> pd.DataFrame:
    """
    Proxy implied-volatility features from VIX level and realized vol.

    Full options chain ingestion deferred to Phase 7+.
    """
    out = df.copy()
    if vix_col not in out.columns:
        out["iv_proxy"] = np.nan
        out["iv_rv_spread"] = np.nan
        return out

    out["iv_proxy"] = out[vix_col]
    if "realized_vol_20d" in out.columns:
        out["iv_rv_spread"] = out["iv_proxy"] - out["realized_vol_20d"]
    else:
        out["iv_rv_spread"] = out.groupby("symbol")["close"].pct_change().rolling(20).std() * np.sqrt(252)
        if vix_col in out.columns:
            out["iv_rv_spread"] = out["iv_proxy"] - out["iv_rv_spread"]

    return out
