from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from radar.config.schemas import FeatureParamsConfig, LabelsConfig


def add_technical_features(df: pd.DataFrame, params: FeatureParamsConfig) -> pd.DataFrame:
    """Compute technical indicators from OHLCV."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    ret_windows = params.returns.get("windows", [1, 5, 20])
    for w in ret_windows:
        out[f"return_{w}d"] = close.pct_change(w)
        out[f"log_return_{w}d"] = np.log(close / close.shift(w))

    rsi_len = params.momentum.get("rsi_length", 14)
    out["rsi"] = ta.rsi(close, length=rsi_len)

    macd = ta.macd(
        close,
        fast=params.momentum.get("macd_fast", 12),
        slow=params.momentum.get("macd_slow", 26),
        signal=params.momentum.get("macd_signal", 9),
    )
    if macd is not None and not macd.empty:
        hist_col = [c for c in macd.columns if "MACDh" in c or "h" in c.lower()]
        if hist_col:
            out["macd_hist"] = macd[hist_col[0]]
        else:
            out["macd_hist"] = macd.iloc[:, -1]

    adx_len = params.momentum.get("adx_length", 14)
    adx = ta.adx(high, low, close, length=adx_len)
    if adx is not None and not adx.empty:
        adx_col = [c for c in adx.columns if "ADX" in c]
        out["adx"] = adx[adx_col[0]] if adx_col else adx.iloc[:, 0]

    roc_len = params.momentum.get("roc_length", 10)
    out["roc"] = ta.roc(close, length=roc_len)

    atr_len = params.volatility.get("atr_length", 14)
    atr = ta.atr(high, low, close, length=atr_len)
    out["atr"] = atr
    out["atr_pct"] = atr / close

    bb_len = int(params.volatility.get("bbands_length", 20))
    bb_std = float(params.volatility.get("bbands_std", 2.0))
    bbands = ta.bbands(close, length=bb_len, std=bb_std)
    if bbands is not None and not bbands.empty:
        lower = [c for c in bbands.columns if "BBL" in c]
        upper = [c for c in bbands.columns if "BBU" in c]
        if lower and upper:
            out["bb_pct_b"] = (close - bbands[lower[0]]) / (bbands[upper[0]] - bbands[lower[0]])

    vol_window = params.volatility.get("realized_vol_window", 20)
    out["realized_vol"] = out["log_return_1d"].rolling(vol_window).std() * np.sqrt(252)

    out["gap"] = (out["open"] - out["close"].shift(1)) / out["close"].shift(1)

    # Rolling VWAP proxy (daily approximation)
    typical = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_tp_vol = (typical * volume).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    out["vwap_dist"] = (close - vwap) / close

    obv = ta.obv(close, volume)
    obv_window = params.volume.get("obv_slope_window", 5)
    out["obv"] = obv
    out["obv_slope"] = obv.diff(obv_window) / obv_window

    vol_z_window = params.volume.get("volume_zscore_window", 20)
    vol_mean = volume.rolling(vol_z_window).mean()
    vol_std = volume.rolling(vol_z_window).std()
    out["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    return out


def get_technical_feature_columns(params: FeatureParamsConfig) -> list[str]:
    ret_windows = params.returns.get("windows", [1, 5, 20])
    cols = []
    for w in ret_windows:
        cols.extend([f"return_{w}d", f"log_return_{w}d"])
    cols.extend([
        "rsi", "macd_hist", "adx", "roc", "atr", "atr_pct", "bb_pct_b",
        "realized_vol", "gap", "vwap_dist", "obv_slope", "volume_zscore",
    ])
    return cols
