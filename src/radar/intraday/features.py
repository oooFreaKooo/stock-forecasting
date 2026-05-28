from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from radar.forecast.market_hours import berlin_time, is_cash_open_window, to_utc_naive
from radar.intraday.event_features import INTRADAY_EVENT_COLUMNS, attach_event_features
from radar.intraday.news_recency import RECENCY_FEATURE_COLUMNS, attach_news_recency_features
from radar.nlp.fusion.memory_enricher import sentiment_values_from_cache


@dataclass
class IntradayFeatureFrame:
    X: pd.DataFrame
    y: pd.Series


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def build_intraday_feature_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    horizon_bars: int = 1,
    settings=None,
    headlines: Optional[pd.DataFrame] = None,
    predict_for_ts: Optional[pd.Timestamp] = None,
    use_live_sentiment: bool = True,
) -> Optional[IntradayFeatureFrame]:
    """
    Build supervised features for 5m intraday bars.

    Target y is next-bar return over `horizon_bars` steps:
      y[t] = close[t+horizon]/close[t] - 1
    """
    if frame is None or frame.empty:
        return None
    work = frame.dropna(subset=["close"]).copy()
    if len(work) < 80:
        return None

    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").reset_index(drop=True)

    if settings is not None:
        work = attach_news_recency_features(work, symbol, settings, headlines=headlines)
        work = attach_event_features(work, settings)

    close = work["close"].astype(float)

    ret1 = close.pct_change().fillna(0.0)

    feats = pd.DataFrame(index=work.index)
    feats["symbol"] = symbol.upper()
    feats["ret_1"] = ret1
    feats["ret_2"] = close.pct_change(2).fillna(0.0)
    feats["ret_3"] = close.pct_change(3).fillna(0.0)
    feats["ret_6"] = close.pct_change(6).fillna(0.0)
    feats["ret_12"] = close.pct_change(12).fillna(0.0)

    for w in (12, 36, 78):  # 1h, 3h, ~1 session
        feats[f"roll_mean_ret_{w}"] = ret1.rolling(w).mean().fillna(0.0)
        feats[f"roll_vol_ret_{w}"] = ret1.rolling(w).std().fillna(0.0)
        feats[f"roll_range_{w}"] = (close.rolling(w).max() - close.rolling(w).min()).fillna(0.0)

    feats["rsi_14"] = _rsi(close, 14)
    feats["ema_12"] = close.ewm(span=12, adjust=False).mean()
    feats["ema_26"] = close.ewm(span=26, adjust=False).mean()
    feats["macd"] = feats["ema_12"] - feats["ema_26"]
    feats["macd_signal"] = feats["macd"].ewm(span=9, adjust=False).mean()

    # Time features (UTC timestamps, displayed in Berlin)
    bt = work["date"].apply(berlin_time)
    feats["berlin_hour"] = bt.apply(lambda t: t.hour).astype(int)
    feats["berlin_minute"] = bt.apply(lambda t: t.minute).astype(int)
    mins = feats["berlin_hour"] * 60 + feats["berlin_minute"]
    feats["tod_sin"] = np.sin(2 * np.pi * mins / (24 * 60))
    feats["tod_cos"] = np.cos(2 * np.pi * mins / (24 * 60))
    feats["is_cash_open"] = work["date"].apply(lambda d: 1.0 if is_cash_open_window(pd.Timestamp(d)) else 0.0)

    if predict_for_ts is not None:
        target_ts = to_utc_naive(pd.Timestamp(predict_for_ts))
        target_bt = berlin_time(target_ts)
        last_idx = feats.index[-1]
        mins = target_bt.hour * 60 + target_bt.minute
        feats.loc[last_idx, "berlin_hour"] = target_bt.hour
        feats.loc[last_idx, "berlin_minute"] = target_bt.minute
        feats.loc[last_idx, "tod_sin"] = np.sin(2 * np.pi * mins / (24 * 60))
        feats.loc[last_idx, "tod_cos"] = np.cos(2 * np.pi * mins / (24 * 60))
        feats.loc[last_idx, "is_cash_open"] = (
            1.0 if is_cash_open_window(target_ts) else 0.0
        )

    # Daily sentiment scalars: live cache only for real-time inference (not training).
    if use_live_sentiment and settings is not None and symbol:
        sent = sentiment_values_from_cache(settings, symbol)
    else:
        sent = {}
    feats["sentiment_mean"] = float(sent.get("sentiment_mean", 0.0))
    feats["sentiment_ma"] = float(sent.get("sentiment_ma", feats["sentiment_mean"].iloc[0] if len(feats) else 0.0))
    feats["headline_count"] = float(sent.get("headline_count", 0.0))
    feats["market_sentiment"] = float(sent.get("market_sentiment", 0.0))
    feats["market_sentiment_dispersion"] = float(sent.get("market_sentiment_dispersion", 0.0))

    for col in RECENCY_FEATURE_COLUMNS + INTRADAY_EVENT_COLUMNS:
        if col in work.columns:
            feats[col] = work[col].astype(float).values

    # Target
    y = close.shift(-horizon_bars) / close - 1.0

    # Drop rows with insufficient history / future
    valid = feats.index[horizon_bars:-1]  # keep most, trimming tail
    X = feats.loc[valid].copy()
    y = y.loc[valid].astype(float)
    y = y.clip(lower=-0.20, upper=0.20)

    return IntradayFeatureFrame(X=X, y=y)

