from __future__ import annotations

import pandas as pd
import pytest

from radar.nlp.sentiment.daily_aggregator import aggregate_daily_sentiment, build_market_sentiment
from radar.nlp.altdata.options_features import add_options_iv_proxy


def test_aggregate_daily_sentiment():
    headlines = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "title": ["Stock rises on strong earnings", "Market falls on fears", "Neutral day"],
    })
    daily = aggregate_daily_sentiment(headlines, window=2)
    assert "sentiment_mean" in daily.columns
    assert len(daily) == 2

    market = build_market_sentiment(daily)
    assert "market_sentiment" in market.columns


def test_options_iv_proxy():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5),
        "symbol": ["AAPL"] * 5,
        "close": [100, 101, 99, 102, 100],
        "vix_level": [20, 21, 22, 19, 20],
    })
    out = add_options_iv_proxy(df)
    assert "iv_proxy" in out.columns
    assert out["iv_proxy"].iloc[0] == 20
