from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_finbert_pipeline = None


def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline

        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=128,
        )
        return _finbert_pipeline
    except Exception:
        return None


def _finbert_scores(texts: pd.Series) -> pd.Series:
    pipe = _load_finbert()
    if pipe is None:
        return _vader_scores(texts)

    scores: list[float] = []
    for text in texts.fillna(""):
        title = str(text).strip()
        if not title:
            scores.append(0.0)
            continue
        try:
            result = pipe(title[:512])[0]
            label = result.get("label", "neutral").lower()
            conf = float(result.get("score", 0.5))
            if label == "positive":
                scores.append(conf)
            elif label == "negative":
                scores.append(-conf)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    return pd.Series(scores, index=texts.index)


def _vader_scores(texts: pd.Series) -> pd.Series:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return texts.fillna("").apply(lambda t: analyzer.polarity_scores(str(t))["compound"])
    except ImportError:
        return texts.fillna("").apply(lambda t: 0.0 if not t else (1.0 if "up" in str(t).lower() else -0.1))


def score_headlines(texts: pd.Series, use_finbert: bool = False) -> pd.Series:
    if use_finbert:
        return _finbert_scores(texts)
    return _vader_scores(texts)


def aggregate_daily_sentiment(
    headlines: pd.DataFrame,
    window: int = 5,
    use_finbert: bool = False,
) -> pd.DataFrame:
    """
    Aggregate headline sentiment to daily symbol-level scalars.

    All timestamps truncated to date; no intraday future leakage.
    """
    if headlines.empty:
        return pd.DataFrame(columns=["date", "symbol", "sentiment_mean", "sentiment_std", "headline_count"])

    df = headlines.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["sentiment"] = score_headlines(df["title"], use_finbert=use_finbert)

    daily = (
        df.groupby(["date", "symbol"])
        .agg(
            sentiment_mean=("sentiment", "mean"),
            sentiment_std=("sentiment", "std"),
            headline_count=("sentiment", "count"),
            negative_headline_ratio=("sentiment", lambda s: float((s < 0).mean())),
        )
        .reset_index()
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

    for symbol in daily["symbol"].unique():
        mask = daily["symbol"] == symbol
        sym = daily.loc[mask].sort_values("date")
        daily.loc[mask, "sentiment_ma"] = sym["sentiment_mean"].rolling(window, min_periods=1).mean()
        daily.loc[mask, "sentiment_delta_1d"] = sym["sentiment_mean"].diff().fillna(0)
        daily.loc[mask, "headline_surprise"] = (
            sym["sentiment_mean"] - sym["sentiment_mean"].rolling(window, min_periods=1).mean()
        ).fillna(0)

    if "sentiment_ma" not in daily.columns:
        daily["sentiment_ma"] = daily["sentiment_mean"]
    if "sentiment_delta_1d" not in daily.columns:
        daily["sentiment_delta_1d"] = 0.0
    if "headline_surprise" not in daily.columns:
        daily["headline_surprise"] = 0.0
    if "negative_headline_ratio" not in daily.columns:
        daily["negative_headline_ratio"] = 0.0

    return daily


def build_market_sentiment(daily: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional market sentiment aggregate per date."""
    market = (
        daily.groupby("date")
        .agg(
            market_sentiment=("sentiment_mean", "mean"),
            market_sentiment_dispersion=("sentiment_mean", "std"),
        )
        .reset_index()
    )
    market["market_sentiment_dispersion"] = market["market_sentiment_dispersion"].fillna(0)
    return market
