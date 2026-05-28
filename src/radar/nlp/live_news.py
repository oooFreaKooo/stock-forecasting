from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.nlp.ingest.news_fetcher import fetch_rss_headlines
from radar.nlp.sentiment.daily_aggregator import aggregate_daily_sentiment, build_market_sentiment

logger = structlog.get_logger(__name__)

LIVE_NEWS_CACHE = "live_news.json"


def _headline_sentiment(title: str) -> float:
    from radar.nlp.sentiment.daily_aggregator import _vader_scores

    return float(_vader_scores(pd.Series([title])).iloc[0])


def _headlines_to_records(headlines: pd.DataFrame, limit_per_symbol: int = 8) -> list[dict[str, Any]]:
    if headlines.empty:
        return []

    df = headlines.copy()
    if "published" in df.columns:
        df = df.sort_values("published", ascending=False)
    else:
        df = df.sort_values("date", ascending=False)

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        published = row.get("published")
        if hasattr(published, "isoformat"):
            published_str = published.isoformat()
        else:
            published_str = str(published) if published is not None else None

        records.append({
            "symbol": str(row.get("symbol", "MARKET")),
            "title": title,
            "sentiment": round(_headline_sentiment(title), 4),
            "published": published_str,
            "date": pd.Timestamp(row.get("date")).date().isoformat(),
        })

    if limit_per_symbol <= 0:
        return records

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault(item["symbol"], []).append(item)

    trimmed: list[dict[str, Any]] = []
    for symbol in sorted(grouped):
        trimmed.extend(grouped[symbol][:limit_per_symbol])
    return trimmed


def refresh_live_news(settings: Settings, persist: bool = True) -> dict[str, Any]:
    """Fetch RSS headlines, score sentiment, and optionally cache for live predictions."""
    headlines = fetch_rss_headlines(settings.nlp.rss_feeds)
    daily = aggregate_daily_sentiment(
        headlines,
        window=settings.nlp.sentiment_window,
        use_finbert=settings.nlp.use_finbert,
    )
    market = build_market_sentiment(daily) if not daily.empty else pd.DataFrame()

    today = pd.Timestamp.now().normalize()
    symbol_stats: dict[str, dict[str, Any]] = {}
    for symbol in settings.universe.traded:
        sym_daily = daily[(daily["symbol"] == symbol) & (daily["date"] == today)] if not daily.empty else pd.DataFrame()
        if sym_daily.empty and not daily.empty:
            sym_hist = daily[daily["symbol"] == symbol].sort_values("date")
            if not sym_hist.empty:
                last = sym_hist.iloc[-1]
                symbol_stats[symbol] = {
                    "sentiment_mean": round(float(last["sentiment_mean"]), 4),
                    "sentiment_ma": round(float(last.get("sentiment_ma", last["sentiment_mean"])), 4),
                    "headline_count": int(last.get("headline_count", 0)),
                    "as_of_date": pd.Timestamp(last["date"]).date().isoformat(),
                }
                continue

        if not sym_daily.empty:
            row = sym_daily.iloc[0]
            symbol_stats[symbol] = {
                "sentiment_mean": round(float(row["sentiment_mean"]), 4),
                "sentiment_ma": round(float(row.get("sentiment_ma", row["sentiment_mean"])), 4),
                "headline_count": int(row.get("headline_count", 0)),
                "as_of_date": today.date().isoformat(),
            }
        else:
            symbol_stats[symbol] = {
                "sentiment_mean": 0.0,
                "sentiment_ma": 0.0,
                "headline_count": 0,
                "as_of_date": today.date().isoformat(),
            }

    market_row = market.iloc[-1] if not market.empty else None
    payload: dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "headline_count": int(len(headlines)),
        "market_sentiment": round(float(market_row["market_sentiment"]), 4) if market_row is not None else 0.0,
        "market_sentiment_dispersion": round(
            float(market_row["market_sentiment_dispersion"]), 4
        ) if market_row is not None else 0.0,
        "symbols": symbol_stats,
        "headlines": _headlines_to_records(headlines),
    }

    if persist:
        cache_path = Path(settings.paths.processed_dir) / LIVE_NEWS_CACHE
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2))
        logger.info("saved_live_news_cache", path=str(cache_path), headlines=len(headlines))

    return payload


def load_live_news_cache(settings: Settings) -> Optional[dict[str, Any]]:
    cache_path = Path(settings.paths.processed_dir) / LIVE_NEWS_CACHE
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        logger.warning("invalid_live_news_cache", path=str(cache_path))
        return None


def get_live_news(settings: Settings, refresh: bool = False) -> dict[str, Any]:
    if refresh:
        return refresh_live_news(settings, persist=True)
    cached = load_live_news_cache(settings)
    if cached is not None:
        return cached
    return refresh_live_news(settings, persist=True)


def get_symbol_sentiment(settings: Settings, symbol: str) -> Optional[dict[str, Any]]:
    cache = load_live_news_cache(settings)
    if not cache:
        return None
    symbols = cache.get("symbols", {})
    return symbols.get(symbol.upper())
