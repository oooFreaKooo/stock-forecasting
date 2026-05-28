from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def fetch_rss_headlines(feed_urls: list[str], max_items: int = 50) -> pd.DataFrame:
    """
    Fetch RSS headlines from configured feeds.

    Returns date, symbol (inferred from URL), title, published columns.
    Falls back to empty frame when feedparser unavailable or fetch fails.
    """
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser_not_installed")
        return pd.DataFrame(columns=["date", "symbol", "title", "published"])

    rows: list[dict] = []
    for url in feed_urls:
        symbol = _symbol_from_feed_url(url)
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    dt = datetime(*published[:6], tzinfo=timezone.utc)
                else:
                    dt = datetime.now(timezone.utc)
                rows.append({
                    "date": pd.Timestamp(dt.date()),
                    "symbol": symbol,
                    "title": entry.get("title", ""),
                    "published": dt,
                })
        except Exception as exc:
            logger.warning("rss_fetch_failed", url=url, error=str(exc))

    return pd.DataFrame(rows)


def _symbol_from_feed_url(url: str) -> str:
    if "s=" in url:
        part = url.split("s=")[1].split("&")[0]
        return part.upper()
    return "MARKET"
