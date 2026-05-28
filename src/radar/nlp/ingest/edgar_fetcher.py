from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# SEC Atom feed for recent 8-K material events (company filings).
SEC_ATOM_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=40&output=atom"

# Common CIK map for traded universe (free, no API key).
SYMBOL_CIK: dict[str, str] = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
}


def fetch_edgar_8k_headlines(symbols: list[str], max_items: int = 20) -> pd.DataFrame:
    """Fetch recent 8-K filing titles from SEC Atom feeds."""
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser_not_installed")
        return pd.DataFrame(columns=["date", "symbol", "title", "published"])

    rows: list[dict] = []
    for symbol in symbols:
        cik = SYMBOL_CIK.get(symbol.upper())
        if not cik:
            continue
        url = SEC_ATOM_URL.format(cik=cik)
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    dt = datetime(*published[:6], tzinfo=timezone.utc)
                else:
                    dt = datetime.now(timezone.utc)
                title = entry.get("title", "").strip()
                if not title:
                    continue
                rows.append({
                    "date": pd.Timestamp(dt.date()),
                    "symbol": symbol.upper(),
                    "title": title,
                    "published": dt,
                    "source": "edgar_8k",
                })
        except Exception as exc:
            logger.warning("edgar_fetch_failed", symbol=symbol, error=str(exc))

    return pd.DataFrame(rows)


def is_earnings_related(title: str) -> bool:
    lowered = title.lower()
    keywords = ("earnings", "results", "quarter", "q1", "q2", "q3", "q4", "eps", "revenue")
    return any(k in lowered for k in keywords)
