from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
from typing import Any, Optional

import pandas as pd
import structlog

from radar.nlp.ingest.rss_state import MAX_SEEN_IDS_PER_FEED

logger = structlog.get_logger(__name__)


def fetch_rss_headlines(feed_urls: list[str], max_items: int = 50) -> pd.DataFrame:
    """
    Fetch RSS headlines from configured feeds (full fetch, no conditional headers).

    Returns date, symbol (inferred from URL), title, published columns.
    """
    headlines, _, _ = fetch_rss_headlines_incremental(
        feed_urls,
        state={"feeds": {}},
        max_items=max_items,
        force_full=True,
    )
    return headlines


def fetch_rss_headlines_incremental(
    feed_urls: list[str],
    state: dict[str, Any],
    *,
    max_items: int = 50,
    force_full: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    """
    Fetch only new RSS entries per feed.

    Uses feedparser conditional GET (etag / modified) when available and skips
    entries whose ids were seen on a previous poll.
    """
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser_not_installed")
        return pd.DataFrame(columns=["date", "symbol", "title", "published"]), state, 0

    feeds_state: dict[str, Any] = state.setdefault("feeds", {})
    rows: list[dict] = []
    new_count = 0

    for url in feed_urls:
        symbol = _symbol_from_feed_url(url)
        feed_meta = feeds_state.setdefault(url, {"seen_ids": []})
        seen_ids: list[str] = list(feed_meta.get("seen_ids", []))
        seen_set = set(seen_ids)

        parse_kwargs: dict[str, Any] = {}
        if not force_full:
            if feed_meta.get("etag"):
                parse_kwargs["etag"] = feed_meta["etag"]
            modified = feed_meta.get("modified")
            if modified:
                try:
                    dt = parsedate_to_datetime(modified)
                    parse_kwargs["modified"] = dt.utctimetuple()
                except (TypeError, ValueError, OverflowError):
                    pass

        try:
            parsed = feedparser.parse(url, **parse_kwargs)
        except Exception as exc:
            logger.warning("rss_fetch_failed", url=url, error=str(exc))
            continue

        status = getattr(parsed, "status", None)
        if status == 304:
            logger.info("rss_not_modified", url=url)
            continue

        if getattr(parsed, "etag", None):
            feed_meta["etag"] = parsed.etag
        if getattr(parsed, "modified", None):
            feed_meta["modified"] = parsed.modified

        for entry in parsed.entries[:max_items]:
            entry_id = _entry_id(entry)
            if not force_full and entry_id in seen_set:
                continue

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
                "entry_id": entry_id,
            })
            new_count += 1
            if entry_id not in seen_set:
                seen_ids.insert(0, entry_id)
                seen_set.add(entry_id)

        feed_meta["seen_ids"] = seen_ids[:MAX_SEEN_IDS_PER_FEED]
        feed_meta["last_polled"] = format_datetime(datetime.now(timezone.utc))
        feeds_state[url] = feed_meta

    state["feeds"] = feeds_state
    df = pd.DataFrame(rows)
    if not df.empty and "entry_id" in df.columns:
        df = df.drop(columns=["entry_id"])
    return df, state, new_count


def _entry_id(entry: Any) -> str:
    for key in ("id", "guid", "link"):
        value = entry.get(key) if hasattr(entry, "get") else None
        if value:
            return str(value)
    title = str(entry.get("title", ""))
    published = str(entry.get("published", ""))
    digest = hashlib.sha256(f"{title}|{published}".encode()).hexdigest()[:32]
    return digest


def _symbol_from_feed_url(url: str) -> str:
    if "s=" in url:
        part = url.split("s=")[1].split("&")[0]
        return part.upper()
    return "MARKET"
