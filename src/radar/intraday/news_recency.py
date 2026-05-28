from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from radar.config.settings import Settings
from radar.nlp.ingest.edgar_fetcher import fetch_edgar_8k_headlines
from radar.nlp.ingest.headline_archive import append_headline_archive, load_headline_archive
from radar.nlp.ingest.news_fetcher import fetch_rss_headlines
from radar.nlp.live_news import load_live_news_cache
from radar.nlp.sentiment.daily_aggregator import score_headlines

RECENCY_WINDOWS_MIN = (30, 60, 180)

RECENCY_FEATURE_COLUMNS = [
    "news_sent_mean_30m",
    "news_sent_mean_60m",
    "news_sent_mean_180m",
    "news_count_30m",
    "news_count_60m",
    "news_count_180m",
    "news_neg_ratio_60m",
]


def _to_utc_naive(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True)
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)


def _headlines_from_live_cache(settings: Settings) -> pd.DataFrame:
    cache = load_live_news_cache(settings)
    if not cache:
        return pd.DataFrame(columns=["symbol", "title", "published", "sentiment"])

    rows: list[dict] = []
    for item in cache.get("headlines", []):
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        pub = item.get("published")
        if pub:
            published = pd.Timestamp(pub)
        else:
            day = pd.Timestamp(item.get("date", pd.Timestamp.now().date()))
            published = day + pd.Timedelta(hours=12)
        rows.append({
            "symbol": str(item.get("symbol", "MARKET")).upper(),
            "title": title,
            "published": published,
            "sentiment": float(item.get("sentiment", 0.0)),
        })
    return pd.DataFrame(rows)


def ensure_scored_headlines(settings: Settings) -> pd.DataFrame:
    """
    Load headlines with published timestamps and sentiment scores.

    Refreshes RSS/EDGAR into the archive when empty; merges live cache headlines.
    """
    archive = load_headline_archive(settings.paths.processed_dir)
    if archive.empty or "published" not in archive.columns:
        rss = fetch_rss_headlines(settings.nlp.rss_feeds)
        edgar = fetch_edgar_8k_headlines(settings.universe.traded)
        fresh = pd.concat([rss, edgar], ignore_index=True) if not edgar.empty else rss
        if not fresh.empty and "published" in fresh.columns:
            fresh = fresh.copy()
            fresh["sentiment"] = score_headlines(fresh["title"], use_finbert=settings.nlp.use_finbert)
            archive = append_headline_archive(settings.paths.processed_dir, fresh)

    live = _headlines_from_live_cache(settings)
    frames = [f for f in [archive, live] if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["symbol", "title", "published", "sentiment"])

    work = pd.concat(frames, ignore_index=True)
    work["title"] = work["title"].fillna("").astype(str).str.strip()
    work = work[work["title"] != ""]
    work["symbol"] = work["symbol"].fillna("MARKET").astype(str).str.upper()

    if "published" not in work.columns:
        work["published"] = pd.to_datetime(work["date"]) + pd.Timedelta(hours=12)
    work["published"] = _to_utc_naive(work["published"])

    if "sentiment" not in work.columns:
        work["sentiment"] = score_headlines(work["title"], use_finbert=settings.nlp.use_finbert)
    else:
        work["sentiment"] = work["sentiment"].astype(float)

    work = work.drop_duplicates(
        subset=["published", "symbol", "title"],
        keep="last",
    ).sort_values("published").reset_index(drop=True)
    return work


def _window_stats(
    pub: np.ndarray,
    sent: np.ndarray,
    bar_times: np.ndarray,
    window: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each bar time t, aggregate headlines with published in (t-window, t]."""
    n = len(bar_times)
    means = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    neg_ratio = np.zeros(n, dtype=float)

    if len(pub) == 0:
        return means, counts, neg_ratio

    delta = np.timedelta64(int(window.total_seconds()), "s")

    for i, t in enumerate(bar_times):
        t64 = np.datetime64(t)
        left = np.searchsorted(pub, t64 - delta, side="left")
        right = np.searchsorted(pub, t64, side="right")
        if right <= left:
            continue
        chunk = sent[left:right]
        counts[i] = float(right - left)
        means[i] = float(np.mean(chunk))
        neg_ratio[i] = float(np.mean(chunk < 0))

    return means, counts, neg_ratio


def attach_news_recency_features(
    frame: pd.DataFrame,
    symbol: str,
    settings: Settings,
    headlines: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add rolling news sentiment/count features aligned to each bar (no look-ahead).
    """
    work = frame.copy()
    work["date"] = _to_utc_naive(work["date"])
    n = len(work)
    defaults = {col: 0.0 for col in RECENCY_FEATURE_COLUMNS}
    for col, val in defaults.items():
        work[col] = val

    if headlines is None:
        headlines = ensure_scored_headlines(settings)
    if headlines.empty:
        return work

    sym = symbol.upper()
    relevant = headlines[headlines["symbol"].isin([sym, "MARKET"])].copy()
    if relevant.empty:
        return work

    pub = relevant["published"].to_numpy(dtype="datetime64[ns]")
    sent = relevant["sentiment"].to_numpy(dtype=float)
    bar_times = work["date"].to_numpy(dtype="datetime64[ns]")

    for window_min in RECENCY_WINDOWS_MIN:
        window = pd.Timedelta(minutes=window_min)
        means, counts, neg = _window_stats(pub, sent, bar_times, window)
        work[f"news_sent_mean_{window_min}m"] = means
        work[f"news_count_{window_min}m"] = counts
        if window_min == 60:
            work["news_neg_ratio_60m"] = neg

    return work
