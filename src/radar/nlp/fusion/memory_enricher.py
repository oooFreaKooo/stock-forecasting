from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.nlp.altdata.options_features import add_options_iv_proxy
from radar.nlp.ingest.edgar_fetcher import fetch_edgar_8k_headlines
from radar.nlp.ingest.gdelt_fetcher import expand_gdelt_to_daily, load_gdelt_daily
from radar.nlp.ingest.gdelt_live import fetch_gdelt_geo_daily
from radar.nlp.ingest.headline_archive import append_headline_archive, load_headline_archive
from radar.nlp.ingest.news_fetcher import fetch_rss_headlines
from radar.nlp.live_news import load_live_news_cache
from radar.nlp.sentiment.daily_aggregator import aggregate_daily_sentiment, build_market_sentiment

logger = structlog.get_logger(__name__)

SENTIMENT_FEATURE_COLUMNS = [
    "sentiment_mean",
    "sentiment_ma",
    "headline_count",
    "market_sentiment",
    "market_sentiment_dispersion",
    "sentiment_delta_1d",
    "headline_surprise",
    "negative_headline_ratio",
    "iv_proxy",
    "iv_rv_spread",
    "geo_nlp_risk",
]


def _collect_headlines(settings: Settings) -> pd.DataFrame:
    rss = fetch_rss_headlines(settings.nlp.rss_feeds)
    edgar = fetch_edgar_8k_headlines(settings.universe.traded)
    fresh = pd.concat([rss, edgar], ignore_index=True) if not edgar.empty else rss
    archive = append_headline_archive(settings.paths.processed_dir, fresh)
    if archive.empty:
        return fresh
    combined = pd.concat([archive, fresh], ignore_index=True)
    combined["_key"] = (
        combined["date"].astype(str)
        + "|"
        + combined["symbol"].astype(str)
        + "|"
        + combined["title"].fillna("").astype(str).str.lower()
    )
    return combined.drop_duplicates(subset=["_key"], keep="last").drop(columns=["_key"])


def _load_geo_daily(settings: Settings, dates: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    live = fetch_gdelt_geo_daily(lookback_days=60)
    seed = load_gdelt_daily(settings.nlp.gdelt_seed_path)
    if not live.empty and not seed.empty:
        seed["date"] = pd.to_datetime(seed["date"])
        geo = pd.concat([seed, live], ignore_index=True).drop_duplicates(subset=["date"], keep="last")
    elif not live.empty:
        geo = live
    elif not seed.empty:
        geo = seed
    else:
        return None
    return expand_gdelt_to_daily(geo, dates)


def build_sentiment_panel(settings: Settings) -> pd.DataFrame:
    """Build daily sentiment aggregates and merge onto feature panel."""
    from radar.features.pipeline import load_feature_panel

    panel = load_feature_panel(settings)
    headlines = _collect_headlines(settings)
    daily_sent = aggregate_daily_sentiment(
        headlines,
        window=settings.nlp.sentiment_window,
        use_finbert=settings.nlp.use_finbert,
    )
    market_sent = build_market_sentiment(daily_sent) if not daily_sent.empty else pd.DataFrame()

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    geo_daily = _load_geo_daily(settings, dates)

    enriched = panel.copy()
    if not daily_sent.empty:
        merge_cols = [
            "date", "symbol", "sentiment_mean", "sentiment_ma", "headline_count",
            "sentiment_delta_1d", "headline_surprise", "negative_headline_ratio",
        ]
        merge_cols = [c for c in merge_cols if c in daily_sent.columns]
        enriched = enriched.merge(daily_sent[merge_cols], on=["date", "symbol"], how="left")

    if not market_sent.empty:
        enriched = enriched.merge(market_sent, on="date", how="left")

    if geo_daily is not None:
        enriched = enriched.merge(
            geo_daily.rename(columns={"geo_risk_flag": "geo_nlp_risk"}),
            on="date",
            how="left",
        )

    enriched = add_options_iv_proxy(enriched)

    for col in SENTIMENT_FEATURE_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = 0.0
        enriched[col] = enriched.groupby("symbol")[col].ffill().fillna(0)

    logger.info("built_sentiment_features", rows=len(enriched))
    return enriched


def save_sentiment_panel(settings: Settings, panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if panel is None:
        panel = build_sentiment_panel(settings)
    path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    logger.info("saved_sentiment_panel", path=str(path), rows=len(panel))
    return panel


def apply_live_sentiment_from_cache(settings: Settings, panel: pd.DataFrame) -> pd.DataFrame:
    """Patch latest panel rows with fresh live_news.json sentiment."""
    cache = load_live_news_cache(settings)
    if not cache:
        return panel

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    market_sent = cache.get("market_sentiment", 0.0)
    market_disp = cache.get("market_sentiment_dispersion", 0.0)

    for symbol, stats in cache.get("symbols", {}).items():
        sym = symbol.upper()
        mask = (out["symbol"] == sym) & (out["date"] == out[out["symbol"] == sym]["date"].max())
        if not mask.any():
            continue
        idx = out.index[mask][0]
        out.at[idx, "sentiment_mean"] = float(stats.get("sentiment_mean", 0.0))
        out.at[idx, "sentiment_ma"] = float(stats.get("sentiment_ma", stats.get("sentiment_mean", 0.0)))
        out.at[idx, "headline_count"] = float(stats.get("headline_count", 0))
        out.at[idx, "market_sentiment"] = float(market_sent)
        out.at[idx, "market_sentiment_dispersion"] = float(market_disp)

    latest_mask = out.groupby("symbol")["date"].transform("max") == out["date"]
    out.loc[latest_mask, "market_sentiment"] = float(market_sent)
    out.loc[latest_mask, "market_sentiment_dispersion"] = float(market_disp)

    for col in SENTIMENT_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0

    return out


def sentiment_values_from_cache(settings: Settings, symbol: str) -> dict[str, Any]:
    cache = load_live_news_cache(settings)
    if not cache:
        return {}
    sym = cache.get("symbols", {}).get(symbol.upper(), {})
    return {
        "sentiment_mean": float(sym.get("sentiment_mean", 0.0)),
        "sentiment_ma": float(sym.get("sentiment_ma", sym.get("sentiment_mean", 0.0))),
        "headline_count": float(sym.get("headline_count", 0)),
        "market_sentiment": float(cache.get("market_sentiment", 0.0)),
        "market_sentiment_dispersion": float(cache.get("market_sentiment_dispersion", 0.0)),
    }
