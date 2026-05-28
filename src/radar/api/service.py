from __future__ import annotations

from typing import Any

import structlog

from radar.config.settings import get_settings
from radar.data.fetcher import fetch_and_store
from radar.features.pipeline import build_feature_panel, enrich_memory_if_available
from radar.forecast.hybrid_predictor import evaluate_gated_performance, predict_symbol
from radar.monitoring.paper_tracker import log_paper_signal
from radar.nlp.fusion.memory_enricher import apply_live_sentiment_from_cache, save_sentiment_panel
from radar.portfolio.allocator import apply_portfolio_limits

logger = structlog.get_logger(__name__)


def _round_price(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _prediction_to_dict(pred) -> dict[str, Any]:
    item = {
        "symbol": pred.symbol,
        "date": pred.date.isoformat() if hasattr(pred.date, "isoformat") else str(pred.date),
        "last_close": _round_price(pred.last_close),
        "p_up": round(float(pred.p_up), 6) if pred.p_up is not None else None,
        "forecast_return_1d": round(float(pred.forecast_return_1d), 6),
        "signal": pred.signal,
        "confidence": pred.confidence,
        "confluence_score": round(float(pred.confluence_score), 4) if pred.confluence_score is not None else None,
        "action": "BUY" if pred.signal else "WAIT",
        "probability_source": getattr(pred, "probability_source", "oos"),
    }
    if getattr(pred, "sentiment_mean", None) is not None:
        item["sentiment_mean"] = round(float(pred.sentiment_mean), 4)
    if getattr(pred, "headline_count", None) is not None:
        item["headline_count"] = int(pred.headline_count)
    if getattr(pred, "market_sentiment", None) is not None:
        item["market_sentiment"] = round(float(pred.market_sentiment), 4)
    if getattr(pred, "news_fetched_at", None):
        item["news_fetched_at"] = pred.news_fetched_at
    if getattr(pred, "entry_quality", None) is not None:
        item["entry_quality"] = round(float(pred.entry_quality), 4)
    if getattr(pred, "position_size", None) is not None:
        item["position_size"] = round(float(pred.position_size), 4)
    if getattr(pred, "predicted_return_1d", None) is not None:
        item["predicted_return_1d"] = round(float(pred.predicted_return_1d), 6)
    return item


def get_all_predictions(config_dir: str = "config") -> dict[str, Any]:
    settings = get_settings(config_dir)
    settings.ensure_dirs()
    predictions = []
    for symbol in settings.universe.traded:
        try:
            pred = predict_symbol(settings, symbol)
            predictions.append(_prediction_to_dict(pred))
        except Exception as exc:
            predictions.append({"symbol": symbol, "error": str(exc)})

    predictions = apply_portfolio_limits(predictions, settings)
    for item in predictions:
        if item.get("signal") == 1:
            log_paper_signal(settings.paths.processed_dir, item)

    return {"predictions": predictions, "strategy": "ensemble_gated"}


def get_performance(config_dir: str = "config") -> dict[str, Any]:
    settings = get_settings(config_dir)
    return evaluate_gated_performance(settings, save_optimized=False)


def get_news(config_dir: str = "config", refresh: bool = False) -> dict[str, Any]:
    from radar.nlp.live_news import get_live_news

    settings = get_settings(config_dir)
    settings.ensure_dirs()
    if not settings.nlp.enabled:
        return {"enabled": False, "headlines": [], "symbols": {}, "headline_count": 0}
    return {"enabled": True, **get_live_news(settings, refresh=refresh)}


def refresh_dashboard(config_dir: str = "config", *, refresh_news: bool = True) -> dict[str, Any]:
    """Fetch latest market data, rebuild features, and regenerate predictions."""
    settings = get_settings(config_dir)
    settings.ensure_dirs()

    logger.info("refresh_start")
    fetched = fetch_and_store(settings)
    build_feature_panel(settings)
    try:
        enrich_memory_if_available(settings)
    except Exception as exc:
        logger.warning("memory_enrich_skipped", error=str(exc))

    if settings.nlp.enabled:
        news = get_news(config_dir, refresh=refresh_news)
        try:
            panel = save_sentiment_panel(settings)
            panel = apply_live_sentiment_from_cache(settings, panel)
            from pathlib import Path

            panel_path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
            panel.to_parquet(panel_path, index=False)
        except Exception as exc:
            logger.warning("sentiment_enrich_skipped", error=str(exc))
    else:
        news = get_news(config_dir, refresh=refresh_news)

    predictions = get_all_predictions(config_dir)

    metrics: dict[str, Any] | None = None
    try:
        metrics = get_performance(config_dir)
    except FileNotFoundError:
        logger.warning("performance_unavailable", reason="no OOS predictions")

    logger.info("refresh_complete", symbols=len(fetched))
    return {
        "fetched": fetched,
        "news": news,
        "metrics": metrics,
        **predictions,
    }
