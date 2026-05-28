from __future__ import annotations

from typing import Any

import structlog

from radar.cache.artifacts import (
    invalidate_chart_cache,
    load_performance_cache,
    load_predictions_cache,
    save_performance_cache,
    save_predictions_cache,
)
from radar.config.settings import get_settings
from radar.data.incremental_fetch import fetch_daily_incremental, fetch_intraday_incremental
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
        "action": "BUY" if pred.signal else "NO TRADE",
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
    if getattr(pred, "gates", None):
        item["gates"] = {k: bool(v) for k, v in pred.gates.items()}
    if getattr(pred, "probability_threshold", None) is not None:
        item["probability_threshold"] = round(float(pred.probability_threshold), 4)
    return item


def get_all_predictions(config_dir: str = "config", *, use_cache: bool = True) -> dict[str, Any]:
    settings = get_settings(config_dir)
    settings.ensure_dirs()
    if use_cache:
        cached = load_predictions_cache(settings)
        if cached and cached.get("predictions") is not None:
            return {
                "predictions": cached["predictions"],
                "strategy": cached.get("strategy", "ensemble_gated"),
            }
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

    payload = {"predictions": predictions, "strategy": "ensemble_gated"}
    save_predictions_cache(settings, payload)
    return payload


def _performance_for_api(metrics: dict[str, Any]) -> dict[str, Any]:
    """Strip non-JSON fields (e.g. nested config objects) for API/cache responses."""
    params = metrics.get("optimized_params") or {}
    if hasattr(params, "model_dump"):
        params = params.model_dump()
    elif not isinstance(params, dict):
        params = dict(params) if params else {}

    return {
        "threshold_used": float(metrics.get("threshold_used", 0.5)),
        "optimized_params": params,
        "simple_hit_rate": float(metrics.get("simple_hit_rate", 0.0)),
        "simple_trades": int(metrics.get("simple_trades", 0)),
        "gated_v1_hit_rate": float(metrics.get("gated_v1_hit_rate", 0.0)),
        "gated_v1_trades": int(metrics.get("gated_v1_trades", 0)),
        "gated_hit_rate": float(metrics.get("gated_hit_rate", 0.0)),
        "gated_trades": int(metrics.get("gated_trades", 0)),
        "coverage_pct": float(metrics.get("coverage_pct", 0.0)),
        "expectancy": float(metrics.get("expectancy", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "paper_trading": metrics.get("paper_trading") or {},
    }


def get_performance(config_dir: str = "config", *, use_cache: bool = True, refresh: bool = False) -> dict[str, Any]:
    settings = get_settings(config_dir)
    if use_cache and not refresh:
        cached = load_performance_cache(settings)
        if cached:
            return {k: v for k, v in cached.items() if k != "cached_at"}

    raw = evaluate_gated_performance(settings, save_optimized=False)
    metrics = _performance_for_api(raw)
    save_performance_cache(settings, metrics)
    return metrics


def refresh_performance_cache(config_dir: str = "config") -> dict[str, Any] | None:
    """Recompute OOS metrics and persist; returns None when artifacts are missing."""
    try:
        return get_performance(config_dir, use_cache=False, refresh=True)
    except FileNotFoundError:
        logger.warning("performance_cache_skipped", reason="no OOS predictions")
        return None


def get_news(config_dir: str = "config", refresh: bool = False) -> dict[str, Any]:
    from radar.cache.artifacts import NEWS_CACHE_TTL_SECONDS
    from radar.nlp.live_news import get_live_news

    settings = get_settings(config_dir)
    settings.ensure_dirs()
    if not settings.nlp.enabled:
        return {"enabled": False, "headlines": [], "symbols": {}, "headline_count": 0}
    return {
        "enabled": True,
        **get_live_news(
            settings,
            refresh=refresh,
            max_age_seconds=None if refresh else NEWS_CACHE_TTL_SECONDS,
        ),
    }


def bootstrap_dashboard(config_dir: str = "config") -> dict[str, Any]:
    """
    Fast warm-up for the UI: serve cached predictions or score from existing artifacts.

    Does not refetch full daily history or rebuild the feature panel.
    """
    settings = get_settings(config_dir)
    settings.ensure_dirs()

    cached = load_predictions_cache(settings)
    news = get_news(config_dir, refresh=False)
    metrics = get_performance(config_dir, use_cache=True)
    if cached and cached.get("predictions"):
        return {
            "status": "cached",
            "cached_at": cached.get("cached_at"),
            "news": news,
            "metrics": metrics,
            **get_all_predictions(config_dir, use_cache=True),
        }

    logger.info("bootstrap_compute_predictions")
    predictions = get_all_predictions(config_dir, use_cache=False)
    return {
        "status": "computed",
        "news": news,
        "metrics": metrics,
        **predictions,
    }


def refresh_dashboard(config_dir: str = "config", *, refresh_news: bool = True) -> dict[str, Any]:
    """Fetch latest market data, rebuild features, and regenerate predictions."""
    settings = get_settings(config_dir)
    settings.ensure_dirs()

    logger.info("refresh_start")
    fetched = fetch_daily_incremental(settings)
    fetch_intraday_incremental(settings)
    invalidate_chart_cache(settings)
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

    predictions = get_all_predictions(config_dir, use_cache=False)

    metrics = refresh_performance_cache(config_dir)

    logger.info("refresh_complete", symbols=len(fetched))
    return {
        "fetched": fetched,
        "news": news,
        "metrics": metrics,
        **predictions,
    }
