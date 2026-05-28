from __future__ import annotations

from typing import Any

import structlog

from radar.cache.artifacts import invalidate_chart_cache, save_predictions_cache
from radar.config.settings import Settings
from radar.data.incremental_fetch import fetch_daily_incremental, fetch_intraday_incremental
from radar.features.pipeline import build_feature_panel

logger = structlog.get_logger(__name__)


def run_news_job(settings: Settings) -> int:
    from radar.nlp.live_news import refresh_live_news

    if not settings.nlp.enabled:
        return 0
    payload = refresh_live_news(settings, persist=True, incremental=True)
    return int(payload.get("new_headlines", 0))


def run_intraday_job(settings: Settings) -> dict[str, int]:
    settings.ensure_dirs()
    results = fetch_intraday_incremental(settings)
    if results:
        invalidate_chart_cache(settings)
    logger.info("intraday_job_complete", symbols=len(results))
    return results


def run_predictions_job(settings: Settings) -> dict[str, Any]:
    """Score symbols from existing artifacts and write predictions cache (no market fetch)."""
    from radar.api.service import get_all_predictions

    settings.ensure_dirs()
    payload = get_all_predictions(use_cache=False)
    save_predictions_cache(settings, payload)
    logger.info(
        "predictions_job_complete",
        symbols=len(payload.get("predictions", [])),
    )
    return payload


def run_startup_bootstrap(settings: Settings) -> None:
    """Fill caches on API start when empty (predictions, news, intraday bars)."""
    from radar.api.service import refresh_performance_cache
    from radar.cache.artifacts import load_performance_cache, load_predictions_cache

    settings.ensure_dirs()
    try:
        run_news_job(settings)
    except Exception as exc:
        logger.warning("startup_news_skipped", error=str(exc))
    try:
        run_intraday_job(settings)
    except Exception as exc:
        logger.warning("startup_intraday_skipped", error=str(exc))
    if not load_predictions_cache(settings):
        try:
            run_predictions_job(settings)
        except Exception as exc:
            logger.warning("startup_predictions_skipped", error=str(exc))
    if not load_performance_cache(settings):
        try:
            refresh_performance_cache()
        except Exception as exc:
            logger.warning("startup_performance_skipped", error=str(exc))


def run_daily_job(settings: Settings, *, rebuild_features: bool = True) -> dict[str, int]:
    settings.ensure_dirs()
    results = fetch_daily_incremental(settings)
    if rebuild_features:
        try:
            build_feature_panel(settings)
        except Exception as exc:
            logger.warning("feature_panel_skipped", error=str(exc))

    try:
        from radar.api.service import get_all_predictions, refresh_performance_cache

        payload = get_all_predictions(use_cache=False)
        save_predictions_cache(settings, payload)
        refresh_performance_cache()
    except Exception as exc:
        logger.warning("predictions_refresh_skipped", error=str(exc))

    logger.info("daily_job_complete", symbols=len(results))
    return results
