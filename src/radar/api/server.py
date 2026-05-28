from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from radar.api.chart_series import SUPPORTED_INTERVALS, get_chart_bundle, get_chart_series
from radar.api.service import (
    bootstrap_dashboard,
    get_all_predictions,
    get_news,
    get_performance,
    refresh_dashboard,
)
from radar.cache.artifacts import load_predictions_cache
from radar.config.settings import get_settings
from radar.jobs.scheduler import BackgroundJobRunner

API_VERSION = "2.0.0"
API_FEATURES = ["refresh", "bootstrap", "news", "meta", "background_jobs"]
STARTED_AT = datetime.now(timezone.utc)
_job_runner: BackgroundJobRunner | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _job_runner
    settings = get_settings()
    settings.ensure_dirs()
    if settings.jobs.enabled and settings.jobs.run_in_api:
        _job_runner = BackgroundJobRunner(settings)
        _job_runner.start()
    yield
    if _job_runner is not None:
        _job_runner.stop()


app = FastAPI(title="Radar AI API", version=API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    from radar.config.settings import get_settings
    from radar.ensemble.live_scorer import load_ensemble_bundle

    settings = get_settings()
    bundle = load_ensemble_bundle(settings)
    model_version = bundle.get("model_version") if bundle else None

    routes = sorted({
        route.path
        for route in app.routes
        if getattr(route, "path", None) and route.path.startswith("/")
    })
    predictions_cache = load_predictions_cache(settings)
    return {
        "version": API_VERSION,
        "started_at": STARTED_AT.isoformat(),
        "features": API_FEATURES,
        "routes": routes,
        "news_enabled": "/api/news" in routes,
        "background_jobs": settings.jobs.enabled and settings.jobs.run_in_api,
        "predictions_cached": bool(predictions_cache and predictions_cache.get("predictions")),
        "predictions_cached_at": predictions_cache.get("cached_at") if predictions_cache else None,
        "model_version": model_version,
    }


@app.get("/api/predictions")
def predictions():
    return get_all_predictions()


@app.post("/api/bootstrap")
def bootstrap():
    """Load cached predictions or score from disk artifacts (no full data refresh)."""
    try:
        return bootstrap_dashboard()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Bootstrap failed: {exc}") from exc


@app.post("/api/refresh")
def refresh():
    try:
        return refresh_dashboard()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Refresh failed: {exc}") from exc


@app.get("/api/performance")
def performance(refresh: bool = Query(False, description="Recompute OOS metrics (slow)")):
    try:
        return get_performance(refresh=refresh)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/chart/{symbol}/bundle")
def chart_bundle(
    symbol: str,
    limit: Optional[int] = Query(None, ge=10, le=5000),
):
    """Canonical 5M intraday + daily chart data (one AI run). Frontend resamples to 1H."""
    try:
        return get_chart_bundle(symbol, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch chart bundle: {exc}") from exc


@app.get("/api/chart/{symbol}")
def chart_series(
    symbol: str,
    interval: str = Query("5m", description="5m, 1h, or 1d"),
    limit: Optional[int] = Query(None, ge=10, le=5000),
):
    if interval.lower() not in SUPPORTED_INTERVALS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported interval '{interval}'. Use: {', '.join(SUPPORTED_INTERVALS)}",
        )
    try:
        return get_chart_series(symbol, interval=interval, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch chart data: {exc}") from exc


@app.get("/api/news")
def news(refresh: bool = Query(False, description="Fetch fresh RSS headlines")):
    try:
        return get_news(refresh=refresh)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load news: {exc}") from exc


def main():
    import uvicorn

    reload = os.getenv("RADAR_API_RELOAD", "").lower() in {"1", "true", "yes"}
    uvicorn.run(
        "radar.api.server:app",
        host="127.0.0.1",
        port=int(os.getenv("RADAR_API_PORT", "8000")),
        reload=reload,
    )


if __name__ == "__main__":
    main()
