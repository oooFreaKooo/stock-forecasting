from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from radar.api.chart_series import SUPPORTED_INTERVALS, get_chart_series
from radar.api.service import (
    get_all_predictions,
    get_news,
    get_performance,
    refresh_dashboard,
)

API_VERSION = "2.0.0"
API_FEATURES = ["refresh", "news", "meta"]
STARTED_AT = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


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
    return {
        "version": API_VERSION,
        "started_at": STARTED_AT.isoformat(),
        "features": API_FEATURES,
        "routes": routes,
        "news_enabled": "/api/news" in routes,
        "model_version": model_version,
    }


@app.get("/api/predictions")
def predictions():
    return get_all_predictions()


@app.post("/api/refresh")
def refresh():
    try:
        return refresh_dashboard()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Refresh failed: {exc}") from exc


@app.get("/api/performance")
def performance():
    try:
        return get_performance()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/chart/{symbol}")
def chart_series(
    symbol: str,
    interval: str = Query("5m", description="5m or 1h"),
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
