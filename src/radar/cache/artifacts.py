"""Disk-backed API caches (predictions, charts, news freshness)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from radar.config.settings import Settings

logger = structlog.get_logger(__name__)

# Defaults: chart recomputation is expensive; news RSS is cheap but redundant every refresh.
CHART_CACHE_TTL_SECONDS = 5 * 60
NEWS_CACHE_TTL_SECONDS = 20 * 60


def _cache_root(settings: Settings) -> Path:
    root = Path(settings.paths.processed_dir) / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def is_stale(fetched_at: Optional[str], ttl_seconds: float) -> bool:
    dt = _parse_iso(fetched_at)
    if dt is None:
        return True
    age = (datetime.now(timezone.utc) - dt).total_seconds()
    return age > ttl_seconds


def load_predictions_cache(settings: Settings) -> Optional[dict[str, Any]]:
    path = _cache_root(settings) / "predictions.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("invalid_predictions_cache", path=str(path))
        return None


def save_predictions_cache(settings: Settings, payload: dict[str, Any]) -> None:
    path = _cache_root(settings) / "predictions.json"
    out = {**payload, "cached_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(out, indent=2))


def load_performance_cache(settings: Settings) -> Optional[dict[str, Any]]:
    path = _cache_root(settings) / "performance.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("invalid_performance_cache", path=str(path))
        return None


def save_performance_cache(settings: Settings, payload: dict[str, Any]) -> None:
    path = _cache_root(settings) / "performance.json"
    out = {**payload, "cached_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(out, indent=2))


def chart_cache_path(settings: Settings, symbol: str) -> Path:
    safe = symbol.upper().replace("^", "").replace("/", "_")
    return _cache_root(settings) / "charts" / f"{safe}.json"


def load_chart_bundle_cache(settings: Settings, symbol: str) -> Optional[dict[str, Any]]:
    path = chart_cache_path(settings, symbol)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("invalid_chart_cache", path=str(path))
        return None


def save_chart_bundle_cache(settings: Settings, symbol: str, bundle: dict[str, Any]) -> None:
    path = chart_cache_path(settings, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {**bundle, "cached_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(out))


def invalidate_chart_cache(settings: Settings, symbol: Optional[str] = None) -> None:
    root = _cache_root(settings) / "charts"
    if not root.exists():
        return
    if symbol is None:
        shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        logger.info("chart_cache_cleared")
        return
    path = chart_cache_path(settings, symbol)
    if path.exists():
        path.unlink()
