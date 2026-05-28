"""Alpha Vantage REST client (compact time series for trend forecasts)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
_REQUEST_GAP_SECONDS = 1.05
_last_request_at = 0.0
_ENV_LOADED = False

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _repo_search_paths() -> list[Path]:
    paths: list[Path] = []
    env_root = os.environ.get("RADAR_ROOT")
    if env_root:
        paths.append(Path(env_root).expanduser().resolve())
    paths.append(_REPO_ROOT)
    paths.append(Path.cwd().resolve())
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _load_project_env() -> None:
    """Load repo ``.env`` into os.environ (API often starts without shell sourcing)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    for base in _repo_search_paths():
        path = base / ".env"
        if not path.is_file():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if not key or key in os.environ:
                    continue
                val = value.strip().strip('"').strip("'")
                if val:
                    os.environ[key] = val
        except OSError as exc:
            logger.warning("env_file_read_failed", path=str(path), error=str(exc))
        break


def api_key() -> Optional[str]:
    _load_project_env()
    return os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get("ALPHA_VANTAGE_API_KEY")


def is_configured() -> bool:
    return bool(api_key())


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _REQUEST_GAP_SECONDS:
        time.sleep(_REQUEST_GAP_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def query(params: dict[str, Any]) -> Optional[dict[str, Any]]:
    key = api_key()
    if not key:
        logger.warning("alphavantage_missing_api_key")
        return None
    payload = {**params, "apikey": key}
    url = f"{BASE_URL}?{urlencode(payload)}"
    _throttle()
    try:
        with urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("alphavantage_request_failed", error=str(exc))
        return None

    if not isinstance(data, dict):
        return None
    if "Note" in data or "Information" in data:
        logger.warning("alphavantage_rate_limited", message=data.get("Note") or data.get("Information"))
        return None
    if "Error Message" in data:
        logger.warning("alphavantage_error", message=data["Error Message"])
        return None
    return data


def _series_key(payload: dict[str, Any]) -> Optional[str]:
    for key in payload:
        if key.startswith("Time Series"):
            return key
    return None


def parse_time_series(payload: dict[str, Any]) -> pd.Series:
    """Return close prices indexed by UTC-naive timestamps, oldest first."""
    key = _series_key(payload)
    if not key:
        return pd.Series(dtype=float)
    block = payload.get(key) or {}
    rows: list[tuple[pd.Timestamp, float]] = []
    for ts_str, bar in block.items():
        if not isinstance(bar, dict):
            continue
        close = bar.get("4. close") or bar.get("5. adjusted close") or bar.get("close")
        if close is None:
            continue
        ts = pd.Timestamp(ts_str)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        rows.append((ts, float(close)))
    if not rows:
        return pd.Series(dtype=float)
    rows.sort(key=lambda r: r[0])
    idx, vals = zip(*rows)
    return pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float)


def _daily_cache_path(symbol: str) -> Path:
    return _REPO_ROOT / "data" / "processed" / "cache" / f"alphavantage_{symbol.upper()}_daily.json"


def fetch_daily_closes(symbol: str, *, adjusted: bool = False) -> pd.Series:
    """Daily OHLC (free tier). Adjusted series requires premium."""
    cache_path = _daily_cache_path(symbol)
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            fetched = cached.get("fetched_at")
            if fetched and not is_stale(fetched, 6 * 3600):
                rows = cached.get("closes") or []
                if rows:
                    idx = pd.DatetimeIndex([r["date"] for r in rows])
                    vals = [float(r["close"]) for r in rows]
                    return pd.Series(vals, index=idx, dtype=float)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    fn = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
    payload = query({
        "function": fn,
        "symbol": symbol.upper(),
        "outputsize": "compact",
        "datatype": "json",
    })
    if payload is None and adjusted:
        payload = query({
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "outputsize": "compact",
            "datatype": "json",
        })
    if payload is None:
        return pd.Series(dtype=float)
    series = parse_time_series(payload)
    if not series.empty:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload_out = {
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "closes": [
                    {"date": ts.strftime("%Y-%m-%d"), "close": float(val)}
                    for ts, val in series.items()
                ],
            }
            cache_path.write_text(json.dumps(payload_out), encoding="utf-8")
        except OSError:
            pass
    return series


def is_stale(fetched_at: str, ttl_seconds: float) -> bool:
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()
        return age > ttl_seconds
    except ValueError:
        return True
