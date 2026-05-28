from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt_geo_daily(
    lookback_days: int = 30,
    query: str = "geopolitical conflict sanctions war",
) -> pd.DataFrame:
    """
    Pull recent geopolitical news volume from GDELT DOC API (free, no key).

    Returns daily geo_risk_flag and conflict_intensity proxies.
    """
    try:
        import urllib.parse
        import urllib.request
        import json
    except ImportError:
        return pd.DataFrame(columns=["date", "geo_risk_flag", "conflict_intensity"])

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    params = urllib.parse.urlencode({
        "query": query,
        "mode": "timelinevol",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "timelinesmooth": "0",
    })
    url = f"{GDELT_DOC_API}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("gdelt_live_fetch_failed", error=str(exc))
        return pd.DataFrame(columns=["date", "geo_risk_flag", "conflict_intensity"])

    timeline = payload.get("timeline") or []
    if not timeline:
        return pd.DataFrame(columns=["date", "geo_risk_flag", "conflict_intensity"])

    rows: list[dict] = []
    for bucket in timeline:
        for point in bucket.get("data", []):
            date_str = point.get("date", "")
            if len(date_str) < 8:
                continue
            dt = pd.Timestamp(date_str[:8], tz="UTC").tz_localize(None)
            volume = float(point.get("value", 0))
            rows.append({"date": dt, "volume": volume})

    if not rows:
        return pd.DataFrame(columns=["date", "geo_risk_flag", "conflict_intensity"])

    df = pd.DataFrame(rows).groupby("date", as_index=False)["volume"].sum()
    vol_max = df["volume"].max() or 1.0
    df["geo_risk_flag"] = (df["volume"] / vol_max).clip(0, 1)
    df["conflict_intensity"] = df["volume"]
    return df[["date", "geo_risk_flag", "conflict_intensity"]].sort_values("date")
