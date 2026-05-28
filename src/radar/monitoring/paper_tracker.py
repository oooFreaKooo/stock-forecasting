from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

PAPER_LOG = "paper_trades.jsonl"


def paper_log_path(processed_dir: str) -> Path:
    return Path(processed_dir) / PAPER_LOG


def log_paper_signal(processed_dir: str, prediction: dict[str, Any]) -> None:
    """Append a paper-trading signal snapshot for forward monitoring."""
    path = paper_log_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "symbol": prediction.get("symbol"),
        "signal": prediction.get("signal"),
        "p_up": prediction.get("p_up"),
        "confluence_score": prediction.get("confluence_score"),
        "forecast_return_1d": prediction.get("forecast_return_1d"),
        "entry_quality": prediction.get("entry_quality"),
        "position_size": prediction.get("position_size"),
        "last_close": prediction.get("last_close"),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def evaluate_paper_trades(processed_dir: str, raw_dir: str) -> dict[str, Any]:
    """
    Compare logged BUY signals against subsequent daily closes.

    Returns hit rate and average forward return for resolved trades.
    """
    path = paper_log_path(processed_dir)
    if not path.exists():
        return {"n_logged": 0, "n_resolved": 0, "hit_rate": 0.0, "avg_return": 0.0}

    from radar.data.store import ParquetStore

    store = ParquetStore(raw_dir)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not rows:
        return {"n_logged": 0, "n_resolved": 0, "hit_rate": 0.0, "avg_return": 0.0}

    returns: list[float] = []
    hits = 0
    resolved = 0
    for row in rows:
        if row.get("signal") != 1:
            continue
        symbol = row.get("symbol")
        if not symbol or not store.exists(symbol):
            continue
        logged_at = pd.Timestamp(row["logged_at"]).normalize()
        prices = store.read(symbol)
        prices["date"] = pd.to_datetime(prices["date"])
        future = prices[prices["date"] > logged_at].sort_values("date")
        if len(future) < 1:
            continue
        entry = float(row.get("last_close") or prices[prices["date"] <= logged_at]["close"].iloc[-1])
        nxt = float(future.iloc[0]["close"])
        ret = (nxt - entry) / entry
        returns.append(ret)
        resolved += 1
        if ret > 0:
            hits += 1

    hit_rate = hits / resolved if resolved else 0.0
    avg_return = float(sum(returns) / len(returns)) if returns else 0.0
    return {
        "n_logged": len(rows),
        "n_resolved": resolved,
        "hit_rate": hit_rate,
        "avg_return": avg_return,
    }
