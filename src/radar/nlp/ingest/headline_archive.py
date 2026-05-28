from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

ARCHIVE_NAME = "headlines_archive.parquet"


def archive_path(processed_dir: str) -> Path:
    return Path(processed_dir) / ARCHIVE_NAME


def load_headline_archive(processed_dir: str) -> pd.DataFrame:
    path = archive_path(processed_dir)
    if not path.exists():
        return pd.DataFrame(columns=["date", "symbol", "title", "published"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def append_headline_archive(processed_dir: str, headlines: pd.DataFrame) -> pd.DataFrame:
    """Persist unique headlines for historical sentiment backfill."""
    if headlines.empty:
        return load_headline_archive(processed_dir)

    incoming = headlines.copy()
    incoming["date"] = pd.to_datetime(incoming["date"]).dt.normalize()
    incoming["title"] = incoming["title"].fillna("").astype(str).str.strip()
    incoming = incoming[incoming["title"] != ""]

    existing = load_headline_archive(processed_dir)
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined["_key"] = (
        combined["date"].astype(str)
        + "|"
        + combined["symbol"].astype(str)
        + "|"
        + combined["title"].str.lower()
    )
    combined = combined.drop_duplicates(subset=["_key"], keep="last").drop(columns=["_key"])
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)

    path = archive_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    logger.info("headline_archive_updated", rows=len(combined))
    return combined
