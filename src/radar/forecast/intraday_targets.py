"""Shared loaders for chart / intraday modules."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from radar.config.settings import Settings


def load_oos_scores(settings: Settings) -> Optional[pd.DataFrame]:
    path = Path(settings.paths.processed_dir) / "ensemble_oos.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df
