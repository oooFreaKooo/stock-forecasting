from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from radar.data.store import _safe_filename

INTRADAY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "symbol"]


class IntradayBarStore:
    """Local 5m (or other intraday) OHLCV bars under processed_dir/intraday."""

    def __init__(self, processed_dir: Union[str, Path]) -> None:
        self.root = Path(processed_dir) / "intraday"
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str) -> Path:
        return self.root / f"{_safe_filename(symbol)}.parquet"

    def exists(self, symbol: str) -> bool:
        return self.path_for(symbol).exists()

    def read(self, symbol: str) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.exists():
            return pd.DataFrame(columns=INTRADAY_COLUMNS)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        return df

    def write(self, symbol: str, df: pd.DataFrame) -> Path:
        path = self.path_for(symbol)
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        out.to_parquet(path, index=False)
        self._update_manifest(symbol, path, len(out))
        return path

    def upsert(self, symbol: str, incoming: pd.DataFrame) -> int:
        """Merge bars by timestamp; returns number of new timestamps."""
        if incoming.empty:
            return 0

        frame = incoming.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")

        if not self.exists(symbol):
            self.write(symbol, frame)
            return len(frame)

        existing = self.read(symbol)
        prior = set(existing["date"].astype("int64"))
        combined = pd.concat([existing, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        combined = combined.reset_index(drop=True)
        new_rows = int((~combined["date"].astype("int64").isin(prior)).sum())
        self.write(symbol, combined)
        return new_rows

    def last_bar_time(self, symbol: str) -> Optional[pd.Timestamp]:
        if not self.exists(symbol):
            return None
        series = self.read(symbol)["date"]
        if series.empty:
            return None
        return pd.Timestamp(series.max())

    def is_fresh(self, symbol: str, max_age_minutes: int) -> bool:
        last = self.last_bar_time(symbol)
        if last is None:
            return False
        age = datetime.now(timezone.utc) - last.tz_localize("UTC")
        return age.total_seconds() <= max_age_minutes * 60

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _update_manifest(self, symbol: str, path: Path, rows: int) -> None:
        manifest: dict = {}
        manifest_path = self.manifest_path()
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        manifest[_safe_filename(symbol)] = {
            "path": str(path),
            "rows": rows,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
