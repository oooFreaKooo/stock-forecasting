from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def _safe_filename(symbol: str) -> str:
    return symbol.replace("^", "").replace("/", "_").upper()


class ParquetStore:
    """Read/write parquet OHLCV with manifest metadata."""

    def __init__(self, raw_dir: Union[str, Path]) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str) -> Path:
        return self.raw_dir / f"{_safe_filename(symbol)}.parquet"

    def manifest_path(self) -> Path:
        return self.raw_dir / "manifest.json"

    def write(self, symbol: str, df: pd.DataFrame, *, rows_added: Optional[int] = None) -> Path:
        path = self.path_for(symbol)
        df.to_parquet(path, index=False)
        self._update_manifest(symbol, path, len(df), rows_added=rows_added)
        return path

    def merge(self, symbol: str, new_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Append/replace by date; returns merged frame and count of new rows."""
        if new_df.empty:
            return self.read(symbol) if self.exists(symbol) else new_df, 0

        incoming = new_df.copy()
        incoming["date"] = pd.to_datetime(incoming["date"]).dt.normalize()

        if not self.exists(symbol):
            self.write(symbol, incoming, rows_added=len(incoming))
            return incoming, len(incoming)

        existing = self.read(symbol)
        existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
        prior_dates = set(existing["date"].astype("int64"))
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        combined = combined.reset_index(drop=True)
        new_rows = int((~combined["date"].astype("int64").isin(prior_dates)).sum())
        self.write(symbol, combined, rows_added=new_rows)
        return combined, new_rows

    def last_date(self, symbol: str) -> Optional[pd.Timestamp]:
        if not self.exists(symbol):
            return None
        series = self.read(symbol)["date"]
        if series.empty:
            return None
        return pd.Timestamp(series.max()).normalize()

    def read(self, symbol: str) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.exists():
            raise FileNotFoundError(f"No cached data for {symbol} at {path}")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def exists(self, symbol: str) -> bool:
        return self.path_for(symbol).exists()

    def _update_manifest(
        self,
        symbol: str,
        path: Path,
        rows: int,
        *,
        rows_added: Optional[int] = None,
    ) -> None:
        manifest: dict = {}
        manifest_path = self.manifest_path()
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        entry = {
            "path": str(path),
            "rows": rows,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        if rows_added is not None:
            entry["rows_added"] = int(rows_added)
        manifest[_safe_filename(symbol)] = entry
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def read_all(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        return {symbol: self.read(symbol) for symbol in symbols}
