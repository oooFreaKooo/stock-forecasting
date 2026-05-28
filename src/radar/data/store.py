from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

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

    def write(self, symbol: str, df: pd.DataFrame) -> Path:
        path = self.path_for(symbol)
        df.to_parquet(path, index=False)
        self._update_manifest(symbol, path, len(df))
        return path

    def read(self, symbol: str) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.exists():
            raise FileNotFoundError(f"No cached data for {symbol} at {path}")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def exists(self, symbol: str) -> bool:
        return self.path_for(symbol).exists()

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

    def read_all(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        return {symbol: self.read(symbol) for symbol in symbols}
