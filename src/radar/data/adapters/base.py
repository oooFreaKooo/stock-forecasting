from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataSource(ABC):
    """Pluggable market data source."""

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: str,
        end: Optional[str],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame with columns: date, open, high, low, close, volume, symbol."""

    @abstractmethod
    def fetch_many(
        self,
        symbols: list[str],
        start: str,
        end: Optional[str],
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple symbols."""
