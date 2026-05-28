from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FREDSource(ABC):
    """FRED API adapter stub for Phase 7+ paid/rate-limited usage."""

    @abstractmethod
    def fetch_series(self, series_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        """Return DataFrame with date, value columns."""


class FREDSourceStub(FREDSource):
    """Stub — use yfinance macro proxies (^TNX, ^IRX) until FRED API key configured."""

    def fetch_series(self, series_id: str, start: str, end: str | None = None) -> pd.DataFrame:
        raise NotImplementedError(
            f"FREDSource stub: set FRED_API_KEY and implement adapter for {series_id}. "
            "Use yfinance macro symbols (^TNX, ^IRX, HYG, LQD) via fetch_data instead."
        )
