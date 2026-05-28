from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import pandas as pd
import structlog
import yfinance as yf

from radar.data.adapters.base import DataSource

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume", "symbol"]


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("^", "").upper()


def _normalize_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["symbol"] = _normalize_symbol(symbol)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[OHLCV_COLUMNS].drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    return df


class YFinanceSource(DataSource):
    """yfinance-backed data source."""

    def __init__(self, retry_count: int = 3, retry_delay: float = 1.0) -> None:
        self.retry_count = retry_count
        self.retry_delay = retry_delay

    def fetch(
        self,
        symbol: str,
        start: str,
        end: Optional[str],
        interval: str = "1d",
    ) -> pd.DataFrame:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retry_count + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
                result = _normalize_frame(df, symbol)
                logger.info("fetched_symbol", symbol=symbol, rows=len(result))
                return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "fetch_retry",
                    symbol=symbol,
                    attempt=attempt,
                    error=str(exc),
                )
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"Failed to fetch {symbol} after {self.retry_count} attempts") from last_error

    def fetch_many(
        self,
        symbols: list[str],
        start: str,
        end: Optional[str],
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        return {
            symbol: self.fetch(symbol, start=start, end=end, interval=interval)
            for symbol in symbols
        }


class AlpacaSource(DataSource):
    """Stub for future Alpaca integration (Phase 4+)."""

    def fetch(
        self,
        symbol: str,
        start: str,
        end: Optional[str],
        interval: str = "1d",
    ) -> pd.DataFrame:
        raise NotImplementedError("AlpacaSource is a Phase 4+ stub. Use YFinanceSource.")

    def fetch_many(
        self,
        symbols: list[str],
        start: str,
        end: Optional[str],
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError("AlpacaSource is a Phase 4+ stub. Use YFinanceSource.")
