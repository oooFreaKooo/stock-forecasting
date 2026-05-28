from __future__ import annotations

import structlog

from radar.config.settings import Settings
from radar.data.adapters.base import DataSource
from radar.data.adapters.yfinance_source import AlpacaSource, YFinanceSource
from radar.data.store import ParquetStore
from radar.data.validators import validate_ohlcv

logger = structlog.get_logger(__name__)


def get_data_source(source_name: str) -> DataSource:
    if source_name == "yfinance":
        return YFinanceSource()
    if source_name == "alpaca":
        return AlpacaSource()
    raise ValueError(f"Unknown data source: {source_name}")


def fetch_and_store(settings: Settings) -> dict[str, int]:
    """Fetch all symbols and persist to parquet."""
    settings.ensure_dirs()
    source = get_data_source(settings.data.source)
    store = ParquetStore(settings.paths.raw_dir)

    symbols = settings.all_symbols
    frames = source.fetch_many(
        symbols=symbols,
        start=settings.data.start_date,
        end=settings.data.end_date,
        interval=settings.data.interval,
    )

    results: dict[str, int] = {}
    for symbol, df in frames.items():
        validated = validate_ohlcv(df, symbol)
        store.write(symbol, validated)
        results[symbol] = len(validated)
        logger.info("stored_symbol", symbol=symbol, rows=len(validated))

    return results
