from __future__ import annotations

from typing import Optional

import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.data.fetcher import get_data_source
from radar.data.intraday_store import IntradayBarStore
from radar.data.store import ParquetStore
from radar.data.validators import validate_ohlcv

logger = structlog.get_logger(__name__)


def _end_date(settings: Settings) -> str:
    return settings.data.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")


def fetch_daily_incremental(settings: Settings) -> dict[str, int]:
    """Append only new daily bars since the last stored date."""
    settings.ensure_dirs()
    source = get_data_source(settings.data.source)
    store = ParquetStore(settings.paths.raw_dir)
    end = _end_date(settings)
    interval = settings.data.interval

    results: dict[str, int] = {}
    for symbol in settings.all_symbols:
        last = store.last_date(symbol)
        if last is None:
            df = source.fetch(
                symbol,
                start=settings.data.start_date,
                end=end,
                interval=interval,
            )
            validated = validate_ohlcv(df, symbol)
            store.write(symbol, validated, rows_added=len(validated))
            results[symbol] = len(validated)
            logger.info("stored_symbol_full", symbol=symbol, rows=len(validated))
            continue

        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if pd.Timestamp(start) > pd.Timestamp(end):
            results[symbol] = len(store.read(symbol))
            logger.info("symbol_up_to_date", symbol=symbol, rows=results[symbol])
            continue

        df = source.fetch(symbol, start=start, end=end, interval=interval)
        if df.empty:
            results[symbol] = len(store.read(symbol))
            logger.info("symbol_no_new_rows", symbol=symbol, rows=results[symbol])
            continue

        validated = validate_ohlcv(df, symbol)
        merged, added = store.merge(symbol, validated)
        results[symbol] = len(merged)
        logger.info("stored_symbol_incremental", symbol=symbol, rows=len(merged), added=added)

    return results


def fetch_intraday_incremental(
    settings: Settings,
    *,
    symbols: Optional[list[str]] = None,
    period: Optional[str] = None,
) -> dict[str, int]:
    """Refresh local 5m bars from yfinance (period window, merged into store)."""
    settings.ensure_dirs()
    from radar.data.adapters.yfinance_source import YFinanceSource

    period = period or settings.jobs.intraday_period
    store = IntradayBarStore(settings.paths.processed_dir)
    source = YFinanceSource()
    target_symbols = symbols or list(settings.universe.traded)

    results: dict[str, int] = {}
    for symbol in target_symbols:
        raw = source.fetch_period(symbol, period=period, interval="5m", prepost=True)
        if raw.empty:
            results[symbol] = store.read(symbol).shape[0] if store.exists(symbol) else 0
            continue
        added = store.upsert(symbol, raw)
        results[symbol] = len(store.read(symbol))
        logger.info("intraday_upsert", symbol=symbol, rows=results[symbol], added=added)

    return results
