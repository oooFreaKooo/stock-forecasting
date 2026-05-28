from __future__ import annotations

import argparse
import sys

import structlog

from radar.config.settings import get_settings
from radar.data.fetcher import fetch_and_store

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch market data via yfinance")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    results = fetch_and_store(settings)
    print(f"Fetched {len(results)} symbols:")
    for symbol, rows in results.items():
        print(f"  {symbol}: {rows} rows")
    sys.exit(0)


if __name__ == "__main__":
    main()
