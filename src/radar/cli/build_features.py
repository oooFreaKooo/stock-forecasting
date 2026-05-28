from __future__ import annotations

import argparse
import sys

import structlog

from radar.config.settings import get_settings
from radar.features.pipeline import build_feature_panel, enrich_memory_if_available

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature panel from raw data")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    parser.add_argument(
        "--with-memory",
        action="store_true",
        help="Enrich with memory features if regime index exists",
    )
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    settings.ensure_dirs()
    panel = build_feature_panel(settings)

    if args.with_memory:
        enriched = enrich_memory_if_available(settings)
        if enriched is not None:
            panel = enriched
            print("Memory features attached.")
        else:
            print("Memory index not found — run build_memory_index first.")

    print(f"Built feature panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols")
    sys.exit(0)


if __name__ == "__main__":
    main()
