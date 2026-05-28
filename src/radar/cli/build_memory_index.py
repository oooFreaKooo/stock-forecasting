from __future__ import annotations

import argparse
import sys

import structlog

from radar.config.settings import get_settings
from radar.features.pipeline import load_feature_panel
from radar.memory.retrieval import (
    build_and_persist_regime_vectors,
    enrich_panel_with_memory,
    save_enriched_panel,
)

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build semantic memory index and enrich feature panel"
    )
    parser.add_argument("--config-dir", default="config", help="Config directory")
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Only build regime vectors and ChromaDB index, skip panel enrichment",
    )
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    settings.ensure_dirs()

    regime_df = build_and_persist_regime_vectors(settings)
    print(f"Built regime vectors: {len(regime_df)} daily rows")

    if args.skip_enrich:
        sys.exit(0)

    panel = load_feature_panel(settings)
    enriched = enrich_panel_with_memory(panel, settings)
    save_enriched_panel(enriched, settings)
    print(f"Enriched feature panel: {len(enriched)} rows with memory features")
    sys.exit(0)


if __name__ == "__main__":
    main()
