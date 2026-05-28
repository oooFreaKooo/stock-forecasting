from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from radar.config.settings import get_settings
from radar.nlp.fusion.memory_enricher import build_sentiment_panel

structlog.configure(
    processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NLP sentiment index and enrich panel")
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    if not settings.nlp.enabled:
        print("NLP disabled in config")
        sys.exit(0)

    settings.ensure_dirs()
    panel = build_sentiment_panel(settings)

    path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
    panel.to_parquet(path, index=False)
    print(f"Sentiment-enriched panel: {len(panel)} rows -> {path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
