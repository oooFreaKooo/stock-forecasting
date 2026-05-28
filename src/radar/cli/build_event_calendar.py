from __future__ import annotations

import argparse
import sys

import structlog

from radar.config.settings import get_settings
from radar.events.calendar_builder import build_event_calendar

structlog.configure(
    processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build macro event calendar")
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    settings.ensure_dirs()
    cal = build_event_calendar(settings)
    print(f"Built event calendar: {len(cal)} days, {cal['is_event_day'].sum()} event days")
    sys.exit(0)


if __name__ == "__main__":
    main()
