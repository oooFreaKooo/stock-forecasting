from __future__ import annotations

import signal
import sys

import structlog

from radar.config.settings import get_settings
from radar.jobs.scheduler import BackgroundJobRunner

logger = structlog.get_logger(__name__)


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()

    if not settings.jobs.enabled:
        logger.error("jobs_disabled_in_config")
        sys.exit(1)

    runner = BackgroundJobRunner(settings)
    runner.start()
    logger.info(
        "scheduler_running",
        news_min=settings.jobs.news_interval_minutes,
        intraday_min=settings.jobs.intraday_interval_minutes,
        daily_min=settings.jobs.daily_interval_minutes,
    )

    def _shutdown(*_args: object) -> None:
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        signal.pause()


if __name__ == "__main__":
    main()
