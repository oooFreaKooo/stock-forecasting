from __future__ import annotations

import threading
import time
from typing import Optional

import structlog

from radar.config.settings import Settings
from radar.jobs.runner import run_daily_job, run_intraday_job, run_news_job, run_startup_bootstrap

logger = structlog.get_logger(__name__)


class BackgroundJobRunner:
    """Poll loop for incremental news, intraday bars, and daily features."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_run: dict[str, float] = {"news": 0.0, "intraday": 0.0, "daily": 0.0}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="radar-background-jobs", daemon=True)
        self._thread.start()
        if self.settings.jobs.enabled and self.settings.jobs.bootstrap_on_startup:
            threading.Thread(
                target=run_startup_bootstrap,
                args=(self.settings,),
                name="radar-startup-bootstrap",
                daemon=True,
            ).start()
        logger.info("background_jobs_started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("background_jobs_stopped")

    def _loop(self) -> None:
        cfg = self.settings.jobs
        while not self._stop.is_set():
            if cfg.enabled:
                now = time.monotonic()
                try:
                    if now - self._last_run["news"] >= cfg.news_interval_minutes * 60:
                        run_news_job(self.settings)
                        self._last_run["news"] = now
                    if now - self._last_run["intraday"] >= cfg.intraday_interval_minutes * 60:
                        run_intraday_job(self.settings)
                        self._last_run["intraday"] = now
                    if now - self._last_run["daily"] >= cfg.daily_interval_minutes * 60:
                        run_daily_job(self.settings)
                        self._last_run["daily"] = now
                except Exception as exc:
                    logger.exception("background_job_failed", error=str(exc))
            self._stop.wait(cfg.poll_seconds)
