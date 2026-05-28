from __future__ import annotations

import argparse
import sys

import structlog

from radar.config.settings import get_settings
from radar.ensemble.orchestrator import run_full_pipeline

structlog.configure(
    processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Hybrid AI Investment Radar pipeline")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--full", action="store_true", help="Run complete pipeline including RL")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL training step")
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    settings.ensure_dirs()

    results = run_full_pipeline(settings, skip_rl=args.skip_rl or not args.full)
    for step, status in results.items():
        print(f"  {step}: {status}")
    sys.exit(0)


if __name__ == "__main__":
    main()
