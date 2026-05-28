from __future__ import annotations

import argparse
import sys

import structlog

from radar.config.settings import get_settings
from radar.rl.train_sizing import train_sizing_policy

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL sizing policy on OOS predictions")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    if not settings.rl.enabled:
        print("RL is disabled in config. Set rl.enabled: true")
        sys.exit(1)

    result = train_sizing_policy(settings)
    print(f"RL training complete: {result['algorithm']} saved to {result['model_path']}")
    print(f"  Train rows: {result['train_rows']}, timesteps: {result['timesteps']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
