from __future__ import annotations

import argparse
import sys

import structlog

from radar.config.settings import Settings
from radar.ensemble.orchestrator import run_ensemble_training

structlog.configure(
    processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ensemble meta-learner")
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    settings = Settings.load(config_dir=args.config_dir)
    settings.ensure_dirs()

    result = run_ensemble_training(settings)
    print(f"Ensemble training complete: AUC={result.metrics.get('auc', float('nan')):.4f}")
    print(f"OOS predictions: {len(result.oos_predictions)} rows")
    sys.exit(0)


if __name__ == "__main__":
    main()
