from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from radar.config.settings import Settings
from radar.validation.walk_forward import run_walk_forward

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train walk-forward LightGBM folds")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    parser.add_argument(
        "--config",
        default=None,
        help="Walk-forward config path (default: config/walkforward.yaml)",
    )
    args = parser.parse_args()

    wf_path = args.config or str(Path(args.config_dir) / "walkforward.yaml")
    settings = Settings.load(config_dir=args.config_dir, walkforward_path=wf_path)
    settings.ensure_dirs()

    result = run_walk_forward(settings)
    print(f"Walk-forward complete: {len(result.splits)} folds, {len(result.oos_predictions)} OOS rows")
    for fm in result.fold_metrics:
        print(
            f"  fold {fm['fold_id']}: AUC={fm.get('auc', float('nan')):.4f} "
            f"Brier={fm.get('brier', float('nan')):.4f}"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
