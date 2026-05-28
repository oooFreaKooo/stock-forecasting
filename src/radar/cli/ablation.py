from __future__ import annotations

import argparse
import sys

import structlog

from radar.config.settings import Settings
from radar.validation.ablation import run_ablation, run_sentiment_ablation

structlog.configure(
    processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward ablation studies")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--config", default=None, help="Walk-forward config path")
    parser.add_argument("--sentiment", action="store_true", help="Compare with/without NLP sentiment features")
    args = parser.parse_args()

    from pathlib import Path

    settings = Settings.load(config_dir=args.config_dir)
    settings.ensure_dirs()

    if args.sentiment:
        result = run_sentiment_ablation(settings)
        print(f"Without sentiment AUC: {result.without_sentiment_auc:.4f}")
        print(f"With sentiment AUC: {result.with_sentiment_auc:.4f}")
        print(f"Lift: {result.sentiment_lift:+.4f}")
        sys.exit(0)

    wf_path = args.config or str(Path(args.config_dir) / "walkforward.yaml")
    settings = Settings.load(config_dir=args.config_dir, walkforward_path=wf_path)
    settings.ensure_dirs()

    result = run_ablation(settings, walkforward_path=wf_path)
    print(f"Baseline AUC mean: {result.baseline_auc_mean:.4f}")
    print(f"Full (macro+events) AUC mean: {result.full_auc_mean:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
