from __future__ import annotations

import argparse
import sys

import structlog

from radar.config.settings import get_settings
from radar.rl.evaluate_policy import evaluate_sizing_policy

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL sizing policy on held-out OOS data")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    results = evaluate_sizing_policy(settings)

    pooled = results["pooled"]
    print("RL Policy Evaluation (held-out OOS):")
    print(f"  Total return:  {pooled['total_return']:.4f}")
    print(f"  Expectancy:    {pooled['expectancy']:.6f}")
    print(f"  Sharpe (ann.): {pooled['sharpe']:.4f}")
    print(f"  Max drawdown:  {pooled['max_drawdown']:.4f}")
    print(f"  Win rate:      {pooled['win_rate']:.4f}")

    for symbol, metrics in results["by_symbol"].items():
        print(
            f"  {symbol}: return={metrics['total_return']:.4f} "
            f"sharpe={metrics['sharpe']:.4f} max_dd={metrics['max_drawdown']:.4f}"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
