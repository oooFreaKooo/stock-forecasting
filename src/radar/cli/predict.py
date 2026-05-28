from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from radar.config.settings import get_settings
from radar.forecast.chart import render_prediction_chart
from radar.forecast.hybrid_predictor import evaluate_gated_performance, predict_symbol

structlog.configure(
    processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid prediction + forecast chart")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--symbol", default="AAPL", help="Symbol to predict")
    parser.add_argument("--all", action="store_true", help="Predict all traded symbols")
    parser.add_argument("--eval", action="store_true", help="Compare gated vs simple hit rate on OOS")
    parser.add_argument("--optimize", action="store_true", help="Run full signal optimizer and save config")
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports/charts",
        help="Chart output directory",
    )
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    settings.ensure_dirs()

    if args.optimize:
        metrics = evaluate_gated_performance(settings, save_optimized=True)
        print("\n=== Signal Optimizer Complete ===")
        print(f"  Saved to: data/processed/hybrid_optimized.json")
        print(f"  Best hit rate: {metrics['gated_hit_rate']:.1%} ({metrics['gated_trades']} trades)")
        print(f"  Threshold:     {metrics['threshold_used']:.2f}")
        print(f"  Params:        {metrics.get('optimized_params', {})}")
        sys.exit(0)

    if args.eval:
        metrics = evaluate_gated_performance(settings)
        print("\n=== Signal Performance (OOS) ===")
        print(f"  Simple (>0.55):     {metrics['simple_hit_rate']:.1%}  ({metrics['simple_trades']} trades)")
        print(f"  Gated v1:           {metrics['gated_v1_hit_rate']:.1%}  ({metrics['gated_v1_trades']} trades)")
        print(f"  Gated v2 optimized: {metrics['gated_hit_rate']:.1%}  ({metrics['gated_trades']} trades)")
        print(f"  Coverage:           {metrics['coverage_pct']:.1f}% of all days")
        if metrics.get("optimized_params"):
            print(f"  Best threshold:     {metrics['threshold_used']:.2f}")
            print(f"  Best params:        {metrics['optimized_params']}")
        sys.exit(0)

    symbols = settings.universe.traded if args.all else [args.symbol.upper()]
    out_dir = Path(args.output_dir)

    for symbol in symbols:
        pred = predict_symbol(settings, symbol)
        chart_path = out_dir / f"{symbol}_forecast.png"
        render_prediction_chart(settings, pred, chart_path)

        action = "BUY" if pred.signal else "WAIT"
        print(f"\n{symbol} @ {pred.date.date()}")
        print(f"  Last close:     ${pred.last_close:.2f}")
        print(f"  P(up):          {pred.p_up:.1%}")
        print(f"  Confluence:     {pred.confluence_score:.1%}")
        print(f"  Forecast 1d:    {pred.forecast_return_1d:+.2%}")
        print(f"  Signal:         {action} ({pred.confidence} confidence)")
        print(f"  Chart:          {chart_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
