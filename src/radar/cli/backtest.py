from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from radar.backtest.expectancy import run_expectancy_backtest
from radar.backtest.gated_signals import (
    apply_gated_signals,
    enrich_predictions_with_panel,
    optimize_gating_params,
    optimize_threshold,
)
from radar.backtest.report import write_report
from radar.config.settings import Settings
from radar.validation.walk_forward import load_oos_predictions

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OOS expectancy backtest")
    parser.add_argument("--config-dir", default="config", help="Config directory")
    parser.add_argument("--report", action="store_true", help="Write JSON/HTML report")
    parser.add_argument("--gated", action="store_true", help="Use high-precision gated signals")
    args = parser.parse_args()

    settings = Settings.load(config_dir=args.config_dir)
    settings.ensure_dirs()

    predictions = load_oos_predictions(settings)

    if args.gated:
        import pandas as pd

        panel_path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
        if panel_path.exists():
            panel = pd.read_parquet(panel_path)
            predictions = enrich_predictions_with_panel(predictions, panel)
        prob_col = "p_ensemble" if "p_ensemble" in predictions.columns else "p_up"
        predictions["p_up"] = predictions.get(prob_col, predictions["p_up"])
        predictions["p_ensemble"] = predictions["p_up"]

        opt_path = Path(settings.paths.processed_dir) / "hybrid_optimized.json"
        threshold = settings.hybrid.min_probability
        hybrid_cfg = settings.hybrid
        if opt_path.exists():
            import json
            saved = json.loads(opt_path.read_text())
            threshold = saved.get("threshold", threshold)
            hybrid_cfg = hybrid_cfg.model_copy(update=saved.get("params", {}))
        elif settings.hybrid.optimize_threshold:
            opt = optimize_gating_params(
                predictions,
                settings.hybrid,
                min_trades=settings.hybrid.min_trades_for_threshold,
            )
            threshold = opt["threshold"]
            hybrid_cfg = settings.hybrid.model_copy(update=opt.get("params", {}))
            print(f"Optimized gated threshold: {threshold:.2f} (hit rate {opt['hit_rate']:.1%}, n={opt['n_trades']})")

        gated = apply_gated_signals(predictions, hybrid_cfg, threshold=threshold)
        predictions = gated.copy()
        predictions["p_up"] = gated["signal"].astype(float)
        settings.backtest.signal_threshold = 0.5
    backtest_results = run_expectancy_backtest(predictions, settings)

    pooled = backtest_results["pooled"]
    print("Pooled OOS Expectancy Backtest:")
    print(f"  Expectancy: {pooled['expectancy']:.6f}")
    print(f"  Win rate:   {pooled['win_rate']:.4f}")
    print(f"  N trades:   {pooled['n_trades']}")
    print(f"  Max DD:     {pooled['max_drawdown']:.4f}")

    for symbol, metrics in backtest_results["by_symbol"].items():
        print(
            f"  {symbol}: E={metrics['expectancy']:.6f} "
            f"trades={metrics['n_trades']} WR={metrics['win_rate']:.4f}"
        )

    if args.report:
        import json
        from radar.models.registry import ModelRegistry

        registry = ModelRegistry(settings.paths.models_dir)
        fold_metrics = []
        for fold_dir in sorted(Path(settings.paths.models_dir).glob("fold_*")):
            manifest_path = fold_dir / "feature_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                fold_metrics.append({"fold_id": manifest["fold_id"], **manifest["metrics"]})

        json_path, html_path = write_report(backtest_results, fold_metrics, settings)
        print(f"Report written: {json_path}")
        print(f"Report written: {html_path}")

    sys.exit(0)


if __name__ == "__main__":
    main()
