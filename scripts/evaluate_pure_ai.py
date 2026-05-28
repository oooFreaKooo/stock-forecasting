#!/usr/bin/env python3
"""Walk-forward evaluation: intraday LGBM vs actual 5m bars."""

from __future__ import annotations

import argparse

import pandas as pd
import yfinance as yf

from radar.config.settings import get_settings
from radar.forecast.chart_eval import evaluate_intraday_backtest, normalize_yfinance_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate intraday LGBM walk-forward")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--period-5m", default="5d")
    args = parser.parse_args()

    settings = get_settings(args.config_dir)
    symbols = list(settings.universe.traded)
    rows: list[dict] = []

    for sym in symbols:
        df = yf.Ticker(sym).history(
            period=args.period_5m,
            interval="5m",
            auto_adjust=True,
            prepost=True,
        )
        frame = normalize_yfinance_frame(df, sym, "5m")
        if len(frame) < 50:
            continue

        metrics = evaluate_intraday_backtest(
            frame, symbol=sym, interval="5m", settings=settings
        )
        rows.append(metrics.as_dict())

    if not rows:
        raise SystemExit("No evaluation rows produced.")

    out = pd.DataFrame(rows).sort_values("symbol")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nAggregate (mean)")
    print(
        out[["n_points", "mae", "direction_accuracy", "vol_ratio", "rmse"]]
        .mean()
        .to_string(float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
