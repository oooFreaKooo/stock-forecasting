from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from radar.backtest.signal_rules import apply_signal_rule
from radar.config.settings import Settings
from radar.validation.metrics import compute_expectancy, max_drawdown


def run_expectancy_backtest(
    predictions: pd.DataFrame,
    settings: Settings,
) -> dict[str, Any]:
    """
    Run expectancy backtest on OOS predictions.

    E = (Pw * Aw) - (Pl * Al) on signaled trades.
    """
    df = predictions.copy()
    if "signal" not in df.columns:
        df = apply_signal_rule(df, threshold=settings.backtest.signal_threshold)
    cost = settings.model.transaction_cost_bps / 10_000

    results: dict[str, Any] = {"by_symbol": {}, "pooled": {}, "by_fold": {}}

    for symbol, group in df.groupby("symbol"):
        results["by_symbol"][symbol] = _compute_symbol_metrics(group, cost)

    results["pooled"] = _compute_symbol_metrics(df, cost)

    if "fold_id" in df.columns:
        for fold_id, group in df.groupby("fold_id"):
            results["by_fold"][int(fold_id)] = _compute_symbol_metrics(group, cost)

    return results


def _compute_symbol_metrics(group: pd.DataFrame, cost: float) -> dict[str, float]:
    signaled = group[group["signal"] == 1].copy()
    if signaled.empty:
        return {
            "expectancy": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "n_trades": 0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }

    trade_returns = signaled["next_return"].values - cost
    exp = compute_expectancy(trade_returns)

    hit_rate = ((signaled["y_direction"] == 1) & (signaled["next_return"] > 0)).mean()
    equity = np.cumprod(1 + trade_returns)
    total_return = equity[-1] - 1 if len(equity) else 0.0

    return {
        **exp,
        "hit_rate": float(hit_rate),
        "max_drawdown": max_drawdown(equity),
        "total_return": float(total_return),
    }
