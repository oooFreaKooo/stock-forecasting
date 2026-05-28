from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute AUC and Brier score."""
    metrics: dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    metrics["accuracy"] = float(((y_prob >= 0.5).astype(int) == y_true).mean())
    return metrics


def compute_expectancy(
    returns: np.ndarray,
    *,
    wins: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Compute expectancy E = (Pw * Aw) - (Pl * Al).

    If wins not provided, infer from return sign.
    """
    if len(returns) == 0:
        return {
            "expectancy": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "n_trades": 0,
        }

    if wins is None:
        wins = returns > 0

    loss_returns = returns[~wins]
    win_returns = returns[wins]

    pw = wins.mean() if len(wins) else 0.0
    pl = 1.0 - pw
    aw = float(win_returns.mean()) if len(win_returns) else 0.0
    al = float(abs(loss_returns.mean())) if len(loss_returns) else 0.0

    expectancy = (pw * aw) - (pl * al)
    gross_profit = win_returns.sum() if len(win_returns) else 0.0
    gross_loss = abs(loss_returns.sum()) if len(loss_returns) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "expectancy": float(expectancy),
        "win_rate": float(pw),
        "avg_win": float(aw),
        "avg_loss": float(al),
        "profit_factor": float(profit_factor),
        "n_trades": int(len(returns)),
    }


def max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(dd.min())
