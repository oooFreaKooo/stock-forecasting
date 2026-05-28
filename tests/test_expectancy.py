from __future__ import annotations

import numpy as np

from radar.validation.metrics import compute_expectancy, max_drawdown


def test_expectancy_formula():
    returns = np.array([0.02, 0.01, -0.01, -0.02, 0.03])
    result = compute_expectancy(returns)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    pw = len(wins) / len(returns)
    pl = 1 - pw
    aw = wins.mean()
    al = abs(losses.mean())
    expected = (pw * aw) - (pl * al)
    assert abs(result["expectancy"] - expected) < 1e-10
    assert result["n_trades"] == 5


def test_max_drawdown():
    equity = np.array([1.0, 1.1, 1.05, 1.2, 0.9])
    dd = max_drawdown(equity)
    assert dd < 0
