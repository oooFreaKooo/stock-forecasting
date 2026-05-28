from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from radar.config.schemas import RLRewardConfig


@dataclass
class RewardContext:
    daily_return: float
    drawdown: float
    rolling_vol: float
    delta_position: float
    sortino_bonus: float = 0.0


def compute_risk_adjusted_reward(
    ctx: RewardContext,
    config: RLRewardConfig,
) -> float:
    """
    Risk-adjusted reward penalizing drawdown, volatility, and overtrading.

    reward = daily_return
             - lambda_dd * max(0, drawdown - dd_threshold)
             - lambda_vol * rolling_vol
             - lambda_turn * |delta_position|
             + lambda_sortino * sortino_bonus
    """
    penalty_dd = config.lambda_dd * max(0.0, ctx.drawdown - config.dd_threshold)
    penalty_vol = config.lambda_vol * ctx.rolling_vol
    penalty_turn = config.lambda_turn * abs(ctx.delta_position)
    bonus = config.lambda_sortino * ctx.sortino_bonus

    return (
        ctx.daily_return
        - penalty_dd
        - penalty_vol
        - penalty_turn
        + bonus
    )


def rolling_sortino_bonus(returns: np.ndarray, window: int) -> float:
    """Monthly-style Sortino bonus from recent daily returns."""
    if len(returns) < 2:
        return 0.0
    recent = returns[-window:]
    downside = recent[recent < 0]
    if len(downside) == 0:
        return float(recent.mean()) if len(recent) else 0.0
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return float(recent.mean() / downside_std)
