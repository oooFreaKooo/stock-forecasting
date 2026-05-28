from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from radar.config.schemas import RLConfig
from radar.rl.rewards.risk_adjusted import RewardContext, compute_risk_adjusted_reward, rolling_sortino_bonus


class RadarSizingEnv(gym.Env):
    """
    RL sizing environment — uses Layer 1 probabilities + portfolio state only.

    State: [p_up, p_down, vol_regime_onehot(3), setup_quality, exposure,
            unrealized_pnl, rolling_drawdown, days_in_position, regime_similarity]
    Action: MultiDiscrete [position_bucket, stop_atr_bucket]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_df: pd.DataFrame,
        rl_config: RLConfig,
        transaction_cost_bps: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.episode_df = episode_df.reset_index(drop=True)
        self.rl_config = rl_config
        self.transaction_cost = transaction_cost_bps / 10_000
        self.position_buckets = np.array(rl_config.position_buckets, dtype=float)
        self.stop_buckets = np.array(rl_config.stop_atr_buckets, dtype=float)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([
            len(self.position_buckets),
            len(self.stop_buckets),
        ])

        self._step_idx = 0
        self._position = 0.0
        self._entry_price = 0.0
        self._days_in_position = 0
        self._equity = 1.0
        self._equity_curve: list[float] = [1.0]
        self._return_history: list[float] = []

        if seed is not None:
            self.reset(seed=seed)

    def _vol_regime_onehot(self, row: pd.Series) -> np.ndarray:
        regime = int(row.get("y_vol_regime", 1))
        regime = max(0, min(2, regime))
        onehot = np.zeros(3, dtype=np.float32)
        onehot[regime] = 1.0
        return onehot

    def _build_observation(self, row: pd.Series) -> np.ndarray:
        vol_onehot = self._vol_regime_onehot(row)
        setup_quality = float(row.get("regime_neighbor_win_rate", 0.5))
        regime_sim = float(row.get("regime_sim_mean", 0.0))

        unrealized = 0.0
        if self._position > 0 and self._entry_price > 0:
            unrealized = (float(row["close"]) - self._entry_price) / self._entry_price

        rolling_dd = self._current_drawdown()

        obs = np.array([
            float(row["p_up"]),
            float(row["p_down"]),
            vol_onehot[0],
            vol_onehot[1],
            vol_onehot[2],
            setup_quality,
            self._position,
            unrealized,
            rolling_dd,
            min(self._days_in_position / 20.0, 1.0),
            regime_sim,
        ], dtype=np.float32)
        return obs

    def _current_drawdown(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        curve = np.array(self._equity_curve)
        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / peak
        return float(dd[-1])

    def _rolling_vol(self) -> float:
        if len(self._return_history) < 2:
            return 0.0
        window = self.rl_config.reward.sortino_window
        recent = self._return_history[-window:]
        return float(np.std(recent))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_idx = 0
        self._position = 0.0
        self._entry_price = 0.0
        self._days_in_position = 0
        self._equity = 1.0
        self._equity_curve = [1.0]
        self._return_history = []

        row = self.episode_df.iloc[0]
        if "close" not in self.episode_df.columns:
            self.episode_df = self.episode_df.copy()
            self.episode_df["close"] = 1.0

        return self._build_observation(row), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        row = self.episode_df.iloc[self._step_idx]
        target_position = float(self.position_buckets[int(action[0])])
        stop_mult = float(self.stop_buckets[int(action[1])])

        if int(row.get("is_event_day", 0)) == 1:
            target_position = min(target_position, self.rl_config.max_size_on_event_day)

        prev_position = self._position
        delta_position = target_position - prev_position

        next_return = float(row["next_return"])
        if np.isnan(next_return):
            next_return = 0.0
        atr_pct = float(row.get("atr_pct", self.rl_config.atr_proxy_pct))
        stop_level = -atr_pct * stop_mult

        if target_position > 0 and next_return < stop_level:
            realized_return = stop_level
        else:
            realized_return = next_return

        daily_return = prev_position * realized_return
        daily_return -= self.transaction_cost * abs(delta_position)

        self._equity *= 1.0 + daily_return
        self._equity_curve.append(self._equity)
        self._return_history.append(daily_return)

        self._position = target_position
        if target_position > 0:
            if prev_position == 0:
                self._entry_price = float(row.get("close", 1.0))
                self._days_in_position = 1
            else:
                self._days_in_position += 1
        else:
            self._entry_price = 0.0
            self._days_in_position = 0

        sortino_bonus = rolling_sortino_bonus(
            np.array(self._return_history),
            self.rl_config.reward.sortino_window,
        )
        reward = compute_risk_adjusted_reward(
            RewardContext(
                daily_return=daily_return,
                drawdown=abs(self._current_drawdown()),
                rolling_vol=self._rolling_vol(),
                delta_position=delta_position,
                sortino_bonus=sortino_bonus,
            ),
            self.rl_config.reward,
        )

        self._step_idx += 1
        terminated = self._step_idx >= len(self.episode_df) - 1
        truncated = False

        if terminated:
            obs = self._build_observation(self.episode_df.iloc[-1])
        else:
            obs = self._build_observation(self.episode_df.iloc[self._step_idx])

        info = {
            "daily_return": daily_return,
            "position": self._position,
            "equity": self._equity,
        }
        return obs, float(reward), terminated, truncated, info
