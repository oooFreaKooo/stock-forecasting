from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.config.schemas import RLConfig, RLRewardConfig
from radar.rl.envs.sizing_env import RadarSizingEnv
from radar.rl.rewards.risk_adjusted import RewardContext, compute_risk_adjusted_reward
from radar.rl.data_stream import split_rl_stream_chronological


def _sample_episode(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "symbol": ["AAPL"] * n,
        "close": 100 + rng.normal(0, 1, n).cumsum(),
        "next_return": rng.normal(0.001, 0.02, n),
        "p_up": rng.uniform(0.4, 0.7, n),
        "p_down": rng.uniform(0.3, 0.6, n),
        "y_vol_regime": rng.integers(0, 3, n),
        "regime_sim_mean": rng.uniform(0.5, 0.9, n),
        "regime_neighbor_win_rate": rng.uniform(0.4, 0.6, n),
        "atr_pct": [0.02] * n,
    })


def test_risk_adjusted_reward_penalizes_drawdown():
    config = RLRewardConfig(lambda_dd=2.0, dd_threshold=0.05)
    low_dd = compute_risk_adjusted_reward(
        RewardContext(0.01, 0.02, 0.01, 0.1), config
    )
    high_dd = compute_risk_adjusted_reward(
        RewardContext(0.01, 0.15, 0.01, 0.1), config
    )
    assert high_dd < low_dd


def test_radar_sizing_env_step():
    rl_config = RLConfig()
    env = RadarSizingEnv(_sample_episode(), rl_config)
    obs, _ = env.reset()
    assert obs.shape == (11,)

    action = np.array([2, 1])  # 50% position, 1.5x stop
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (11,)
    assert isinstance(reward, float)
    assert "daily_return" in info


def test_env_state_uses_predictions_not_raw_price():
    """State must include p_up/p_down from Layer 1, not raw OHLCV alone."""
    rl_config = RLConfig()
    df = _sample_episode(5)
    df.loc[0, "p_up"] = 0.99
    df.loc[0, "p_down"] = 0.01
    env = RadarSizingEnv(df, rl_config)
    obs, _ = env.reset()
    assert obs[0] == pytest.approx(0.99, abs=0.01)
    assert obs[1] == pytest.approx(0.01, abs=0.01)


def test_chronological_split_no_shuffle():
    stream = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="B").tolist() * 2,
        "symbol": ["AAPL"] * 10 + ["MSFT"] * 10,
        "next_return": [0.01] * 20,
    })
    train, test = split_rl_stream_chronological(stream, 0.7)
    train_dates = set(train["date"].unique())
    test_dates = set(test["date"].unique())
    assert train_dates.isdisjoint(test_dates)
    assert max(train_dates) <= min(test_dates)


def test_env_caps_position_on_event_day():
    rl_config = RLConfig(max_size_on_event_day=0.25)
    df = _sample_episode(5)
    df["is_event_day"] = 0
    df.loc[0, "is_event_day"] = 1
    env = RadarSizingEnv(df, rl_config)
    env.reset()
    env.step(np.array([4, 0]))  # 100% bucket on event day row
    assert env._position <= 0.25


def test_env_episode_terminates():
    rl_config = RLConfig()
    env = RadarSizingEnv(_sample_episode(5), rl_config)
    env.reset()
    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert steps == 4  # n-1 steps for n rows
