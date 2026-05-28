from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.rl.data_stream import build_rl_stream, episodes_by_symbol, split_rl_stream_chronological
from radar.rl.envs.sizing_env import RadarSizingEnv

logger = structlog.get_logger(__name__)


def _make_env_factory(episode_df: pd.DataFrame, settings: Settings) -> Callable[[], RadarSizingEnv]:
    def _init() -> RadarSizingEnv:
        return RadarSizingEnv(
            episode_df=episode_df,
            rl_config=settings.rl,
            transaction_cost_bps=settings.model.transaction_cost_bps,
        )
    return _init


def train_sizing_policy(settings: Settings) -> dict[str, Any]:
    """
    Train PPO/A2C sizing policy on OOS prediction stream (chronological split).

    RL never receives raw OHLCV as sole input — only Layer 1 outputs + memory + state.
    """
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.env_util import make_vec_env

    settings.ensure_dirs()
    stream = build_rl_stream(settings)
    train_stream, _ = split_rl_stream_chronological(stream, settings.rl.train_fraction)

    episodes = episodes_by_symbol(train_stream)
    if not episodes:
        raise RuntimeError("No RL training episodes available.")

    primary_symbol = settings.universe.traded[0]
    episode_df = episodes.get(primary_symbol, next(iter(episodes.values())))

    env = make_vec_env(
        _make_env_factory(episode_df, settings),
        n_envs=1,
    )

    algo = settings.rl.algorithm.lower()
    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=settings.model.random_seed,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
        )
    elif algo == "a2c":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            seed=settings.model.random_seed,
            learning_rate=7e-4,
        )
    else:
        raise ValueError(f"Unsupported RL algorithm: {algo}")

    logger.info(
        "training_rl_policy",
        algorithm=algo,
        timesteps=settings.rl.total_timesteps,
        train_rows=len(train_stream),
    )
    model.learn(total_timesteps=settings.rl.total_timesteps)

    from pathlib import Path
    model_dir = Path(settings.rl.models_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{algo}_sizing"
    model.save(str(model_path))
    logger.info("saved_rl_model", path=str(model_path))

    return {
        "model_path": str(model_path),
        "algorithm": algo,
        "train_rows": len(train_stream),
        "timesteps": settings.rl.total_timesteps,
    }


def load_sizing_policy(settings: Settings):
    """Load trained sizing policy from disk."""
    from pathlib import Path

    from stable_baselines3 import A2C, PPO

    algo = settings.rl.algorithm.lower()
    model_path = Path(settings.rl.models_dir) / f"{algo}_sizing"
    if not model_path.with_suffix(".zip").exists() and not Path(str(model_path) + ".zip").exists():
        raise FileNotFoundError(f"RL model not found at {model_path}. Run train_rl first.")

    if algo == "ppo":
        return PPO.load(str(model_path))
    if algo == "a2c":
        return A2C.load(str(model_path))
    raise ValueError(f"Unsupported RL algorithm: {algo}")
