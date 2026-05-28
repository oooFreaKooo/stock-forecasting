from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.rl.data_stream import build_rl_stream, episodes_by_symbol, split_rl_stream_chronological
from radar.rl.envs.sizing_env import RadarSizingEnv
from radar.rl.train_sizing import load_sizing_policy
from radar.validation.metrics import compute_expectancy, max_drawdown

logger = structlog.get_logger(__name__)


def evaluate_policy_on_episode(
    model,
    episode_df: pd.DataFrame,
    settings: Settings,
) -> dict[str, float]:
    """Run trained policy on one symbol episode and collect metrics."""
    env = RadarSizingEnv(
        episode_df=episode_df,
        rl_config=settings.rl,
        transaction_cost_bps=settings.model.transaction_cost_bps,
    )

    obs, _ = env.reset()
    done = False
    daily_returns: list[float] = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        daily_returns.append(info["daily_return"])
        done = terminated or truncated

    returns = np.array(daily_returns, dtype=float)
    returns = np.nan_to_num(returns, nan=0.0)
    equity = np.cumprod(np.clip(1 + returns, 0.01, 10.0))
    exp = compute_expectancy(returns)

    return {
        **exp,
        "total_return": float(equity[-1] - 1) if len(equity) else 0.0,
        "max_drawdown": max_drawdown(equity),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0,
        "final_equity": float(equity[-1]) if len(equity) else 1.0,
    }


def evaluate_sizing_policy(settings: Settings) -> dict[str, Any]:
    """Evaluate RL sizing policy on held-out OOS chronological split."""
    stream = build_rl_stream(settings)
    _, test_stream = split_rl_stream_chronological(stream, settings.rl.train_fraction)

    if test_stream.empty:
        raise RuntimeError("No test data for RL evaluation.")

    model = load_sizing_policy(settings)
    episodes = episodes_by_symbol(test_stream)

    by_symbol: dict[str, dict] = {}

    for symbol, episode_df in episodes.items():
        metrics = evaluate_policy_on_episode(model, episode_df, settings)
        by_symbol[symbol] = metrics
        logger.info("rl_eval_symbol", symbol=symbol, **metrics)

    pooled = {
        "total_return": float(np.mean([m["total_return"] for m in by_symbol.values()])),
        "expectancy": float(np.mean([m["expectancy"] for m in by_symbol.values()])),
        "sharpe": float(np.mean([m["sharpe"] for m in by_symbol.values()])),
        "max_drawdown": float(np.mean([m["max_drawdown"] for m in by_symbol.values()])),
        "win_rate": float(np.mean([m["win_rate"] for m in by_symbol.values()])),
        "n_trades": int(sum(m["n_trades"] for m in by_symbol.values())),
    }

    results = {
        "by_symbol": by_symbol,
        "pooled": pooled,
        "test_rows": len(test_stream),
    }

    reports_dir = Path(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "rl_evaluation.json"
    import json
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("saved_rl_evaluation", path=str(out_path))

    return results
