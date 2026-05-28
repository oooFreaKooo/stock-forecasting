from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.backtest.gated_signals import apply_gated_signals, optimize_threshold
from radar.config.schemas import HybridConfig
from radar.forecast.baseline import forecast_baseline, forecast_return_1d


def test_baseline_forecast_shape():
    dates = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(100 + np.arange(60) * 0.5, index=dates)
    result = forecast_baseline(close, horizon_days=5)
    assert len(result.prices) == 5
    assert result.engine == "baseline"
    assert result.prices[-1] > 0


def test_baseline_forecast_not_constant_growth():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=120)
    noise = rng.normal(0, 0.015, len(dates))
    close = pd.Series(100 * np.cumprod(1 + noise), index=dates)
    result = forecast_baseline(close, horizon_days=5)
    step_returns = np.diff(result.prices) / result.prices[:-1]
    assert len({round(r, 6) for r in step_returns}) > 1


def test_forecast_return_1d():
    dates = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(100 + np.arange(60) * 0.5, index=dates)
    result = forecast_baseline(close, horizon_days=3)
    ret = forecast_return_1d(result, float(close.iloc[-1]))
    assert isinstance(ret, float)


def test_gated_signals_filters_low_confidence():
    config = HybridConfig(
        min_probability=0.60, min_memory_win_rate=0.55, skip_event_days=True,
        top_n_per_day=0, require_multi_horizon=False, require_model_agreement=False,
    )
    df = pd.DataFrame({
        "p_up": [0.65, 0.55, 0.70],
        "p_ensemble": [0.65, 0.55, 0.70],
        "forecast_return_1d": [0.01, 0.01, -0.02],
        "regime_neighbor_win_rate": [0.60, 0.60, 0.60],
        "is_event_day": [0, 0, 0],
    })
    out = apply_gated_signals(df, config, threshold=0.58)
    assert out.iloc[0]["signal"] == 1
    assert out.iloc[1]["signal"] == 0
    assert out.iloc[2]["signal"] == 0


def test_gated_signals_multi_horizon():
    config = HybridConfig(
        min_probability=0.55,
        require_multi_horizon=True,
        require_model_agreement=True,
        max_model_disagreement=0.06,
        top_n_per_day=0,
    )
    df = pd.DataFrame({
        "p_up": [0.65, 0.65, 0.65],
        "p_ensemble": [0.65, 0.65, 0.65],
        "p_lightgbm": [0.60, 0.60, 0.60],
        "p_xgboost": [0.62, 0.62, 0.62],
        "p_logistic": [0.58, 0.58, 0.58],
        "trade_allowed": [True, False, True],
        "forecast_return_1d": [0.01, 0.01, 0.01],
        "regime_neighbor_win_rate": [0.55, 0.55, 0.55],
        "is_event_day": [0, 0, 0],
        "momentum_rank": [0.6, 0.6, 0.6],
        "y_vol_regime": [1, 1, 1],
    })
    df["model_disagreement"] = df[["p_lightgbm", "p_xgboost", "p_logistic"]].std(axis=1)
    out = apply_gated_signals(df, config, threshold=0.55)
    assert out.iloc[0]["signal"] == 1
    assert out.iloc[1]["signal"] == 0


def test_confluence_score():
    config = HybridConfig()
    df = pd.DataFrame({
        "p_ensemble": [0.7],
        "regime_neighbor_win_rate": [0.6],
        "momentum_rank": [0.8],
        "forecast_return_1d": [0.01],
        "model_disagreement": [0.02],
        "regime_sim_mean": [0.7],
        "trade_allowed": [True],
    })
    from radar.backtest.gated_signals import compute_confluence_score
    score = compute_confluence_score(df, config)
    assert score.iloc[0] > 0.6


def test_optimize_threshold():
    rng = np.random.default_rng(42)
    n = 200
    probs = rng.uniform(0.45, 0.75, n)
    returns = rng.normal(0.001, 0.02, n)
    returns[probs > 0.62] += 0.005
    df = pd.DataFrame({
        "p_up": probs,
        "p_ensemble": probs,
        "next_return": returns,
        "y_direction": (returns > 0).astype(int),
        "forecast_return_1d": returns,
        "regime_neighbor_win_rate": [0.55] * n,
        "is_event_day": [0] * n,
    })
    config = HybridConfig(
        require_forecast_agreement=False, skip_event_days=False,
        top_n_per_day=0, require_multi_horizon=False, require_model_agreement=False,
    )
    result = optimize_threshold(df, min_trades=20, hybrid_config=config)
    assert result["best_threshold"] >= 0.52
    assert result["n_trades"] >= 20
