from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from radar.config.schemas import HybridConfig


def enrich_predictions_with_panel(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Merge feature panel columns needed for advanced gating."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"])

    extra_cols = [
        "close", "momentum_rank", "y_vol_regime", "vix_level", "regime_sim_mean",
        "regime_neighbor_win_rate", "regime_neighbor_avg_return", "is_event_day",
        "rsi", "SPY_trend_20d", "realized_vol", "regime_sim_top1",
        "sentiment_mean", "sentiment_ma", "headline_count", "market_sentiment",
        "sentiment_delta_1d", "headline_surprise", "negative_headline_ratio",
    ]
    merge_cols = ["date", "symbol"] + [c for c in extra_cols if c in panel.columns]
    merged = out.merge(panel[merge_cols], on=["date", "symbol"], how="left", suffixes=("", "_panel"))

    for col in extra_cols:
        panel_col = f"{col}_panel"
        if col not in merged.columns and panel_col in merged.columns:
            merged[col] = merged[panel_col]
        elif col in merged.columns and panel_col in merged.columns:
            merged[col] = merged[col].fillna(merged[panel_col])

    if "close" in merged.columns and "forecast_return_1d" not in merged.columns:
        merged["forecast_return_1d"] = (
            merged.groupby("symbol")["close"].pct_change(1).shift(1).fillna(0)
        )

    if all(c in merged.columns for c in ("p_lightgbm", "p_xgboost", "p_logistic")):
        merged["model_disagreement"] = merged[["p_lightgbm", "p_xgboost", "p_logistic"]].std(axis=1)
        merged["model_unanimous"] = (
            (merged["p_lightgbm"] > 0.5)
            & (merged["p_xgboost"] > 0.5)
            & (merged["p_logistic"] > 0.5)
        ).astype(int)

    return merged


def compute_confluence_score(df: pd.DataFrame, config: HybridConfig) -> pd.Series:
    """
    Weighted confluence score [0, 1] combining multiple independent signals.

    Higher = stronger agreement across models, memory, momentum, and forecast.
    """
    prob_col = "p_ensemble" if "p_ensemble" in df.columns else "p_up"
    score = pd.Series(0.0, index=df.index)

    score += 0.30 * df[prob_col].clip(0, 1)

    if "regime_neighbor_win_rate" in df.columns:
        score += 0.20 * df["regime_neighbor_win_rate"].fillna(0.5).clip(0, 1)

    if "momentum_rank" in df.columns:
        score += 0.15 * df["momentum_rank"].fillna(0.5).clip(0, 1)

    if "forecast_return_1d" in df.columns:
        score += 0.10 * (df["forecast_return_1d"] > 0).astype(float)

    if "model_disagreement" in df.columns:
        agreement = (1.0 - df["model_disagreement"].fillna(0.5) * 4).clip(0, 1)
        score += 0.10 * agreement

    if "regime_sim_mean" in df.columns:
        score += 0.05 * df["regime_sim_mean"].fillna(0.5).clip(0, 1)

    if "trade_allowed" in df.columns:
        score += 0.05 * df["trade_allowed"].astype(float)

    if "sentiment_mean" in df.columns:
        normalized = (df["sentiment_mean"].fillna(0) + 1) / 2
        score += 0.025 * normalized.clip(0, 1)

    if "market_sentiment" in df.columns:
        normalized = (df["market_sentiment"].fillna(0) + 1) / 2
        score += 0.025 * normalized.clip(0, 1)

    return score.clip(0, 1)


def apply_gated_signals(
    predictions: pd.DataFrame,
    config: HybridConfig,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    High-precision signal filter — trades less often, higher win rate.

    Gates (all must pass):
    1. Model probability above threshold
    2. Forecast direction agrees (optional)
    3. Memory neighbor win rate above minimum
    4. Skip macro event days (optional)
    5. Model agreement — base models not conflicting (optional)
    6. Multi-horizon agreement via trade_allowed (optional)
    7. Momentum rank above minimum (optional)
    8. Volatility regime below maximum (optional)
    9. Cross-sectional top-N per day (optional)
    10. Confluence score above minimum (optional)
    """
    out = predictions.copy()
    prob_col = "p_ensemble" if "p_ensemble" in out.columns else "p_up"
    thresh = threshold if threshold is not None else config.min_probability

    prob_ok = out[prob_col] >= thresh

    if config.require_forecast_agreement and "forecast_return_1d" in out.columns:
        forecast_ok = out["forecast_return_1d"] > 0
    else:
        forecast_ok = pd.Series(True, index=out.index)

    if "regime_neighbor_win_rate" in out.columns:
        memory_ok = out["regime_neighbor_win_rate"].fillna(0.5) >= config.min_memory_win_rate
    else:
        memory_ok = pd.Series(True, index=out.index)

    if config.skip_event_days and "is_event_day" in out.columns:
        event_ok = out["is_event_day"].fillna(0).astype(int) == 0
    else:
        event_ok = pd.Series(True, index=out.index)

    if config.require_model_agreement and "model_disagreement" in out.columns:
        agreement_ok = out["model_disagreement"] <= config.max_model_disagreement
    elif config.require_model_agreement and "model_unanimous" in out.columns:
        agreement_ok = out["model_unanimous"] == 1
    else:
        agreement_ok = pd.Series(True, index=out.index)

    if config.require_multi_horizon and "trade_allowed" in out.columns:
        horizon_ok = out["trade_allowed"].astype(bool)
    else:
        horizon_ok = pd.Series(True, index=out.index)

    if config.min_momentum_rank > 0 and "momentum_rank" in out.columns:
        momentum_ok = out["momentum_rank"].fillna(0) >= config.min_momentum_rank
    else:
        momentum_ok = pd.Series(True, index=out.index)

    if config.max_vol_regime < 2 and "y_vol_regime" in out.columns:
        vol_ok = out["y_vol_regime"].fillna(1) < config.max_vol_regime + 1
    else:
        vol_ok = pd.Series(True, index=out.index)

    if config.max_vix_level is not None and "vix_level" in out.columns:
        vix_ok = out["vix_level"].fillna(20) <= config.max_vix_level
    else:
        vix_ok = pd.Series(True, index=out.index)

    out["confluence_score"] = compute_confluence_score(out, config)
    if config.min_confluence_score > 0:
        confluence_ok = out["confluence_score"] >= config.min_confluence_score
    else:
        confluence_ok = pd.Series(True, index=out.index)

    base_signal = prob_ok & forecast_ok & memory_ok & event_ok & agreement_ok
    base_signal = base_signal & horizon_ok & momentum_ok & vol_ok & vix_ok & confluence_ok

    out["signal"] = base_signal.astype(int)

    if config.top_n_per_day > 0 and "date" in out.columns:
        out["_day_rank"] = out.groupby("date")[prob_col].rank(ascending=False, method="first")
        out.loc[out["_day_rank"] > config.top_n_per_day, "signal"] = 0
        out.drop(columns=["_day_rank"], inplace=True)

    def _confidence(row: pd.Series) -> str:
        if row["signal"] != 1:
            return "none"
        conf = float(row.get("confluence_score", row[prob_col]))
        if conf >= 0.75:
            return "high"
        if conf >= 0.65:
            return "medium"
        return "low"

    out["confidence"] = out.apply(_confidence, axis=1)
    out["gate_probability"] = prob_ok.astype(int)
    out["gate_forecast"] = forecast_ok.astype(int)
    out["gate_memory"] = memory_ok.astype(int)
    out["gate_event"] = event_ok.astype(int)
    out["gate_agreement"] = agreement_ok.astype(int)
    out["gate_horizon"] = horizon_ok.astype(int)
    out["gate_momentum"] = momentum_ok.astype(int)
    out["gate_vol"] = vol_ok.astype(int)
    out["gate_confluence"] = confluence_ok.astype(int)
    return out


def _hit_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float(((trades["next_return"] > 0) & (trades["y_direction"] == 1)).mean())


def optimize_threshold(
    predictions: pd.DataFrame,
    min_trades: int = 30,
    threshold_range: Optional[np.ndarray] = None,
    use_gates: bool = True,
    hybrid_config: Optional[HybridConfig] = None,
) -> dict:
    """Find threshold maximizing hit rate with minimum trade count."""
    if threshold_range is None:
        threshold_range = np.arange(0.52, 0.72, 0.02)

    best = {"best_threshold": 0.55, "hit_rate": 0.0, "n_trades": 0}

    for thresh in threshold_range:
        if use_gates and hybrid_config is not None:
            gated = apply_gated_signals(predictions, hybrid_config, threshold=float(thresh))
            trades = gated[gated["signal"] == 1]
        else:
            prob_col = "p_ensemble" if "p_ensemble" in predictions.columns else "p_up"
            trades = predictions[predictions[prob_col] > thresh]

        if len(trades) < min_trades:
            continue

        hr = _hit_rate(trades)
        if hr > best["hit_rate"]:
            best = {"best_threshold": float(thresh), "hit_rate": hr, "n_trades": len(trades)}

    return best


def optimize_gating_params(
    predictions: pd.DataFrame,
    base_config: HybridConfig,
    min_trades: int = 25,
) -> dict:
    """
    Grid-search gating parameters to maximize OOS hit rate.

    Returns best config overrides and performance metrics.
    """
    best: dict = {
        "hit_rate": 0.0,
        "n_trades": 0,
        "params": {},
        "threshold": base_config.min_probability,
    }

    thresholds = np.arange(0.54, 0.68, 0.02)
    memory_rates = [0.50, 0.52, 0.54]
    momentum_ranks = [0.0, 0.50, 0.55, 0.60]
    confluence_scores = [0.0, 0.60, 0.65, 0.70]
    top_n_options = [0, 2, 3]

    for thresh in thresholds:
        for mem in memory_rates:
            for mom in momentum_ranks:
                for conf in confluence_scores:
                    for top_n in top_n_options:
                        cfg = base_config.model_copy(update={
                            "min_probability": float(thresh),
                            "min_memory_win_rate": mem,
                            "min_momentum_rank": mom,
                            "min_confluence_score": conf,
                            "top_n_per_day": top_n,
                        })
                        gated = apply_gated_signals(predictions, cfg, threshold=float(thresh))
                        trades = gated[gated["signal"] == 1]
                        if len(trades) < min_trades:
                            continue
                        hr = _hit_rate(trades)
                        if hr > best["hit_rate"]:
                            best = {
                                "hit_rate": hr,
                                "n_trades": len(trades),
                                "threshold": float(thresh),
                                "params": {
                                    "min_memory_win_rate": mem,
                                    "min_momentum_rank": mom,
                                    "min_confluence_score": conf,
                                    "top_n_per_day": top_n,
                                },
                            }

    return best


def evaluate_signal_tiers(predictions: pd.DataFrame, config: HybridConfig) -> dict:
    """Compare hit rates across signal tiers: simple, gated, optimized."""
    prob_col = "p_ensemble" if "p_ensemble" in predictions.columns else "p_up"
    preds = predictions.copy()
    preds["p_up"] = preds[prob_col]
    preds["p_ensemble"] = preds[prob_col]

    simple = preds[preds[prob_col] > 0.55]

    opt = optimize_gating_params(preds, config, min_trades=config.min_trades_for_threshold)
    optimized_cfg = config.model_copy(update={
        "min_probability": opt["threshold"],
        **opt.get("params", {}),
    })
    gated_v1 = apply_gated_signals(preds, config, threshold=config.min_probability)
    gated_v1_trades = gated_v1[gated_v1["signal"] == 1]

    gated_v2 = apply_gated_signals(preds, optimized_cfg, threshold=opt["threshold"])
    gated_v2_trades = gated_v2[gated_v2["signal"] == 1]

    return {
        "simple": {"hit_rate": _hit_rate(simple), "n_trades": len(simple)},
        "gated_v1": {"hit_rate": _hit_rate(gated_v1_trades), "n_trades": len(gated_v1_trades)},
        "gated_v2_optimized": {
            "hit_rate": opt["hit_rate"],
            "n_trades": opt["n_trades"],
            "threshold": opt["threshold"],
            "params": opt.get("params", {}),
        },
        "best_config": optimized_cfg,
    }
