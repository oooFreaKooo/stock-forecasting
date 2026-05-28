from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from radar.backtest.gated_signals import (
    apply_gated_signals,
    enrich_predictions_with_panel,
    evaluate_signal_tiers,
)
from radar.config.settings import Settings
from radar.data.store import ParquetStore
from radar.ensemble.live_scorer import score_live_symbol
from radar.forecast.baseline import forecast_baseline, forecast_return_1d
from radar.forecast.intraday_timing import compute_intraday_timing
from radar.monitoring.paper_tracker import log_paper_signal
from radar.nlp.live_news import load_live_news_cache
from radar.portfolio.sizing import fractional_kelly_size
from radar.validation.walk_forward import load_oos_predictions

logger = structlog.get_logger(__name__)

OPTIMIZED_CONFIG_PATH = "hybrid_optimized.json"


@dataclass
class SymbolPrediction:
    symbol: str
    date: pd.Timestamp
    last_close: float
    p_up: float
    forecast_return_1d: float
    signal: int
    confidence: str
    confluence_score: float
    probability_source: str = "oos"
    sentiment_mean: Optional[float] = None
    headline_count: Optional[int] = None
    market_sentiment: Optional[float] = None
    news_fetched_at: Optional[str] = None
    entry_quality: Optional[float] = None
    position_size: Optional[float] = None
    predicted_return_1d: Optional[float] = None


def _load_panel(settings: Settings) -> Optional[pd.DataFrame]:
    path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _get_effective_config(settings: Settings) -> tuple:
    """Return (config, threshold) using saved optimization if available."""
    config = settings.hybrid
    threshold = config.min_probability
    opt_path = Path(settings.paths.processed_dir) / OPTIMIZED_CONFIG_PATH
    if opt_path.exists():
        saved = json.loads(opt_path.read_text())
        threshold = saved.get("threshold", threshold)
        overrides = saved.get("params", {})
        config = config.model_copy(update=overrides)
    return config, threshold


def _load_latest_probability(
    settings: Settings,
    symbol: str,
) -> tuple[float, pd.Timestamp, Optional[pd.Series], str]:
    live = score_live_symbol(settings, symbol)
    if live is not None:
        return (
            float(live["p_up"]),
            pd.Timestamp(live["date"]),
            pd.Series(live),
            "live",
        )

    df = load_oos_predictions(settings)
    sym = df[df["symbol"] == symbol].sort_values("date")
    if sym.empty:
        return 0.5, pd.Timestamp.now().normalize(), None, "default"
    row = sym.iloc[-1]
    prob_col = "p_ensemble" if "p_ensemble" in row else "p_up"
    return float(row[prob_col]), row["date"], row, "oos"


def _load_latest_panel_row(settings: Settings, symbol: str) -> Optional[pd.Series]:
    panel = _load_panel(settings)
    if panel is None:
        return None
    sym = panel[panel["symbol"] == symbol].sort_values("date")
    if sym.empty:
        return None
    return sym.iloc[-1]


def _load_intraday_bars(symbol: str) -> pd.DataFrame:
    try:
        import yfinance as yf

        df = yf.Ticker(symbol).history(period="5d", interval="5m", auto_adjust=True, prepost=True)
        if df.empty:
            return pd.DataFrame(columns=["date", "close", "volume"])
        frame = df.reset_index()
        rename = {"Datetime": "date", "Date": "date", "Close": "close", "Volume": "volume"}
        frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        return frame[["date", "close", "volume"]].dropna()
    except Exception as exc:
        logger.warning("intraday_bars_failed", symbol=symbol, error=str(exc))
        return pd.DataFrame(columns=["date", "close", "volume"])


def predict_symbol(settings: Settings, symbol: str) -> SymbolPrediction:
    """Score one symbol: ensemble probability + gated signal + 1d return hint."""
    store = ParquetStore(settings.paths.raw_dir)
    if not store.exists(symbol):
        raise FileNotFoundError(f"No data for {symbol}. Run refresh or fetch_data first.")

    raw = store.read(symbol)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").sort_index()
    close = raw["close"]

    fc = forecast_baseline(
        close,
        horizon_days=settings.forecast.horizon_days,
        context_days=settings.forecast.context_days,
    )

    p_up, pred_date, score_row, probability_source = _load_latest_probability(settings, symbol)
    panel_row = _load_latest_panel_row(settings, symbol)
    last_close = float(close.iloc[-1])
    forecast_ret = forecast_return_1d(fc, last_close)
    predicted_return = None
    if score_row is not None and "predicted_return_1d" in score_row.index:
        predicted_return = float(score_row["predicted_return_1d"])

    row_data: dict = {
        "date": pred_date,
        "symbol": symbol,
        "p_up": p_up,
        "p_ensemble": p_up,
        "forecast_return_1d": forecast_ret,
    }
    if score_row is not None:
        for col in ("p_lightgbm", "p_xgboost", "p_logistic", "trade_allowed", "p_up_5d", "p_up_20d"):
            if col in score_row.index and pd.notna(score_row[col]):
                row_data[col] = score_row[col]
    if panel_row is not None:
        for col in (
            "regime_neighbor_win_rate", "is_event_day", "momentum_rank",
            "y_vol_regime", "vix_level", "regime_sim_mean",
            "sentiment_mean", "sentiment_ma", "headline_count", "market_sentiment",
            "sentiment_delta_1d", "headline_surprise", "negative_headline_ratio",
        ):
            if col in panel_row.index:
                row_data[col] = panel_row[col]

    if all(k in row_data for k in ("p_lightgbm", "p_xgboost", "p_logistic")):
        probs = [row_data["p_lightgbm"], row_data["p_xgboost"], row_data["p_logistic"]]
        row_data["model_disagreement"] = float(np.std(probs))

    config, threshold = _get_effective_config(settings)
    gated = apply_gated_signals(pd.DataFrame([row_data]), config, threshold=threshold)
    row = gated.iloc[0]
    confluence_score = float(row.get("confluence_score", p_up))

    sentiment_mean = row_data.get("sentiment_mean")
    headline_count = row_data.get("headline_count")
    market_sentiment = row_data.get("market_sentiment")
    news_fetched_at = None

    news_cache = load_live_news_cache(settings)
    if news_cache:
        news_fetched_at = news_cache.get("fetched_at")
        if market_sentiment is None:
            market_sentiment = news_cache.get("market_sentiment")
        sym_news = news_cache.get("symbols", {}).get(symbol)
        if sym_news:
            if sentiment_mean is None:
                sentiment_mean = sym_news.get("sentiment_mean")
            if headline_count is None:
                headline_count = sym_news.get("headline_count")

    signal = int(row["signal"])
    intraday = compute_intraday_timing(
        _load_intraday_bars(symbol),
        daily_signal=signal,
        daily_forecast_return=forecast_ret,
    )

    position_size = fractional_kelly_size(
        p_up,
        confluence_score,
        max_weight=settings.ensemble.max_single_name_weight,
    )
    if signal != 1:
        position_size = 0.0

    return SymbolPrediction(
        symbol=symbol,
        date=pred_date,
        last_close=last_close,
        p_up=p_up,
        forecast_return_1d=forecast_ret,
        signal=signal,
        confidence=str(row["confidence"]),
        confluence_score=confluence_score,
        probability_source=probability_source,
        sentiment_mean=float(sentiment_mean) if sentiment_mean is not None else None,
        headline_count=int(headline_count) if headline_count is not None else None,
        market_sentiment=float(market_sentiment) if market_sentiment is not None else None,
        news_fetched_at=news_fetched_at,
        entry_quality=intraday.entry_quality,
        position_size=position_size,
        predicted_return_1d=predicted_return,
    )


def evaluate_gated_performance(settings: Settings, *, save_optimized: bool = False) -> dict:
    """Measure hit rate across signal tiers on OOS backtest data."""
    from radar.backtest.expectancy import run_expectancy_backtest
    from radar.backtest.gated_signals import apply_gated_signals as apply_gated
    from radar.monitoring.paper_tracker import evaluate_paper_trades

    preds = load_oos_predictions(settings)
    prob_col = "p_ensemble" if "p_ensemble" in preds.columns else "p_up"
    preds["p_up"] = preds[prob_col]
    preds["p_ensemble"] = preds[prob_col]

    panel = _load_panel(settings)
    if panel is not None:
        preds = enrich_predictions_with_panel(preds, panel)

    tiers = evaluate_signal_tiers(preds, settings.hybrid)

    if save_optimized and settings.hybrid.auto_optimize:
        opt = tiers["gated_v2_optimized"]
        path = Path(settings.paths.processed_dir) / OPTIMIZED_CONFIG_PATH
        payload = {
            "threshold": opt.get("threshold", settings.hybrid.min_probability),
            "params": opt.get("params", {}),
            "hit_rate": opt.get("hit_rate", 0),
            "n_trades": opt.get("n_trades", 0),
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.info("saved_optimized_config", path=str(path), **payload)

    opt = tiers["gated_v2_optimized"]
    gated_hit_rate = opt["hit_rate"] if opt["n_trades"] > 0 else tiers["simple"]["hit_rate"]
    gated_trades = opt["n_trades"] if opt["n_trades"] > 0 else tiers["simple"]["n_trades"]

    config, threshold = _get_effective_config(settings)
    gated_preds = apply_gated(preds, config, threshold=threshold)
    expectancy = run_expectancy_backtest(gated_preds, settings)
    paper = evaluate_paper_trades(settings.paths.processed_dir, settings.paths.raw_dir)

    return {
        "threshold_used": opt.get("threshold", settings.hybrid.min_probability),
        "optimized_params": opt.get("params", {}),
        "simple_hit_rate": tiers["simple"]["hit_rate"],
        "simple_trades": tiers["simple"]["n_trades"],
        "gated_v1_hit_rate": tiers["gated_v1"]["hit_rate"],
        "gated_v1_trades": tiers["gated_v1"]["n_trades"],
        "gated_hit_rate": gated_hit_rate,
        "gated_trades": gated_trades,
        "coverage_pct": 100.0 * gated_trades / max(len(preds), 1),
        "expectancy": expectancy.get("pooled", {}).get("expectancy", 0.0),
        "profit_factor": expectancy.get("pooled", {}).get("profit_factor", 0.0),
        "max_drawdown": expectancy.get("pooled", {}).get("max_drawdown", 0.0),
        "paper_trading": paper,
        "tiers": tiers,
    }
