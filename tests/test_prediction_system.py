from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from radar.config.settings import get_settings
from radar.ensemble.live_scorer import score_live_symbol
from radar.forecast.intraday_timing import compute_intraday_timing
from radar.nlp.fusion.memory_enricher import SENTIMENT_FEATURE_COLUMNS, apply_live_sentiment_from_cache
from radar.portfolio.allocator import apply_portfolio_limits
from radar.portfolio.sizing import fractional_kelly_size


def test_score_live_symbol_imputes_nlp_features(tmp_path, monkeypatch):
    settings = get_settings("config")
    settings.paths.processed_dir = str(tmp_path / "processed")
    settings.paths.models_dir = str(tmp_path / "models")

    dates = pd.bdate_range("2024-01-01", periods=10)
    panel = pd.DataFrame({
        "date": dates,
        "symbol": ["AAPL"] * len(dates),
        "close": np.linspace(100, 110, len(dates)),
        "y_direction": [1] * len(dates),
        "feat_a": np.random.default_rng(1).normal(size=len(dates)),
        "feat_b": np.random.default_rng(2).normal(size=len(dates)),
    })

    class DummyModel:
        def predict_proba(self, X):
            return np.column_stack([1 - np.full(len(X), 0.61), np.full(len(X), 0.61)])

    class DummyCal:
        def transform(self, raw):
            return raw

    bundle = {
        "base": {"lightgbm": DummyModel()},
        "meta": DummyModel(),
        "calibrator": DummyCal(),
        "feature_cols": ["feat_a", "feat_b", "sentiment_mean"],
        "fill_values": np.array([0.0, 0.0, 0.0]),
        "base_models": ["lightgbm"],
        "horizon_models": {},
    }

    monkeypatch.setattr("radar.ensemble.live_scorer.load_ensemble_bundle", lambda s: bundle)
    monkeypatch.setattr("radar.ensemble.live_scorer.load_feature_panel", lambda s: panel.copy())
    monkeypatch.setattr(
        "radar.ensemble.live_scorer.sentiment_values_from_cache",
        lambda s, sym: {"sentiment_mean": 0.25},
    )

    result = score_live_symbol(settings, "AAPL")
    assert result is not None
    assert result["source"] == "live"


def test_apply_live_sentiment_from_cache(tmp_path):
    settings = get_settings("config")
    settings.paths.processed_dir = str(tmp_path / "processed")
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    (processed / "live_news.json").write_text(json.dumps({
        "fetched_at": "2026-05-28T10:00:00+00:00",
        "market_sentiment": 0.1,
        "market_sentiment_dispersion": 0.05,
        "symbols": {
            "AAPL": {
                "sentiment_mean": 0.42,
                "sentiment_ma": 0.35,
                "headline_count": 3,
            }
        },
    }))

    panel = pd.DataFrame({
        "date": pd.to_datetime(["2026-05-27", "2026-05-28"]),
        "symbol": ["AAPL", "AAPL"],
        "close": [100.0, 101.0],
    })
    out = apply_live_sentiment_from_cache(settings, panel)
    last = out.iloc[-1]
    assert last["sentiment_mean"] == pytest.approx(0.42)
    assert last["headline_count"] == pytest.approx(3)


def test_intraday_timing_scores_buy_context():
    bars = pd.DataFrame({
        "date": pd.date_range("2026-05-28 14:00", periods=20, freq="5min"),
        "close": np.linspace(100, 101.5, 20),
        "volume": np.full(20, 1000),
    })
    result = compute_intraday_timing(bars, daily_signal=1, daily_forecast_return=0.01)
    assert 0 <= result.entry_quality <= 1
    assert result.entry_quality > 0.5


def test_portfolio_limits_top_n():
    settings = get_settings("config")
    settings.ensemble.top_n_symbols = 1
    preds = [
        {"symbol": "AAPL", "signal": 1, "p_up": 0.6, "confluence_score": 0.8, "position_size": 0.2},
        {"symbol": "MSFT", "signal": 1, "p_up": 0.58, "confluence_score": 0.7, "position_size": 0.2},
    ]
    out = apply_portfolio_limits(preds, settings)
    signals = {p["symbol"]: p["signal"] for p in out}
    assert signals["AAPL"] == 1
    assert signals["MSFT"] == 0


def test_fractional_kelly_size():
    assert fractional_kelly_size(0.45, 0.8) == 0.0
    size = fractional_kelly_size(0.62, 0.75, max_weight=0.35)
    assert 0 < size <= 0.35
