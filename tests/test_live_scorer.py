from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from radar.config.settings import get_settings
from radar.ensemble.live_scorer import score_live_symbol
from radar.validation.walk_forward import load_oos_predictions


def test_load_oos_predictions_prefers_ensemble(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()

    legacy = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"]),
        "symbol": ["AAPL"],
        "p_up": [0.4],
    })
    ensemble = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"]),
        "symbol": ["AAPL"],
        "p_up": [0.7],
        "p_ensemble": [0.7],
    })
    legacy.to_parquet(processed / "oos_predictions.parquet", index=False)
    ensemble.to_parquet(processed / "ensemble_oos.parquet", index=False)

    settings = get_settings("config")
    settings.paths.processed_dir = str(processed)

    loaded = load_oos_predictions(settings)
    assert float(loaded.iloc[0]["p_ensemble"]) == pytest.approx(0.7)


def test_score_live_symbol_uses_bundle(tmp_path, monkeypatch):
    settings = get_settings("config")
    settings.paths.processed_dir = str(tmp_path / "processed")
    settings.paths.models_dir = str(tmp_path / "models")
    Path(settings.paths.processed_dir).mkdir(parents=True)

    dates = pd.bdate_range("2024-01-01", periods=30)
    panel = pd.DataFrame({
        "date": np.tile(dates, 2),
        "symbol": ["AAPL"] * len(dates) + ["MSFT"] * len(dates),
        "close": np.linspace(100, 130, len(dates)).tolist() + np.linspace(200, 230, len(dates)).tolist(),
        "y_direction": [1] * (len(dates) * 2),
        "feat_a": np.random.default_rng(1).normal(size=len(dates) * 2),
        "feat_b": np.random.default_rng(2).normal(size=len(dates) * 2),
    })

    class DummyModel:
        def predict_proba(self, X):
            return np.column_stack([1 - np.full(len(X), 0.62), np.full(len(X), 0.62)])

    class DummyCal:
        def transform(self, raw):
            return raw

    bundle = {
        "base": {"lightgbm": DummyModel(), "xgboost": DummyModel(), "logistic": DummyModel()},
        "meta": DummyModel(),
        "calibrator": DummyCal(),
        "feature_cols": ["feat_a", "feat_b"],
        "fill_values": np.array([0.0, 0.0]),
        "base_models": ["lightgbm", "xgboost", "logistic"],
    }

    monkeypatch.setattr("radar.ensemble.live_scorer.load_ensemble_bundle", lambda s: bundle)
    monkeypatch.setattr(
        "radar.ensemble.live_scorer.load_feature_panel",
        lambda s: panel.copy(),
    )
    monkeypatch.setattr(
        "radar.ensemble.live_scorer.get_feature_columns",
        lambda s, df=None: ["feat_a", "feat_b"],
    )

    result = score_live_symbol(settings, "AAPL")
    assert result is not None
    assert result["source"] == "live"
    assert result["p_up"] == pytest.approx(0.62)
    assert "p_lightgbm" in result
