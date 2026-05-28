from __future__ import annotations

import numpy as np
import pandas as pd

from radar.api.chart_validation import build_intraday_validation
from radar.forecast.intraday_forecast import forecast_intraday_series


def _frame(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2026-05-20 13:30", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.05, n))
    return pd.DataFrame({"date": dates, "close": close, "symbol": "TEST"})


def test_intraday_validation_uses_lgbm_walk_forward():
    frame = _frame()
    val_points, metrics = build_intraday_validation(frame, "5m", symbol="TEST")
    assert metrics["n_points"] > 0
    assert len(val_points) == metrics["n_points"]


def test_forecast_without_model_uses_baseline():
    frame = _frame()
    from unittest.mock import patch

    with patch("radar.forecast.intraday_forecast.load_bundle", return_value=None):
        result = forecast_intraday_series(frame, "5m")
    assert result.engine == "baseline_bars"
