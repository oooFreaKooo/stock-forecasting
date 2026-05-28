from __future__ import annotations

import numpy as np
import pandas as pd

from radar.forecast.intraday_forecast import _forecast_baseline_bars, forecast_intraday_series


def test_baseline_intraday_not_flat():
    rng = np.random.default_rng(3)
    close = 300 + np.cumsum(rng.normal(0, 0.4, 200))
    forecast = _forecast_baseline_bars(close, horizon=24)
    assert len(forecast) == 24
    step_returns = np.diff(forecast) / forecast[:-1]
    assert len({round(r, 6) for r in step_returns}) > 1


def test_forecast_intraday_series_shape():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2026-05-20 13:30", periods=200, freq="5min")
    close = 300 + np.cumsum(rng.normal(0, 0.2, len(dates)))
    frame = pd.DataFrame({"date": dates, "close": close})
    result = forecast_intraday_series(frame, "5m")
    assert result.horizon_bars == 64
    assert len(result.points) == 64
    assert result.engine in ("baseline_bars", "intraday_lgbm")
