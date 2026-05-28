from __future__ import annotations

import numpy as np
import pandas as pd

from radar.forecast.intraday_forecast import _reanchor_price_path


def test_reanchor_starts_at_last_close():
    path = np.array([290.0, 285.0, 280.0])
    out = _reanchor_price_path(path, 310.0)
    assert abs(out[0] - 310.0) < 1e-6
    assert out[1] < out[0]


def test_reanchor_preserves_return_shape():
    path = np.array([100.0, 102.0, 101.0])
    out = _reanchor_price_path(path, 200.0)
    assert abs((out[1] / out[0]) - (path[1] / path[0])) < 1e-9
    assert abs((out[2] / out[1]) - (path[2] / path[1])) < 1e-9


def test_scale_path_preserves_volatility_shape():
    from radar.forecast.intraday_forecast import _scale_path_to_total_return

    # Alternating up/down should remain alternating after scaling.
    anchor = 100.0
    prices = np.array([101.0, 99.0, 101.0, 99.0], dtype=float)
    out = _scale_path_to_total_return(prices, anchor, target_total_return=0.02)
    diffs = np.sign(np.diff(np.r_[anchor, out]))
    assert list(diffs) == [1.0, -1.0, 1.0, -1.0]


def test_boost_path_volatility_increases_step_std():
    from radar.forecast.intraday_forecast import _boost_path_volatility

    anchor = 100.0
    prices = np.array([100.1, 100.15, 100.2, 100.18, 100.22], dtype=float)
    out = _boost_path_volatility(prices, anchor, target_step_vol=0.005, max_boost=10.0)
    def step_std(path):
        prev = anchor
        steps=[]
        for p in path:
            steps.append(p/prev-1.0)
            prev=p
        return float(np.std(steps))
    assert step_std(out) >= step_std(prices)
