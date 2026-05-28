import pandas as pd

from radar.api.chart_series import _forward_points_from_ai_return


def test_forward_points_from_ai_return_day_one_only():
    last_ts = pd.Timestamp("2026-05-27")
    pts = _forward_points_from_ai_return(100.0, last_ts, 3, 0.05)
    assert len(pts) == 3
    assert pts[0]["close"] == 105.0
    assert pts[1]["close"] == 105.0
    assert pts[2]["close"] == 105.0
