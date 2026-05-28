from __future__ import annotations

import pandas as pd

from radar.forecast.bar_alignment import align_forecast_points


def test_align_forecast_uses_forecast_timestamp_not_misindexed_actual():
    """yfinance 1h bars can land on :30; strict +1h projection lands on :00."""
    actual = pd.DataFrame({
        "date": pd.to_datetime([
            "2026-05-26 12:00:00",
            "2026-05-26 13:00:00",
            "2026-05-26 13:30:00",
            "2026-05-26 14:30:00",
            "2026-05-26 15:30:00",
        ]),
        "close": [270.0, 269.0, 268.0, 265.0, 264.0],
    })
    forecast_points = [
        {"date": "2026-05-26T13:00:00Z", "close": 268.5},
        {"date": "2026-05-26T14:00:00Z", "close": 266.0},
        {"date": "2026-05-26T15:00:00Z", "close": 264.5},
    ]

    val_points, predicted, actual_vals, _ = align_forecast_points(actual, forecast_points, "1h")

    assert [p["date"] for p in val_points] == [
        "2026-05-26T13:00:00Z",
        "2026-05-26T14:00:00Z",
        "2026-05-26T15:00:00Z",
    ]
    assert predicted == [268.5, 266.0, 264.5]
    assert actual_vals == [269.0, 268.0, 265.0]
