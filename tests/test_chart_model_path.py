from __future__ import annotations

from radar.forecast.chart_paths import build_unified_model_path


def test_unified_model_path_connects_at_last_actual():
    history = [
        {"date": "2026-05-28T14:00:00Z", "close": 100.0},
        {"date": "2026-05-28T14:05:00Z", "close": 101.0},
    ]
    validation = [
        {"date": "2026-05-28T13:55:00Z", "close": 99.5},
        {"date": "2026-05-28T14:05:00Z", "close": 100.2},
    ]
    forward = [
        {"date": "2026-05-28T14:10:00Z", "close": 101.5},
        {"date": "2026-05-28T14:15:00Z", "close": 102.0},
    ]

    path = build_unified_model_path(history, validation, forward)

    assert path[0]["date"] == "2026-05-28T13:55:00Z"
    assert path[1] == history[-1]
    assert path[2]["date"] == "2026-05-28T14:10:00Z"
    assert path[-1]["close"] == 102.0
    assert len(path) == 4


def test_unified_model_path_drops_validation_at_now():
    history = [{"date": "2026-05-28T14:00:00Z", "close": 50.0}]
    validation = [
        {"date": "2026-05-28T14:00:00Z", "close": 49.0},
    ]
    forward = [{"date": "2026-05-28T14:05:00Z", "close": 51.0}]

    path = build_unified_model_path(history, validation, forward)

    assert path == [
        {"date": "2026-05-28T14:00:00Z", "close": 50.0},
        {"date": "2026-05-28T14:05:00Z", "close": 51.0},
    ]
