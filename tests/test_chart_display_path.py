from radar.forecast.chart_paths import build_forward_display_path, build_unified_model_path


def test_forward_display_path_anchors_on_last_actual():
    history = [
        {"date": "2026-05-28T18:00:00Z", "close": 100.0},
        {"date": "2026-05-28T18:05:00Z", "close": 101.0},
    ]
    forward = [
        {"date": "2026-05-28T18:10:00Z", "close": 101.5},
        {"date": "2026-05-28T18:15:00Z", "close": 102.0},
    ]
    out = build_forward_display_path(history, forward)
    assert len(out) == 3
    assert out[0]["close"] == 101.0
    assert out[-1]["close"] == 102.0


def test_unified_model_path_includes_backtest_and_forward():
    history = [
        {"date": "2026-05-28T17:00:00Z", "close": 100.0},
        {"date": "2026-05-28T18:00:00Z", "close": 101.0},
    ]
    validation = [
        {"date": "2026-05-28T17:30:00Z", "close": 100.5},
        {"date": "2026-05-28T18:10:00Z", "close": 101.2},
    ]
    forward = [
        {"date": "2026-05-28T18:15:00Z", "close": 101.5},
    ]
    out = build_unified_model_path(history, validation, forward)
    assert len(out) == 3
    assert out[0]["date"] == "2026-05-28T17:30:00Z"
    assert out[-1]["date"] == "2026-05-28T18:15:00Z"


def test_forward_display_path_excludes_backtest_before_anchor():
    history = [{"date": "2026-05-28T18:00:00Z", "close": 50.0}]
    backtest_style = [
        {"date": "2026-05-28T17:00:00Z", "close": 200.0},
        {"date": "2026-05-28T18:10:00Z", "close": 51.0},
    ]
    out = build_forward_display_path(history, backtest_style)
    assert out[0]["close"] == 50.0
    assert all(p["date"] >= "2026-05-28T18:00:00Z" for p in out)
