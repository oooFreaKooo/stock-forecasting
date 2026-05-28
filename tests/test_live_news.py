from __future__ import annotations

import pandas as pd

from radar.nlp.live_news import _headlines_to_records, refresh_live_news


def test_headlines_to_records_limits_per_symbol():
    headlines = pd.DataFrame([
        {"date": "2026-05-28", "symbol": "AAPL", "title": "Apple rises", "published": "2026-05-28T10:00:00Z"},
        {"date": "2026-05-28", "symbol": "AAPL", "title": "Apple falls", "published": "2026-05-28T09:00:00Z"},
        {"date": "2026-05-28", "symbol": "NVDA", "title": "Nvidia update", "published": "2026-05-28T08:00:00Z"},
    ])
    records = _headlines_to_records(headlines, limit_per_symbol=1)
    symbols = {item["symbol"] for item in records}
    assert symbols == {"AAPL", "NVDA"}
    assert len(records) == 2


def test_refresh_live_news_structure(tmp_path, monkeypatch):
    from radar.config.settings import get_settings

    settings = get_settings("config")
    settings.paths.processed_dir = str(tmp_path)

    headlines = pd.DataFrame([
        {
            "date": pd.Timestamp("2026-05-28"),
            "symbol": "AAPL",
            "title": "Apple stock jumps on strong guidance",
            "published": pd.Timestamp("2026-05-28T12:00:00Z"),
        },
        {
            "date": pd.Timestamp("2026-05-28"),
            "symbol": "NVDA",
            "title": "Nvidia faces supply concerns",
            "published": pd.Timestamp("2026-05-28T11:00:00Z"),
        },
    ])

    monkeypatch.setattr("radar.nlp.live_news.fetch_rss_headlines", lambda feeds: headlines)

    payload = refresh_live_news(settings, persist=True)

    assert payload["headline_count"] == 2
    assert "AAPL" in payload["symbols"]
    assert len(payload["headlines"]) >= 2
    assert (tmp_path / "live_news.json").exists()
