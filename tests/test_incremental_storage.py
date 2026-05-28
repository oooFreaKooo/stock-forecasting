from __future__ import annotations

import pandas as pd

from radar.data.intraday_store import IntradayBarStore
from radar.data.store import ParquetStore
from radar.nlp.ingest.news_fetcher import _entry_id, fetch_rss_headlines_incremental


def test_parquet_store_merge_appends_new_dates(tmp_path):
    store = ParquetStore(tmp_path)
    base = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "open": [1.0, 2.0],
        "high": [1.1, 2.1],
        "low": [0.9, 1.9],
        "close": [1.0, 2.0],
        "volume": [100, 200],
        "symbol": ["AAPL", "AAPL"],
    })
    store.write("AAPL", base)

    incoming = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [2.5, 3.0],
        "high": [2.6, 3.1],
        "low": [2.4, 2.9],
        "close": [2.5, 3.0],
        "volume": [250, 300],
        "symbol": ["AAPL", "AAPL"],
    })
    merged, added = store.merge("AAPL", incoming)

    assert len(merged) == 3
    assert added == 1
    assert store.last_date("AAPL") == pd.Timestamp("2024-01-03")


def test_intraday_store_upsert_dedupes_timestamps(tmp_path):
    store = IntradayBarStore(tmp_path)
    t0 = pd.Timestamp("2024-06-01 14:00:00")
    t1 = pd.Timestamp("2024-06-01 14:05:00")
    first = pd.DataFrame({
        "date": [t0, t1],
        "open": [100.0, 101.0],
        "high": [100.5, 101.5],
        "low": [99.5, 100.5],
        "close": [100.0, 101.0],
        "volume": [1000, 1100],
        "symbol": ["AAPL", "AAPL"],
    })
    assert store.upsert("AAPL", first) == 2

    updated = pd.DataFrame({
        "date": [t1, pd.Timestamp("2024-06-01 14:10:00")],
        "open": [101.0, 102.0],
        "high": [101.5, 102.5],
        "low": [100.5, 101.5],
        "close": [101.5, 102.0],
        "volume": [1200, 1300],
        "symbol": ["AAPL", "AAPL"],
    })
    added = store.upsert("AAPL", updated)
    assert added == 1
    assert len(store.read("AAPL")) == 3


def test_rss_incremental_skips_seen_ids(monkeypatch):
    class FakeEntry:
        def __init__(self, entry_id: str, title: str):
            self._data = {
                "id": entry_id,
                "title": title,
                "published_parsed": (2024, 6, 1, 12, 0, 0),
            }

        def get(self, key, default=None):
            return self._data.get(key, default)

    class FakeFeed:
        status = 200
        etag = "etag-1"
        modified = "Mon, 01 Jun 2024 12:00:00 GMT"
        entries = [
            FakeEntry("id-1", "First headline"),
            FakeEntry("id-2", "Second headline"),
        ]

    def fake_parse(url, **kwargs):
        return FakeFeed()

    monkeypatch.setitem(
        __import__("sys").modules,
        "feedparser",
        type("M", (), {"parse": staticmethod(fake_parse)}),
    )

    state = {"feeds": {"http://example.com/feed": {"seen_ids": ["id-1"]}}}
    df, new_state, n_new = fetch_rss_headlines_incremental(
        ["http://example.com/feed"],
        state,
    )
    assert n_new == 1
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Second headline"
    assert "id-2" in new_state["feeds"]["http://example.com/feed"]["seen_ids"]


def test_entry_id_fallback_is_stable():
    class Entry:
        def get(self, key, default=None):
            return {"title": "Hello", "published": "today"}.get(key, default)

    assert _entry_id(Entry()) == _entry_id(Entry())
