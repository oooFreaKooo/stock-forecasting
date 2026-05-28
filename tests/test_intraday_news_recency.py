from __future__ import annotations

import pandas as pd

from radar.intraday.news_recency import RECENCY_FEATURE_COLUMNS, attach_news_recency_features


def test_news_recency_no_lookahead():
  bar_times = pd.date_range("2025-05-28 14:00:00", periods=5, freq="5min", tz="UTC")
  frame = pd.DataFrame({"date": bar_times, "close": [100.0] * 5})

  headlines = pd.DataFrame({
      "symbol": ["AAPL", "AAPL", "MARKET"],
      "title": ["good", "bad", "macro"],
      "published": pd.to_datetime([
          "2025-05-28 13:50:00",
          "2025-05-28 14:02:00",
          "2025-05-28 14:04:00",
      ], utc=True).tz_convert("UTC").tz_localize(None),
      "sentiment": [0.8, -0.6, 0.1],
  })

  out = attach_news_recency_features(frame, "AAPL", settings=None, headlines=headlines)

  # 14:00 bar: only headline at 13:50 in 30m window
  assert out.loc[0, "news_count_30m"] == 1.0
  assert out.loc[0, "news_sent_mean_30m"] == 0.8

  # 14:10 bar (index 2): 13:50, 14:02, 14:04 all within 30m
  assert out.loc[2, "news_count_30m"] == 3.0

  # Last bar 14:20: all three headlines <= 14:20
  assert out.loc[4, "news_count_60m"] == 3.0

  for col in RECENCY_FEATURE_COLUMNS:
      assert col in out.columns


def test_news_recency_empty_headlines():
  frame = pd.DataFrame({
      "date": pd.date_range("2025-05-28 14:00:00", periods=3, freq="5min", tz="UTC"),
      "close": [1.0, 1.0, 1.0],
  })
  out = attach_news_recency_features(
      frame,
      "AAPL",
      settings=None,
      headlines=pd.DataFrame(columns=["symbol", "title", "published", "sentiment"]),
  )
  assert out["news_count_30m"].sum() == 0.0
