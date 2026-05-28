from __future__ import annotations

import pandas as pd

from radar.intraday.event_features import INTRADAY_EVENT_COLUMNS, attach_event_features


def test_attach_event_features_by_calendar_date():
  bars = pd.DataFrame({
      "date": pd.to_datetime([
          "2025-05-28 14:35:00",
          "2025-05-28 15:10:00",
          "2025-05-29 09:30:00",
      ]),
      "close": [100.0, 101.0, 102.0],
  })
  events = pd.DataFrame({
      "date": pd.to_datetime(["2025-05-28", "2025-05-29"]),
      "is_event_day": [1.0, 0.0],
      "is_fomc_day": [1.0, 0.0],
      "is_cpi_day": [0.0, 1.0],
      "is_nfp_day": [0.0, 0.0],
      "is_post_event_day": [0.0, 1.0],
      "days_to_next_event": [2.0, 5.0],
      "days_since_last_event": [1.0, 0.0],
      "geo_risk_flag": [0.0, 1.0],
      "conflict_intensity": [0.1, 0.4],
  })

  class _Settings:
      class paths:
          processed_dir = "/tmp/unused"

  out = attach_event_features(bars, _Settings(), events=events)

  assert out.loc[0, "is_fomc_day"] == 1.0
  assert out.loc[1, "is_fomc_day"] == 1.0
  assert out.loc[2, "is_cpi_day"] == 1.0
  assert out.loc[2, "geo_risk_flag"] == 1.0

  for col in INTRADAY_EVENT_COLUMNS:
      assert col in out.columns
