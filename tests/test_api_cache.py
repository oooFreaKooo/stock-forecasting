from __future__ import annotations

from datetime import datetime, timedelta, timezone

from radar.cache.artifacts import is_stale


def test_is_stale_old_timestamp():
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert is_stale(old, 60.0) is True


def test_is_stale_recent_timestamp():
    recent = datetime.now(timezone.utc).isoformat()
    assert is_stale(recent, 300.0) is False
