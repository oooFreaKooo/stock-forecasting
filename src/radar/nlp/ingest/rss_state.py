from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATE_FILENAME = "rss_feed_state.json"
MAX_SEEN_IDS_PER_FEED = 500


def state_path(processed_dir: Path) -> Path:
    return Path(processed_dir) / STATE_FILENAME


def load_feed_state(processed_dir: Path) -> dict[str, Any]:
    path = state_path(processed_dir)
    if not path.exists():
        return {"feeds": {}}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"feeds": {}}
    if "feeds" not in data:
        data["feeds"] = {}
    return data


def save_feed_state(processed_dir: Path, state: dict[str, Any]) -> None:
    path = state_path(processed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
