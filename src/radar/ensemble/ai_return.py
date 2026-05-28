"""Live 1d return from the trained ensemble return model (no baseline fallback)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from radar.config.settings import Settings
from radar.ensemble.live_scorer import score_live_symbol


def get_live_ai_return_1d(
    settings: Settings,
    symbol: str,
) -> tuple[Optional[float], Optional[float], Optional[dict[str, Any]]]:
    """
    Return (predicted_return_1d, p_up, full live score row).

    Uses only the saved ensemble bundle — no baseline forecast.
    """
    scores = score_live_symbol(settings, symbol)
    if scores is None:
        return None, None, None

    p_up = scores.get("p_up")
    p_up_f = float(p_up) if p_up is not None and not pd.isna(p_up) else None

    ret = scores.get("predicted_return_1d")
    if ret is None or pd.isna(ret):
        return None, p_up_f, scores

    return float(ret), p_up_f, scores
