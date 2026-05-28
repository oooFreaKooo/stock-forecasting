from __future__ import annotations

import pandas as pd

# Max single-bar move before treating as a bad tick (yfinance extended-hours glitches).
_MAX_BAR_MOVE = {
    "5m": 0.02,
    "1h": 0.04,
}


def sanitize_intraday_closes(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Replace isolated bad ticks where price jumps and immediately reverts.

    yfinance pre/post-market data occasionally emits single-bar spikes with volume 0.
    """
    if frame.empty or len(frame) < 3 or "close" not in frame.columns:
        return frame

    max_move = _MAX_BAR_MOVE.get(interval.lower(), 0.03)
    out = frame.copy()
    closes = out["close"].astype(float).to_numpy()
    cleaned = closes.copy()

    for i in range(1, len(closes) - 1):
        prev = float(closes[i - 1])
        cur = float(closes[i])
        nxt = float(closes[i + 1])
        if prev <= 0 or cur <= 0 or nxt <= 0:
            continue

        jump_in = abs(cur / prev - 1.0)
        jump_out = abs(nxt / cur - 1.0)
        anchor_move = abs(nxt / prev - 1.0)
        if jump_in > max_move and jump_out > max_move and anchor_move < max_move / 2:
            cleaned[i] = (prev + nxt) / 2.0

    out["close"] = cleaned
    return out
