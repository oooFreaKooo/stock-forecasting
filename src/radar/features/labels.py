from __future__ import annotations

import numpy as np
import pandas as pd

from radar.config.schemas import LabelsConfig


def add_labels(df: pd.DataFrame, config: LabelsConfig) -> pd.DataFrame:
    """Add next-day direction, volatility regime, and setup quality labels."""
    out = df.copy()
    close = out["close"]
    next_close = close.shift(-1)
    next_return = (next_close - close) / close

    min_move = config.direction_min_move_pct
    out["next_return"] = next_return
    out["y_direction"] = (next_return > min_move).astype(int)
    out.loc[next_return < -min_move, "y_direction"] = 0
    out.loc[next_return.abs() <= min_move, "y_direction"] = np.nan

    vol_window = config.vol_regime_window
    next_vol = next_return.abs().rolling(vol_window).std().shift(-vol_window + 1)
    out["next_day_abs_return"] = next_return.abs()

    valid_vol = out["next_day_abs_return"].dropna()
    if len(valid_vol) > 0:
        q33, q66 = valid_vol.quantile([0.33, 0.66])
        out["y_vol_regime"] = pd.cut(
            out["next_day_abs_return"],
            bins=[-np.inf, q33, q66, np.inf],
            labels=[0, 1, 2],
        ).astype(float)
    else:
        out["y_vol_regime"] = np.nan

    cost_threshold = min_move
    direction_correct = ((out["y_direction"] == 1) & (next_return > 0)) | (
        (out["y_direction"] == 0) & (next_return <= 0)
    )
    out["setup_quality"] = (
        direction_correct & (next_return.abs() > cost_threshold)
    ).astype(float)

    return out


def add_cross_section_ranks(panel: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rank momentum across symbols per date (no future info)."""
    out = panel.copy()
    out["momentum"] = out.groupby("symbol")["close"].pct_change(window)
    out["momentum_rank"] = out.groupby("date")["momentum"].rank(pct=True)
    return out
