from __future__ import annotations

import numpy as np
import pandas as pd

from radar.config.schemas import EnsembleConfig


def add_multi_horizon_labels(df: pd.DataFrame, horizons: list[int], min_move_pct: float = 0.001) -> pd.DataFrame:
    """Add directional labels for multiple forward horizons."""
    out = df.copy()
    close = out["close"]
    for h in horizons:
        fwd_return = close.shift(-h) / close - 1
        col = f"y_direction_{h}d"
        out[col] = (fwd_return > min_move_pct).astype(float)
        out.loc[fwd_return < -min_move_pct, col] = 0.0
        out.loc[fwd_return.abs() <= min_move_pct, col] = np.nan
        out[f"fwd_return_{h}d"] = fwd_return
    return out


def horizon_agreement_gate(
    row: pd.Series,
    horizons: list[int],
    uncertainty_threshold: float,
) -> bool:
    """
    Gate trades when horizons disagree or ensemble uncertainty is high.

    uncertainty_threshold = minimum distance of p_ensemble from 0.5.
    """
    p = float(row.get("p_ensemble", row.get("p_up", 0.5)))
    if abs(p - 0.5) < uncertainty_threshold:
        return False

    dirs = []
    for h in horizons:
        prob_col = f"p_up_{h}d"
        if prob_col in row and not pd.isna(row[prob_col]):
            dirs.append(1 if float(row[prob_col]) >= 0.5 else 0)
            continue
        col = f"y_direction_{h}d"
        if col in row and not pd.isna(row[col]):
            dirs.append(int(row[col]))
    if len(dirs) < 2:
        return True
    return len(set(dirs)) == 1


def apply_agreement_filter(preds: pd.DataFrame, config: EnsembleConfig) -> pd.DataFrame:
    out = preds.copy()
    out["trade_allowed"] = out.apply(
        lambda r: horizon_agreement_gate(r, config.horizons, config.uncertainty_threshold),
        axis=1,
    )
    return out
