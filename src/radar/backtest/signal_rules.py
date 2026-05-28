from __future__ import annotations

import numpy as np
import pandas as pd


def apply_signal_rule(
    predictions: pd.DataFrame,
    threshold: float = 0.55,
) -> pd.DataFrame:
    """Long when p_up > threshold, flat otherwise."""
    out = predictions.copy()
    out["signal"] = (out["p_up"] > threshold).astype(int)
    return out
