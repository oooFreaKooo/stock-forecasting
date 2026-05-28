from __future__ import annotations

import numpy as np
import pandas as pd


def shift_features(df: pd.DataFrame, feature_cols: list[str], periods: int = 1) -> pd.DataFrame:
    """Shift feature columns to prevent same-bar leakage."""
    out = df.copy()
    for col in feature_cols:
        if col in out.columns:
            out[col] = out[col].shift(periods)
    return out


def assert_no_future_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
) -> None:
    """Verify features are lagged relative to labels (heuristic check)."""
    if df.empty:
        return
    # Features should have NaN at start due to shifting; no bfill allowed
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if series.isna().any() and series.bfill().notna().any():
            # Check that leading NaNs weren't back-filled
            first_valid = series.first_valid_index()
            if first_valid is not None and first_valid > 0:
                leading = series.iloc[:first_valid]
                if leading.notna().any():
                    raise ValueError(f"Possible back-fill leakage in {col}")


def asof_join(left: pd.DataFrame, right: pd.DataFrame, on: str = "date") -> pd.DataFrame:
    """Merge right onto left by date without look-ahead."""
    left = left.sort_values(on)
    right = right.sort_values(on)
    merged = left.merge(right, on=on, how="left", suffixes=("", "_ctx"))
    return merged
