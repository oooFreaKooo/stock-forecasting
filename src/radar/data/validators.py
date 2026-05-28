from __future__ import annotations

import pandas as pd


def validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate and clean OHLCV data."""
    if df.empty:
        raise ValueError(f"Empty OHLCV for {symbol}")

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for {symbol}: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    dupes = df["date"].duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates(subset=["date"], keep="last")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid = (df["high"] < df["low"]) | (df["close"] > df["high"]) | (df["close"] < df["low"])
    if invalid.any():
        df = df.loc[~invalid]

    return df.sort_values("date").reset_index(drop=True)


def align_context_series(
    traded_df: pd.DataFrame,
    context_df: pd.DataFrame,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """Align context data to traded symbol dates."""
    merged = traded_df[["date"]].merge(context_df, on="date", how="left")
    if forward_fill:
        value_cols = [c for c in merged.columns if c != "date"]
        merged[value_cols] = merged[value_cols].ffill()
    return merged
