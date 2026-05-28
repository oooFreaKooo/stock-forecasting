from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional, Union

import pandas as pd

from radar.config.schemas import FoldSplit, WalkForwardConfig


def _parse_period_end(start: pd.Timestamp, window: str) -> pd.Timestamp:
    if window == "monthly":
        next_month = (start + pd.offsets.MonthBegin(1))
        return next_month + pd.offsets.MonthEnd(0)
    if window.endswith("D"):
        days = int(window[:-1])
        return start + timedelta(days=days - 1)
    raise ValueError(f"Unsupported window: {window}")


def _next_step(current: pd.Timestamp, step: str) -> pd.Timestamp:
    if step == "monthly":
        return current + pd.offsets.MonthBegin(1)
    if step.endswith("D"):
        return current + timedelta(days=int(step[:-1]))
    raise ValueError(f"Unsupported step: {step}")


def generate_splits(
    dates: Union[pd.Series, List[date]],
    config: WalkForwardConfig,
    data_start: Optional[date] = None,
) -> list[FoldSplit]:
    """
    Generate anchored expanding walk-forward splits.

    Train window always starts at data_start (or earliest date + min_train_days).
    Test windows advance monthly (or per step config).
    """
    date_index = pd.DatetimeIndex(pd.to_datetime(dates)).unique().sort_values()
    if len(date_index) == 0:
        return []

    if data_start is None:
        data_start = date_index.min().date()
    else:
        data_start = pd.Timestamp(data_start).date()

    min_train_end = date_index.min() + timedelta(days=config.min_train_days)
    test_start = pd.Timestamp(min_train_end.date()) + timedelta(days=config.purge_days + 1)

    splits: list[FoldSplit] = []
    fold_id = 0

    while test_start <= date_index.max():
        test_end = _parse_period_end(test_start, config.test_window)
        test_end = min(test_end, date_index.max())

        train_end = test_start - timedelta(days=config.purge_days + config.embargo_days + 1)
        train_start = pd.Timestamp(data_start)

        train_dates = date_index[(date_index >= train_start) & (date_index <= train_end)]
        test_dates = date_index[(date_index >= test_start) & (date_index <= test_end)]

        if len(train_dates) >= config.min_train_days and len(test_dates) > 0:
            fold_id += 1
            splits.append(
                FoldSplit(
                    fold_id=fold_id,
                    train_start=train_start.date(),
                    train_end=train_end.date(),
                    test_start=test_start.date(),
                    test_end=test_end.date(),
                )
            )

        test_start = _next_step(test_start, config.step)

    return splits


def mask_split(df: pd.DataFrame, split: FoldSplit, part: str) -> pd.DataFrame:
    """Return rows for train or test portion of a fold."""
    dates = pd.to_datetime(df["date"])
    if part == "train":
        mask = (dates >= pd.Timestamp(split.train_start)) & (dates <= pd.Timestamp(split.train_end))
    elif part == "test":
        mask = (dates >= pd.Timestamp(split.test_start)) & (dates <= pd.Timestamp(split.test_end))
    else:
        raise ValueError(f"part must be 'train' or 'test', got {part}")
    return df.loc[mask].copy()
