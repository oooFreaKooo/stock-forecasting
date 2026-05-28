"""
Daily ensemble walk-forward training on the feature panel.

Produces ``oos_predictions.parquet`` / ``ensemble_oos.parquet`` used by the daily
ensemble and hybrid gating.

This is NOT intraday chart backtest evaluation — see ``radar.forecast.chart_eval``
and ``radar.api.chart_validation``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from radar.config.schemas import FoldSplit
from radar.config.settings import Settings
from radar.features.pipeline import get_feature_columns, load_feature_panel
from radar.models.registry import ModelRegistry
from radar.models.supervised.trainer import train_fold
from radar.validation.splits import generate_splits, mask_split

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardResult:
    oos_predictions: pd.DataFrame
    fold_metrics: list[dict]
    splits: list[FoldSplit]


def run_walk_forward(settings: Settings) -> WalkForwardResult:
    """Execute anchored expanding walk-forward training across all folds."""
    panel = load_feature_panel(settings)
    feature_cols = get_feature_columns(settings, panel)

    splits = generate_splits(
        panel["date"],
        settings.walkforward,
        data_start=pd.Timestamp(settings.data.start_date).date(),
    )
    if not splits:
        raise RuntimeError("No walk-forward splits generated. Check date range and min_train_days.")

    logger.info("walk_forward_splits", count=len(splits))

    registry = ModelRegistry(settings.paths.models_dir)
    oos_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []

    for split in splits:
        train_df = mask_split(panel, split, "train")
        test_df = mask_split(panel, split, "test")

        if train_df.empty or test_df.empty:
            logger.warning("skip_empty_fold", fold=str(split))
            continue

        fold_result = train_fold(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            settings=settings,
            fold_id=split.fold_id,
        )

        registry.save_fold(
            fold_id=split.fold_id,
            model=fold_result["model"],
            calibrator=fold_result["calibrator"],
            feature_cols=feature_cols,
            metrics=fold_result["metrics"],
        )

        preds = fold_result["predictions"].copy()
        preds["fold_id"] = split.fold_id
        oos_frames.append(preds)
        fold_metrics.append({"fold_id": split.fold_id, **fold_result["metrics"]})
        logger.info("completed_fold", fold=str(split), **fold_result["metrics"])

    if not oos_frames:
        raise RuntimeError("No OOS predictions produced.")

    oos = pd.concat(oos_frames, ignore_index=True)
    oos = oos.sort_values(["date", "symbol"]).reset_index(drop=True)

    out_path = settings.paths.processed_dir
    from pathlib import Path
    Path(out_path).mkdir(parents=True, exist_ok=True)
    oos.to_parquet(Path(out_path) / "oos_predictions.parquet", index=False)

    return WalkForwardResult(
        oos_predictions=oos,
        fold_metrics=fold_metrics,
        splits=splits,
    )


def load_oos_predictions(settings: Settings) -> pd.DataFrame:
    from pathlib import Path

    processed = Path(settings.paths.processed_dir)
    for name in ("ensemble_oos.parquet", "oos_predictions.parquet"):
        path = processed / name
        if path.exists():
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            return df
    raise FileNotFoundError("No OOS predictions found. Run train_ensemble or train first.")
