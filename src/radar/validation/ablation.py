from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.features.pipeline import get_feature_columns, load_feature_panel
from radar.models.supervised.trainer import train_fold
from radar.nlp.fusion.memory_enricher import SENTIMENT_FEATURE_COLUMNS
from radar.validation.splits import generate_splits, mask_split

logger = structlog.get_logger(__name__)


@dataclass
class AblationResult:
    baseline_metrics: list[dict]
    full_metrics: list[dict]
    baseline_auc_mean: float
    full_auc_mean: float


@dataclass
class SentimentAblationResult:
    without_sentiment_auc: float
    with_sentiment_auc: float
    sentiment_lift: float


def _feature_subset(settings: Settings, panel: pd.DataFrame, include_macro_events: bool) -> list[str]:
    from radar.features.context import get_context_feature_columns
    from radar.features.events import get_event_feature_columns
    from radar.features.macro import get_macro_feature_columns
    from radar.features.technical import get_technical_feature_columns
    from radar.memory.retrieval import MEMORY_FEATURE_COLUMNS

    tech = get_technical_feature_columns(settings.features)
    ctx = get_context_feature_columns()
    cols = tech + ctx + ["momentum_rank"]
    if include_macro_events:
        cols = cols + get_macro_feature_columns() + get_event_feature_columns()
    if settings.memory.enabled:
        cols = cols + MEMORY_FEATURE_COLUMNS
    return [c for c in cols if c in panel.columns]


def run_ablation(settings: Settings, walkforward_path: Optional[str] = None) -> AblationResult:
    """
    Compare walk-forward AUC with and without macro/event features.

    Uses the same splits; only feature column sets differ.
    """
    from radar.config.settings import Settings as SettingsCls

    wf_path = walkforward_path or str(Path(settings.config_dir) / "walkforward.yaml")
    settings = SettingsCls.load(config_dir=settings.config_dir, walkforward_path=wf_path)

    panel = load_feature_panel(settings)
    baseline_cols = _feature_subset(settings, panel, include_macro_events=False)
    full_cols = _feature_subset(settings, panel, include_macro_events=True)

    splits = generate_splits(
        panel["date"],
        settings.walkforward,
        data_start=pd.Timestamp(settings.data.start_date).date(),
    )

    baseline_metrics: list[dict] = []
    full_metrics: list[dict] = []

    for split in splits:
        train_df = mask_split(panel, split, "train")
        test_df = mask_split(panel, split, "test")
        if train_df.empty or test_df.empty:
            continue

        base_result = train_fold(
            train_df, test_df, baseline_cols, settings, split.fold_id
        )
        full_result = train_fold(
            train_df, test_df, full_cols, settings, split.fold_id
        )
        baseline_metrics.append({"fold_id": split.fold_id, **base_result["metrics"]})
        full_metrics.append({"fold_id": split.fold_id, **full_result["metrics"]})

    baseline_auc = float(pd.DataFrame(baseline_metrics)["auc"].mean()) if baseline_metrics else 0.0
    full_auc = float(pd.DataFrame(full_metrics)["auc"].mean()) if full_metrics else 0.0

    report_path = Path(settings.paths.reports_dir) / "ablation_macro_events.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        pd.DataFrame({
            "variant": ["baseline", "full"],
            "auc_mean": [baseline_auc, full_auc],
        }).to_json(orient="records")
    )
    logger.info(
        "ablation_complete",
        baseline_auc=baseline_auc,
        full_auc=full_auc,
        report=str(report_path),
    )
    return AblationResult(
        baseline_metrics=baseline_metrics,
        full_metrics=full_metrics,
        baseline_auc_mean=baseline_auc,
        full_auc_mean=full_auc,
    )


def run_sentiment_ablation(settings: Settings) -> SentimentAblationResult:
    """Compare walk-forward AUC with and without NLP sentiment features."""
    panel = load_feature_panel(settings)
    base_cols = [c for c in get_feature_columns(settings, panel) if c not in SENTIMENT_FEATURE_COLUMNS]
    full_cols = get_feature_columns(settings, panel)

    splits = generate_splits(
        panel["date"],
        settings.walkforward,
        data_start=pd.Timestamp(settings.data.start_date).date(),
    )
    if settings.ensemble.max_folds:
        splits = splits[-settings.ensemble.max_folds :]

    without_metrics: list[float] = []
    with_metrics: list[float] = []

    for split in splits:
        train_df = mask_split(panel, split, "train")
        test_df = mask_split(panel, split, "test")
        if train_df.empty or test_df.empty:
            continue
        base_result = train_fold(train_df, test_df, base_cols, settings, split.fold_id)
        full_result = train_fold(train_df, test_df, full_cols, settings, split.fold_id)
        without_metrics.append(float(base_result["metrics"]["auc"]))
        with_metrics.append(float(full_result["metrics"]["auc"]))

    without_auc = float(sum(without_metrics) / len(without_metrics)) if without_metrics else 0.0
    with_auc = float(sum(with_metrics) / len(with_metrics)) if with_metrics else 0.0
    lift = with_auc - without_auc

    report_path = Path(settings.paths.reports_dir) / "ablation_sentiment.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        pd.DataFrame({
            "variant": ["without_sentiment", "with_sentiment"],
            "auc_mean": [without_auc, with_auc],
            "lift": [0.0, lift],
        }).to_json(orient="records")
    )
    logger.info("sentiment_ablation_complete", without_auc=without_auc, with_auc=with_auc, lift=lift)
    return SentimentAblationResult(without_sentiment_auc=without_auc, with_sentiment_auc=with_auc, sentiment_lift=lift)
