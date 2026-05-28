from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.ensemble.base_models import fit_imputer, impute_features, predict_proba, train_base_model
from radar.ensemble.meta_learner import apply_meta_learner, build_meta_features, train_meta_learner
from radar.ensemble.multi_horizon import add_multi_horizon_labels, apply_agreement_filter
from radar.features.pipeline import get_feature_columns, load_feature_panel
from radar.validation.splits import generate_splits, mask_split

logger = structlog.get_logger(__name__)


@dataclass
class EnsembleResult:
    oos_predictions: pd.DataFrame
    base_models: dict[str, Any]
    meta_model: Any
    meta_calibrator: Any
    metrics: dict[str, float]


def run_ensemble_training(settings: Settings) -> EnsembleResult:
    """Train base models + stacking meta-learner on walk-forward splits."""
    panel = load_feature_panel(settings)
    panel = add_multi_horizon_labels(panel, settings.ensemble.horizons, settings.labels.direction_min_move_pct)
    feature_cols = get_feature_columns(settings, panel)

    splits = generate_splits(
        panel["date"],
        settings.walkforward,
        data_start=pd.Timestamp(settings.data.start_date).date(),
    )
    if settings.ensemble.max_folds is not None and len(splits) > settings.ensemble.max_folds:
        splits = splits[-settings.ensemble.max_folds :]
        logger.info("ensemble_fold_limit", using=len(splits), max_folds=settings.ensemble.max_folds)

    total_folds = len(splits)
    logger.info("ensemble_training_start", folds=total_folds, base_models=settings.ensemble.base_models)

    oos_frames: list[pd.DataFrame] = []
    all_y: list[np.ndarray] = []

    for fold_idx, split in enumerate(splits, start=1):
        train_df = mask_split(panel, split, "train")
        test_df = mask_split(panel, split, "test")
        if train_df.empty or test_df.empty:
            continue

        logger.info(
            "ensemble_fold_start",
            fold=fold_idx,
            total=total_folds,
            train_end=str(split.train_end),
            test_start=str(split.test_start),
        )

        n = len(train_df)
        inner_split = int(n * 0.85)
        inner_train = train_df.iloc[:inner_split]
        inner_val = train_df.iloc[inner_split:]

        X_inner_train = inner_train[feature_cols].values
        y_inner_train = inner_train["y_direction"].values.astype(int)
        X_inner_val = inner_val[feature_cols].values
        y_inner_val = inner_val["y_direction"].values.astype(int)
        X_test = test_df[feature_cols].values
        y_test = test_df["y_direction"].values.astype(int)

        fill_values = fit_imputer(X_inner_train)
        X_inner_train = impute_features(X_inner_train, fill_values)
        X_inner_val = impute_features(X_inner_val, fill_values)
        X_test = impute_features(X_test, fill_values)

        fold_base: dict[str, np.ndarray] = {}
        val_base: dict[str, np.ndarray] = {}
        fold_models: dict[str, Any] = {}
        for model_name in settings.ensemble.base_models:
            model = train_base_model(
                model_name,
                X_inner_train,
                y_inner_train,
                seed=settings.model.random_seed,
                feature_names=feature_cols,
                fast=True,
            )
            fold_models[model_name] = model
            val_base[model_name] = predict_proba(model, X_inner_val)
            fold_base[model_name] = predict_proba(model, X_test)

        meta_features, meta_names = build_meta_features(val_base)
        meta_model, meta_cal = train_meta_learner(
            meta_features,
            y_inner_val,
            settings.ensemble.meta_model,
            settings.model.random_seed,
            feature_names=meta_names,
            fast=True,
        )
        meta_probs = apply_meta_learner(meta_model, meta_cal, fold_base)

        preds = test_df[["date", "symbol", "close", "next_return", "y_direction"]].copy()
        for name, probs in fold_base.items():
            preds[f"p_{name}"] = probs
        preds["p_ensemble"] = meta_probs
        preds["p_up"] = meta_probs
        preds["p_down"] = 1.0 - meta_probs
        for h in settings.ensemble.horizons:
            label_col = f"y_direction_{h}d"
            if h == 1:
                preds["p_up_1d"] = meta_probs
            elif label_col in train_df.columns:
                train_h = train_df.dropna(subset=[label_col])
                if len(train_h) >= 50 and train_h[label_col].nunique() > 1:
                    y_h = train_h[label_col].values.astype(int)
                    X_h = impute_features(train_h[feature_cols].values, fill_values)
                    h_model = train_base_model(
                        "logistic",
                        X_h,
                        y_h,
                        seed=settings.model.random_seed,
                        feature_names=feature_cols,
                        fast=True,
                    )
                    X_test_h = impute_features(X_test, fill_values)
                    preds[f"p_up_{h}d"] = predict_proba(h_model, X_test_h)
            if label_col in test_df.columns:
                preds[label_col] = test_df[label_col].values

        oos_frames.append(preds)
        all_y.append(y_test)
        logger.info("ensemble_fold_complete", fold=fold_idx, total=total_folds, oos_rows=len(preds))

    if not oos_frames:
        raise RuntimeError("No ensemble OOS predictions produced.")

    oos = pd.concat(oos_frames, ignore_index=True)
    oos = apply_agreement_filter(oos, settings.ensemble)

    from radar.validation.metrics import compute_classification_metrics

    metrics = compute_classification_metrics(
        oos["y_direction"].values.astype(int),
        oos["p_ensemble"].values,
    )

    models_dir = Path(settings.paths.models_dir) / "ensemble"
    models_dir.mkdir(parents=True, exist_ok=True)
    oos_path = Path(settings.paths.processed_dir) / "ensemble_oos.parquet"
    oos.to_parquet(oos_path, index=False)

    final_train = panel.dropna(subset=["y_direction"])
    X_all = final_train[feature_cols].values
    y_all = final_train["y_direction"].values.astype(int)
    all_fill = fit_imputer(X_all)
    X_all = impute_features(X_all, all_fill)

    final_base: dict[str, Any] = {}
    final_preds: dict[str, np.ndarray] = {}
    for model_name in settings.ensemble.base_models:
        m = train_base_model(
            model_name,
            X_all,
            y_all,
            seed=settings.model.random_seed,
            feature_names=feature_cols,
        )
        final_base[model_name] = m
        final_preds[model_name] = predict_proba(m, X_all)

    meta_features, meta_names = build_meta_features(final_preds)
    meta_model, meta_cal = train_meta_learner(
        meta_features,
        y_all,
        settings.ensemble.meta_model,
        settings.model.random_seed,
        feature_names=meta_names,
    )

    horizon_models: dict[str, Any] = {}
    for h in settings.ensemble.horizons:
        if h == 1:
            continue
        label_col = f"y_direction_{h}d"
        train_h = final_train.dropna(subset=[label_col])
        if len(train_h) < 50 or train_h[label_col].nunique() <= 1:
            continue
        y_h = train_h[label_col].values.astype(int)
        X_h = impute_features(train_h[feature_cols].values, all_fill)
        horizon_models[f"p_up_{h}d"] = train_base_model(
            "logistic",
            X_h,
            y_h,
            seed=settings.model.random_seed,
            feature_names=feature_cols,
        )

    from sklearn.linear_model import Ridge

    ret_train = final_train.dropna(subset=["next_return"])
    y_ret = ret_train["next_return"].values.astype(float)
    X_ret = impute_features(ret_train[feature_cols].values, all_fill)
    return_model = Ridge(alpha=1.0)
    return_model.fit(X_ret, y_ret)

    joblib.dump(
        {
            "base": final_base,
            "meta": meta_model,
            "calibrator": meta_cal,
            "feature_cols": feature_cols,
            "fill_values": all_fill,
            "base_models": list(settings.ensemble.base_models),
            "horizon_models": horizon_models,
            "return_model": return_model,
            "model_version": pd.Timestamp.now().isoformat(),
        },
        models_dir / "ensemble.joblib",
    )

    logger.info("ensemble_training_complete", auc=metrics.get("auc"), path=str(oos_path))
    return EnsembleResult(
        oos_predictions=oos,
        base_models=final_base,
        meta_model=meta_model,
        meta_calibrator=meta_cal,
        metrics=metrics,
    )


def run_full_pipeline(settings: Settings, skip_rl: bool = False) -> dict[str, str]:
    """Execute end-to-end pipeline: fetch -> features -> events -> memory -> sentiment -> train -> ensemble -> RL."""
    from radar.events.calendar_builder import build_event_calendar
    from radar.features.pipeline import build_feature_panel, enrich_memory_if_available
    from radar.nlp.fusion.memory_enricher import build_sentiment_panel
    from radar.validation.walk_forward import run_walk_forward

    results: dict[str, str] = {}

    build_event_calendar(settings)
    results["events"] = "ok"

    build_feature_panel(settings)
    results["features"] = "ok"

    enrich_memory_if_available(settings)
    results["memory"] = "ok"

    if settings.nlp.enabled:
        from pathlib import Path

        panel = build_sentiment_panel(settings)
        panel_path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
        panel.to_parquet(panel_path, index=False)
        results["sentiment"] = "ok"

    if settings.ensemble.enabled:
        ens = run_ensemble_training(settings)
        results["ensemble"] = f"AUC={ens.metrics.get('auc', float('nan')):.4f}"
    else:
        wf = run_walk_forward(settings)
        results["walk_forward"] = f"{len(wf.oos_predictions)} OOS rows"

    if settings.rl.enabled and not skip_rl:
        from radar.rl.train_sizing import train_sizing_policy
        train_sizing_policy(settings)
        results["rl"] = "ok"

    return results
