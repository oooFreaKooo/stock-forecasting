from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import structlog

from radar.config.schemas import MemoryConfig
from radar.config.settings import Settings
from radar.data.store import ParquetStore
from radar.memory.regime_encoder import (
    REGIME_VECTOR_COLUMNS,
    build_daily_regime_frame,
    regime_vectors_as_matrix,
)
from radar.memory.vector_store import InMemoryRegimeIndex, RegimeVectorStore

logger = structlog.get_logger(__name__)

MEMORY_FEATURE_COLUMNS = [
    "regime_sim_top1",
    "regime_sim_mean",
    "regime_neighbor_win_rate",
    "regime_neighbor_avg_return",
    "regime_vol_cluster",
]


def build_and_persist_regime_vectors(settings: Settings) -> pd.DataFrame:
    """Build regime vectors from raw data and persist to parquet + ChromaDB."""
    store = ParquetStore(settings.paths.raw_dir)
    raw_frames = store.read_all(settings.all_symbols)
    regime_df = build_daily_regime_frame(
        raw_frames,
        settings.universe.traded,
        settings.memory,
    )

    out_path = Path(settings.paths.processed_dir) / "regime_vectors.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    regime_df.to_parquet(out_path, index=False)
    logger.info("saved_regime_vectors", path=str(out_path), rows=len(regime_df))

    if settings.memory.enabled:
        vector_store = RegimeVectorStore(settings.memory.store_dir)
        count = vector_store.upsert_regimes(regime_df)
        logger.info("indexed_regime_vectors", count=count, store=settings.memory.store_dir)

    return regime_df


def load_regime_vectors(settings: Settings) -> pd.DataFrame:
    path = Path(settings.paths.processed_dir) / "regime_vectors.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Regime vectors not found at {path}. Run build_memory_index first."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _daily_market_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-date market outcomes for neighbor statistics."""
    daily = (
        panel.groupby("date")
        .agg(
            market_win_rate=("y_direction", "mean"),
            market_avg_return=("next_return", "mean"),
        )
        .reset_index()
    )
    daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")
    return daily


def compute_memory_features(
    regime_df: pd.DataFrame,
    panel: pd.DataFrame,
    memory_config: MemoryConfig,
) -> pd.DataFrame:
    """
    Compute retrieval features for each date with strict no-look-ahead.

    For date t, similar regimes are searched only among dates < t.
    """
    index = InMemoryRegimeIndex()
    index.build_from_frame(regime_df)

    daily_outcomes = _daily_market_outcomes(panel)
    outcome_map = daily_outcomes.set_index("date_str")

    vectors, date_ids = regime_vectors_as_matrix(regime_df)
    date_to_idx = {d: i for i, d in enumerate(date_ids)}

    records = []
    sorted_dates = sorted(date_ids)

    for date_str in sorted_dates:
        idx = date_to_idx[date_str]
        query_vec = vectors[idx]
        row = regime_df.iloc[idx]

        if sorted_dates.index(date_str) < memory_config.min_history_days:
            records.append({"date_str": date_str, **{c: np.nan for c in MEMORY_FEATURE_COLUMNS}})
            continue

        matches = index.query_similar(
            query_vec,
            before_date=date_str,
            top_k=memory_config.top_k,
        )

        if not matches:
            records.append({"date_str": date_str, **{c: np.nan for c in MEMORY_FEATURE_COLUMNS}})
            continue

        sims = [m.similarity for m in matches]
        neighbor_dates = [m.date for m in matches]

        win_rates = []
        avg_returns = []
        for nd in neighbor_dates:
            if nd in outcome_map.index:
                win_rates.append(outcome_map.loc[nd, "market_win_rate"])
                avg_returns.append(outcome_map.loc[nd, "market_avg_return"])

        records.append({
            "date_str": date_str,
            "regime_sim_top1": sims[0],
            "regime_sim_mean": float(np.mean(sims)),
            "regime_neighbor_win_rate": float(np.mean(win_rates)) if win_rates else np.nan,
            "regime_neighbor_avg_return": float(np.mean(avg_returns)) if avg_returns else np.nan,
            "regime_vol_cluster": float(row.get("vol_cluster", np.nan)),
        })

    memory_df = pd.DataFrame(records)
    memory_df["date"] = pd.to_datetime(memory_df["date_str"])
    return memory_df.drop(columns=["date_str"])


def enrich_panel_with_memory(
    panel: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Join memory retrieval features onto the feature panel by date."""
    regime_df = load_regime_vectors(settings)
    memory_features = compute_memory_features(regime_df, panel, settings.memory)

    enriched = panel.merge(
        memory_features,
        on="date",
        how="left",
    )
    logger.info(
        "enriched_panel_with_memory",
        rows=len(enriched),
        memory_cols=MEMORY_FEATURE_COLUMNS,
    )
    return enriched


def save_enriched_panel(panel: pd.DataFrame, settings: Settings) -> Path:
    path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
    panel.to_parquet(path, index=False)
    logger.info("saved_enriched_feature_panel", path=str(path), rows=len(panel))
    return path
