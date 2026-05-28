from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.data.store import ParquetStore
from radar.features.context import add_context_features, get_context_feature_columns
from radar.features.events import add_event_features, get_event_feature_columns, load_events_calendar
from radar.features.labels import add_cross_section_ranks, add_labels
from radar.features.leakage import shift_features
from radar.features.macro import add_macro_features, get_macro_feature_columns
from radar.features.technical import add_technical_features, get_technical_feature_columns
from radar.memory.retrieval import MEMORY_FEATURE_COLUMNS, enrich_panel_with_memory

logger = structlog.get_logger(__name__)

META_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
LABEL_COLUMNS = ["y_direction", "y_vol_regime", "setup_quality", "next_return"]


def build_features_for_symbol(
    traded_df: pd.DataFrame,
    context_frames: dict[str, pd.DataFrame],
    macro_frames: dict[str, pd.DataFrame],
    events: Optional[pd.DataFrame],
    settings: Settings,
) -> pd.DataFrame:
    """Build full feature matrix for one traded symbol."""
    df = traded_df.copy()
    df = add_technical_features(df, settings.features)
    df = add_context_features(df, context_frames, settings.features)
    if macro_frames:
        df = add_macro_features(df, macro_frames, settings.macro)
    if events is not None:
        df = add_event_features(df, events)
    df = add_labels(df, settings.labels)

    feature_cols = (
        get_technical_feature_columns(settings.features)
        + get_context_feature_columns()
        + get_macro_feature_columns()
        + get_event_feature_columns()
    )
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = shift_features(df, feature_cols, periods=1)

    keep = META_COLUMNS + feature_cols + LABEL_COLUMNS
    keep = [c for c in keep if c in df.columns]
    return df[keep].dropna(subset=["y_direction"])


def build_feature_panel(settings: Settings) -> pd.DataFrame:
    """Build features for all traded symbols and save processed panel."""
    store = ParquetStore(settings.paths.raw_dir)
    context_symbols = settings.universe.context
    context_frames = {
        sym: store.read(sym)
        for sym in context_symbols
        if store.exists(sym)
    }

    macro_frames = {
        sym: store.read(sym)
        for sym in settings.universe.macro
        if store.exists(sym)
    }

    events_path = Path(settings.paths.processed_dir) / "events.parquet"
    events = load_events_calendar(settings.paths.processed_dir) if events_path.exists() else None

    panels: list[pd.DataFrame] = []
    for symbol in settings.universe.traded:
        if not store.exists(symbol):
            raise FileNotFoundError(f"Missing raw data for {symbol}. Run fetch first.")
        traded_df = store.read(symbol)
        featured = build_features_for_symbol(
            traded_df, context_frames, macro_frames, events, settings
        )
        panels.append(featured)
        logger.info("built_features", symbol=symbol, rows=len(featured))

    panel = pd.concat(panels, ignore_index=True)
    rank_window = settings.features.cross_section.get("momentum_rank_window", 5)
    panel = add_cross_section_ranks(panel, window=rank_window)

    out_path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    logger.info("saved_feature_panel", path=str(out_path), rows=len(panel))
    return panel


def load_feature_panel(settings: Settings) -> pd.DataFrame:
    path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature panel not found at {path}. Run build_features first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_feature_columns(settings: Settings, df: Optional[pd.DataFrame] = None) -> list[str]:
    """Return model feature column names."""
    tech = get_technical_feature_columns(settings.features)
    ctx = get_context_feature_columns()
    macro = get_macro_feature_columns()
    events = get_event_feature_columns()
    cols = tech + ctx + macro + events + ["momentum_rank"]
    if settings.memory.enabled:
        cols = cols + MEMORY_FEATURE_COLUMNS
    if settings.nlp.enabled:
        from radar.nlp.fusion.memory_enricher import SENTIMENT_FEATURE_COLUMNS
        cols = cols + SENTIMENT_FEATURE_COLUMNS
    if df is not None:
        cols = [c for c in cols if c in df.columns]
    return cols


def enrich_memory_if_available(settings: Settings) -> Optional[pd.DataFrame]:
    """Enrich feature panel with memory features when regime index exists."""
    from pathlib import Path

    regime_path = Path(settings.paths.processed_dir) / "regime_vectors.parquet"
    if not settings.memory.enabled or not regime_path.exists():
        return None

    panel = load_feature_panel(settings)
    enriched = enrich_panel_with_memory(panel, settings)
    out_path = Path(settings.paths.processed_dir) / "feature_panel.parquet"
    enriched.to_parquet(out_path, index=False)
    logger.info("memory_enrichment_complete", rows=len(enriched))
    return enriched
