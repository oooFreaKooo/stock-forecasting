from __future__ import annotations

from typing import Optional

import pandas as pd
import structlog

from radar.config.settings import Settings
from radar.features.pipeline import load_feature_panel
from radar.memory.retrieval import MEMORY_FEATURE_COLUMNS
from radar.validation.walk_forward import load_oos_predictions

logger = structlog.get_logger(__name__)

RL_STREAM_COLUMNS = [
    "date",
    "symbol",
    "close",
    "next_return",
    "p_up",
    "p_down",
    "y_vol_regime",
    "regime_sim_mean",
    "regime_neighbor_win_rate",
    "atr_pct",
]


def _resolve_panel_column(panel: pd.DataFrame, base: str) -> Optional[str]:
    if base in panel.columns:
        return base
    matches = [c for c in panel.columns if c.startswith(f"{base}_") or c.startswith(base)]
    return matches[0] if matches else None


def build_rl_stream(settings: Settings) -> pd.DataFrame:
    """
    Merge OOS Layer-1 predictions with memory and context for RL training.

    Uses only walk-forward OOS predictions — never in-sample model outputs.
    """
    oos = load_oos_predictions(settings)
    panel = load_feature_panel(settings)

    extras: dict[str, str] = {}
    for base in ["y_vol_regime", "atr_pct", *MEMORY_FEATURE_COLUMNS]:
        resolved = _resolve_panel_column(panel, base)
        if resolved:
            extras[base] = resolved

    panel_cols = ["date", "symbol"] + list(extras.values())
    panel_subset = panel[panel_cols].drop_duplicates(subset=["date", "symbol"]).copy()
    panel_subset = panel_subset.rename(columns={v: k for k, v in extras.items()})

    stream = oos.merge(panel_subset, on=["date", "symbol"], how="left")

    if "atr_pct" not in stream.columns:
        stream["atr_pct"] = settings.rl.atr_proxy_pct
    else:
        stream["atr_pct"] = stream["atr_pct"].fillna(settings.rl.atr_proxy_pct)

    for col in ["regime_sim_mean", "regime_neighbor_win_rate", "y_vol_regime"]:
        if col not in stream.columns:
            stream[col] = 0.0
        stream[col] = stream[col].fillna(0.0)

    stream = stream.sort_values(["symbol", "date"]).reset_index(drop=True)
    stream = stream.dropna(subset=["next_return", "p_up", "p_down"])
    logger.info("built_rl_stream", rows=len(stream), symbols=stream["symbol"].nunique())
    return stream


def split_rl_stream_chronological(
    stream: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split RL stream by unique dates (chronological, no shuffle)."""
    dates = sorted(stream["date"].unique())
    split_idx = int(len(dates) * train_fraction)
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])

    train = stream[stream["date"].isin(train_dates)].copy()
    test = stream[stream["date"].isin(test_dates)].copy()
    return train, test


def episodes_by_symbol(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group RL stream into per-symbol episodes sorted by date."""
    return {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in df.groupby("symbol")
    }
