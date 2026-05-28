from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from radar.config.schemas import MemoryConfig
from radar.memory.regime_encoder import REGIME_VECTOR_COLUMNS, build_daily_regime_frame
from radar.memory.retrieval import MEMORY_FEATURE_COLUMNS, compute_memory_features
from radar.memory.vector_store import InMemoryRegimeIndex


def _make_ohlcv(symbol: str, dates: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    rng = np.random.default_rng(hash(symbol) % 2**32)
    returns = rng.normal(0.001, 0.02, len(dates))
    close = base * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
        "symbol": symbol,
    })


@pytest.fixture
def synthetic_raw_frames():
    dates = pd.bdate_range("2020-01-01", periods=300)
    symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "SPY", "QQQ", "SOXX", "^VIX"]
    frames = {}
    for i, sym in enumerate(symbols):
        base = 100 + i * 10
        if sym == "^VIX":
            base = 20
        frames[sym] = _make_ohlcv(sym, dates, base)
    return frames


def test_build_daily_regime_frame(synthetic_raw_frames):
    config = MemoryConfig(min_history_days=30)
    traded = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    regime = build_daily_regime_frame(synthetic_raw_frames, traded, config)

    assert len(regime) > 0
    assert "date" in regime.columns
    for col in REGIME_VECTOR_COLUMNS:
        assert col in regime.columns


def test_inmemory_index_no_lookahead():
    config = MemoryConfig(top_k=3, min_history_days=5)
    dates = pd.bdate_range("2020-01-01", periods=30)
    regime = pd.DataFrame({
        "date": dates,
        **{col: np.linspace(0, 1, len(dates)) + i * 0.01 for i, col in enumerate(REGIME_VECTOR_COLUMNS)},
    })

    panel = pd.DataFrame({
        "date": np.repeat(dates, 2),
        "symbol": ["AAPL", "MSFT"] * len(dates),
        "y_direction": np.random.randint(0, 2, len(dates) * 2),
        "next_return": np.random.normal(0, 0.01, len(dates) * 2),
    })

    memory = compute_memory_features(regime, panel, config)
    assert len(memory) == len(dates)
    for col in MEMORY_FEATURE_COLUMNS:
        assert col in memory.columns

    # First min_history_days rows should be NaN for similarity features
    assert memory.iloc[0]["regime_sim_top1"] != memory.iloc[0]["regime_sim_top1"] or pd.isna(
        memory.iloc[0]["regime_sim_top1"]
    )


def test_inmemory_query_respects_before_date():
    index = InMemoryRegimeIndex()
    dates = pd.bdate_range("2020-01-01", periods=10)
    regime = pd.DataFrame({
        "date": dates,
        **{col: np.arange(len(dates), dtype=float) for col in REGIME_VECTOR_COLUMNS},
    })
    index.build_from_frame(regime)

    query = np.ones(len(REGIME_VECTOR_COLUMNS), dtype=float)
    query = query / np.linalg.norm(query)
    matches = index.query_similar(query, before_date="2020-01-08", top_k=3)

    assert len(matches) <= 3
    for m in matches:
        assert m.date < "2020-01-08"


def test_chroma_store_roundtrip(tmp_path, synthetic_raw_frames):
    pytest.importorskip("chromadb")
    from radar.memory.regime_encoder import regime_vectors_as_matrix
    from radar.memory.vector_store import RegimeVectorStore

    config = MemoryConfig()
    traded = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    regime = build_daily_regime_frame(synthetic_raw_frames, traded, config)

    store = RegimeVectorStore(tmp_path / "chroma")
    count = store.upsert_regimes(regime.head(50))
    assert count == 50
    assert store.count() == 50

    vectors, _ = regime_vectors_as_matrix(regime.iloc[[50]])
    matches = store.query_similar(vectors[0], before_date="2020-03-01", top_k=3)
    assert all(m.date < "2020-03-01" for m in matches)
