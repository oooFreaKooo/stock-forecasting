from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class UniverseConfig(BaseModel):
    traded: list[str]
    context: list[str]
    macro: list[str] = Field(default_factory=list)


class MacroParamsConfig(BaseModel):
    trend_window: int = 20
    curve_symbols: dict = Field(default_factory=dict)


class EventsConfig(BaseModel):
    seed_path: str = "src/radar/events/seed/macro_dates.csv"
    earnings_lookahead_days: int = 7
    geo_seed_path: str = "src/radar/events/seed/geo_flags.csv"


class NLPConfig(BaseModel):
    enabled: bool = True
    rss_feeds: list[str] = Field(default_factory=list)
    gdelt_seed_path: str = "src/radar/events/seed/geo_flags.csv"
    use_finbert: bool = False
    sentiment_window: int = 5


class EnsembleConfig(BaseModel):
    enabled: bool = True
    base_models: list[str] = Field(default_factory=lambda: ["lightgbm", "xgboost", "logistic"])
    meta_model: str = "lightgbm"
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 20])
    uncertainty_threshold: float = 0.15
    max_single_name_weight: float = 0.35
    max_gross_exposure: float = 1.0
    top_n_symbols: int = 3
    max_folds: Optional[int] = None


class ForecastConfig(BaseModel):
    enabled: bool = True
    horizon_days: int = 5
    context_days: int = 120
    chart_history_days: int = 252
    daily_validation_blend: float = 0.55
    daily_validation_context_days: int = 15
    intraday_context_bars_5m: int = 256
    intraday_horizon_bars_5m: int = 78
    intraday_validation_horizon_5m: int = 8
    intraday_context_bars_1h: int = 120
    intraday_horizon_bars_1h: int = 24


class HybridConfig(BaseModel):
    enabled: bool = True
    min_probability: float = 0.58
    min_memory_win_rate: float = 0.52
    require_forecast_agreement: bool = True
    skip_event_days: bool = True
    optimize_threshold: bool = True
    min_trades_for_threshold: int = 30
    require_model_agreement: bool = True
    max_model_disagreement: float = 0.06
    require_multi_horizon: bool = True
    min_momentum_rank: float = 0.55
    max_vol_regime: int = 1
    max_vix_level: Optional[float] = 35.0
    min_confluence_score: float = 0.0
    top_n_per_day: int = 2
    auto_optimize: bool = True


class DataConfig(BaseModel):
    start_date: str
    end_date: Optional[str] = None
    interval: str = "1d"
    source: str = "yfinance"


class JobsConfig(BaseModel):
    enabled: bool = True
    bootstrap_on_startup: bool = True
    run_in_api: bool = True
    poll_seconds: int = 30
    news_interval_minutes: int = 20
    intraday_interval_minutes: int = 15
    daily_interval_minutes: int = 60
    intraday_period: str = "5d"
    intraday_max_age_minutes: int = 20


class PathsConfig(BaseModel):
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "artifacts/models"
    reports_dir: str = "artifacts/reports"


class ModelConfig(BaseModel):
    type: Literal["lightgbm", "xgboost"] = "lightgbm"
    direction_threshold: float = 0.55
    transaction_cost_bps: float = 5.0
    val_fraction: float = 0.1
    random_seed: int = 42


class LabelsConfig(BaseModel):
    direction_min_move_pct: float = 0.001
    vol_regime_window: int = 20


class BacktestConfig(BaseModel):
    signal_threshold: float = 0.55
    initial_capital: float = 1_000_000.0


class WalkForwardConfig(BaseModel):
    mode: Literal["anchored_expanding"] = "anchored_expanding"
    min_train_days: int = 504
    test_window: str = "monthly"
    step: str = "monthly"
    purge_days: int = 1
    embargo_days: int = 0


class FeatureParamsConfig(BaseModel):
    returns: dict = Field(default_factory=dict)
    momentum: dict = Field(default_factory=dict)
    volatility: dict = Field(default_factory=dict)
    volume: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)
    cross_section: dict = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    enabled: bool = True
    store_dir: str = "artifacts/memory"
    top_k: int = 5
    min_history_days: int = 60
    cluster_count: int = 4
    similarity_metric: Literal["cosine", "euclidean"] = "cosine"
    correlation_window: int = 20
    vix_zscore_window: int = 20
    vol_cluster_window: int = 252


class RLRewardConfig(BaseModel):
    lambda_dd: float = 2.0
    dd_threshold: float = 0.05
    lambda_vol: float = 0.5
    lambda_turn: float = 0.1
    lambda_sortino: float = 0.25
    sortino_window: int = 20


class RLConfig(BaseModel):
    enabled: bool = True
    models_dir: str = "artifacts/rl"
    algorithm: Literal["ppo", "a2c"] = "ppo"
    total_timesteps: int = 100_000
    train_fraction: float = 0.7
    max_position: float = 1.0
    position_buckets: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    stop_atr_buckets: list[float] = Field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])
    atr_proxy_pct: float = 0.02
    drawdown_window: int = 20
    max_size_on_event_day: float = 0.25
    reward: RLRewardConfig = Field(default_factory=RLRewardConfig)


@dataclass(frozen=True)
class FoldSplit:
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @property
    def purge_end(self) -> date:
        """Last date usable for training before purge gap."""
        from datetime import timedelta

        return self.train_end - timedelta(days=0)

    def __str__(self) -> str:
        return (
            f"fold_{self.fold_id}: train[{self.train_start}..{self.train_end}] "
            f"test[{self.test_start}..{self.test_end}]"
        )
