from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from radar.config.schemas import (
    BacktestConfig,
    DataConfig,
    EnsembleConfig,
    EventsConfig,
    FeatureParamsConfig,
    ForecastConfig,
    HybridConfig,
    LabelsConfig,
    MacroParamsConfig,
    MemoryConfig,
    ModelConfig,
    NLPConfig,
    PathsConfig,
    RLConfig,
    UniverseConfig,
    WalkForwardConfig,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RADAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    config_dir: Path = Field(default=Path("config"))
    data_dir: Path = Field(default=Path("data"))
    artifacts_dir: Path = Field(default=Path("artifacts"))
    log_level: str = "INFO"

    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    walkforward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    features: FeatureParamsConfig = Field(default_factory=FeatureParamsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    macro: MacroParamsConfig = Field(default_factory=MacroParamsConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

    @classmethod
    def load(
        cls,
        config_dir: Union[Path, str] = "config",
        walkforward_path: Optional[Union[Path, str]] = None,
    ) -> Settings:
        config_dir = Path(config_dir)
        merged: dict[str, Any] = {}
        for name in ("default.yaml", "features.yaml", "memory.yaml", "rl.yaml", "macro.yaml", "nlp.yaml", "ensemble.yaml", "forecast.yaml"):
            path = config_dir / name
            if path.exists():
                merged = _merge_dicts(merged, _load_yaml(path))

        if walkforward_path is not None:
            wf = _load_yaml(Path(walkforward_path))
            merged = _merge_dicts(merged, wf)
        elif (config_dir / "walkforward.yaml").exists():
            merged = _merge_dicts(merged, _load_yaml(config_dir / "walkforward.yaml"))

        return cls(**merged)

    @property
    def all_symbols(self) -> list[str]:
        return list(dict.fromkeys(
            self.universe.traded + self.universe.context + self.universe.macro
        ))

    def ensure_dirs(self) -> None:
        for path in (
            self.paths.raw_dir,
            self.paths.processed_dir,
            self.paths.models_dir,
            self.paths.reports_dir,
            self.memory.store_dir,
            self.rl.models_dir,
            "data/cache",
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


def get_settings(config_dir: Union[Path, str] = "config") -> Settings:
    return Settings.load(config_dir=config_dir)
