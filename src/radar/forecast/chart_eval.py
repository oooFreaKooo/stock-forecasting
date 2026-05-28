"""
Intraday chart backtest metrics (walk-forward replay vs actual bars).

Daily ensemble walk-forward *training* lives in ``radar.validation.walk_forward``.
Chart replay logic lives in ``radar.api.chart_validation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from radar.api.chart_series import prepare_intraday_frame
from radar.api.chart_validation import build_intraday_validation
from radar.config.settings import Settings


@dataclass
class IntradayBacktestMetrics:
    symbol: str
    interval: str
    n_points: int
    mae: float
    rmse: float
    mape: float
    direction_accuracy: float
    vol_ratio: float
    segments: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "n_points": self.n_points,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "direction_accuracy": self.direction_accuracy,
            "vol_ratio": self.vol_ratio,
            "segments": self.segments,
        }


def _metrics_from_validation(
    symbol: str,
    interval: str,
    val_metrics: dict[str, Any],
) -> IntradayBacktestMetrics:
    def _f(key: str) -> float:
        v = val_metrics.get(key)
        if v is None:
            return float("nan")
        return float(v)

    return IntradayBacktestMetrics(
        symbol=symbol.upper(),
        interval=interval,
        n_points=int(val_metrics.get("n_points", 0)),
        mae=_f("mae"),
        rmse=_f("rmse"),
        mape=_f("mape"),
        direction_accuracy=_f("direction_accuracy"),
        vol_ratio=_f("vol_ratio"),
        segments=int(val_metrics.get("segments", 0)),
    )


def evaluate_intraday_backtest(
    frame: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    settings: Settings,
) -> IntradayBacktestMetrics:
    """Score walk-forward 5m LGBM predictions against actual OHLC."""
    if interval != "5m":
        raise ValueError(
            f"evaluate_intraday_backtest only supports 5m bars (got {interval!r}); "
            "resample chart output for 1h metrics."
        )
    _, val_metrics = build_intraday_validation(
        frame,
        interval,
        symbol=symbol,
        config_dir=str(settings.config_dir),
    )
    return _metrics_from_validation(symbol, interval, val_metrics)


def normalize_yfinance_frame(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    return prepare_intraday_frame(df, symbol, interval)
