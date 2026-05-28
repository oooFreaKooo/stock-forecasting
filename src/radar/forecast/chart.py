from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from radar.config.settings import Settings
from radar.data.store import ParquetStore
from radar.forecast.hybrid_predictor import SymbolPrediction


def render_prediction_chart(
    settings: Settings,
    prediction: SymbolPrediction,
    output_path: Path,
) -> Path:
    """Draw historical close + forecast prediction line."""
    store = ParquetStore(settings.paths.raw_dir)
    raw = store.read(prediction.symbol)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date")

    history_days = settings.forecast.chart_history_days
    history = raw.tail(history_days)
    last_close = float(history["close"].iloc[-1])
    last_date = pd.Timestamp(history["date"].iloc[-1])
    ai_ret = prediction.forecast_return_1d or 0.0
    horizon = settings.forecast.horizon_days
    forecast_dates = [last_date + pd.offsets.BDay(i + 1) for i in range(horizon)]
    target_price = last_close * (1.0 + ai_ret)
    forecast_prices = [target_price] * horizon

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        history["date"],
        history["close"],
        color="#2563eb",
        linewidth=2,
        label="Historical close",
    )

    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "close": forecast_prices,
    })
    bridge = pd.concat([
        history[["date", "close"]].tail(1),
        forecast_df,
    ])
    ax.plot(
        bridge["date"],
        bridge["close"],
        color="#f97316",
        linewidth=2,
        linestyle="--",
        label="Forecast (ensemble AI)",
    )
    ax.scatter(
        forecast_df["date"],
        forecast_df["close"],
        color="#f97316",
        s=40,
        zorder=5,
    )

    ax.axvline(history["date"].iloc[-1], color="#94a3b8", linestyle=":", alpha=0.8)

    signal_label = "BUY" if prediction.signal else "NO TRADE"
    title = (
        f"{prediction.symbol} — Hybrid AI Radar\n"
        f"P(up)={prediction.p_up:.1%} | AI return 1d={prediction.forecast_return_1d:+.2%} | "
        f"Signal: {signal_label} ({prediction.confidence})"
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
