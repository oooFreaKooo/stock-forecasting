from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from radar.config.settings import get_settings


def _fetch_json(url: str) -> dict[str, Any]:
    raw = subprocess.check_output(["curl", "-s", url]).decode("utf-8")
    return json.loads(raw)


@dataclass
class EvalRow:
    symbol: str
    interval: str
    n: int
    mae: float
    dir_acc: float
    vol_ratio: float
    worst_abs_err: float
    worst_ts: str


def _eval_series(actual_points: list[dict], pred_points: list[dict], *, window_start: Optional[pd.Timestamp] = None) -> Optional[EvalRow]:
    if not actual_points or not pred_points:
        return None
    a = pd.DataFrame(actual_points)
    p = pd.DataFrame(pred_points)
    if a.empty or p.empty:
        return None
    a["ts"] = pd.to_datetime(a["date"])
    p["ts"] = pd.to_datetime(p["date"])
    a = a.dropna(subset=["close"]).copy()
    p = p.dropna(subset=["close"]).copy()
    a["close"] = a["close"].astype(float)
    p["close"] = p["close"].astype(float)

    merged = pd.merge(a[["ts", "close"]], p[["ts", "close"]], on="ts", suffixes=("_act", "_pred"))
    if window_start is not None:
        merged = merged[merged["ts"] >= window_start]
    if len(merged) < 5:
        return None

    err = merged["close_pred"] - merged["close_act"]
    mae = float(np.mean(np.abs(err)))

    prior = merged["close_act"].shift(1)
    dir_acc = float(np.mean(((merged["close_pred"] > prior) == (merged["close_act"] > prior)).iloc[1:]))

    # volatility ratio: predicted step abs mean / actual step abs mean
    act_step = merged["close_act"].diff().abs()
    pred_step = merged["close_pred"].diff().abs()
    denom = float(np.mean(act_step.iloc[1:]))
    vol_ratio = float(np.mean(pred_step.iloc[1:]) / denom) if denom > 1e-12 else float("nan")

    worst_idx = int(np.abs(err).idxmax())
    worst_abs = float(np.abs(err.loc[worst_idx]))
    worst_ts = merged.loc[worst_idx, "ts"].strftime("%Y-%m-%dT%H:%M:%SZ")
    return EvalRow(
        symbol="",
        interval="",
        n=int(len(merged)),
        mae=mae,
        dir_acc=dir_acc,
        vol_ratio=vol_ratio,
        worst_abs_err=worst_abs,
        worst_ts=worst_ts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chart backtest vs actual")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--hours", type=int, default=24, help="Evaluate last N hours (intraday)")
    args = parser.parse_args()

    settings = get_settings("config")
    symbols = list(settings.universe.traded)
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=args.hours)

    rows: list[EvalRow] = []
    for sym in symbols:
        bundle = _fetch_json(f"{args.base_url}/api/chart/{sym}/bundle")
        views = [
            ("5m", bundle["intraday"]),
            ("1h", bundle["intraday_1h"]),
            ("1d", bundle["daily"]),
        ]
        for interval, d in views:
            actual = d.get("points", [])
            pred = (d.get("validation") or {}).get("points", [])
            w = since if interval in ("5m", "1h") else None
            r = _eval_series(actual, pred, window_start=w)
            if r is None:
                continue
            r.symbol = sym
            r.interval = interval
            rows.append(r)

    if not rows:
        raise SystemExit("No evaluation rows produced.")

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.sort_values(["interval", "mae"], ascending=[True, True])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Quick aggregate summary
    print("\nAggregate by interval")
    agg = df.groupby("interval").agg(
        symbols=("symbol", "nunique"),
        n=("n", "sum"),
        mae=("mae", "mean"),
        dir_acc=("dir_acc", "mean"),
        vol_ratio=("vol_ratio", "mean"),
        worst_abs_err=("worst_abs_err", "max"),
    )
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

