from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yfinance as yf

from radar.api.chart_series import _normalize_intraday
from radar.config.settings import Settings
from radar.forecast.intraday_sanitize import sanitize_intraday_closes
from radar.forecast.market_hours import filter_trading_frame
from radar.intraday.event_features import ensure_events_calendar
from radar.intraday.features import build_intraday_feature_frame
from radar.intraday.model import encode_features_for_bundle, fit_intraday_model, save_bundle
from radar.intraday.news_recency import ensure_scored_headlines


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intraday 5m model (next-bar returns)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--period", default="60d", help="yfinance period for 5m data (max ~60d)")
    args = parser.parse_args()

    settings = Settings.load(config_dir=args.config_dir)
    settings.ensure_dirs()

    ensure_events_calendar(settings)
    headlines = ensure_scored_headlines(settings)
    print(f"Headlines for intraday news features: {len(headlines)} rows")

    Xs: list[pd.DataFrame] = []
    ys: list[pd.Series] = []

    for sym in settings.universe.traded:
        df = yf.Ticker(sym).history(period=args.period, interval="5m", auto_adjust=True, prepost=True)
        norm = filter_trading_frame(sanitize_intraday_closes(_normalize_intraday(df, sym), "5m"))
        feat = build_intraday_feature_frame(
            norm,
            symbol=sym,
            horizon_bars=1,
            settings=settings,
            headlines=headlines,
            use_live_sentiment=False,
        )
        if feat is None:
            continue
        Xs.append(feat.X)
        ys.append(feat.y)

    if not Xs:
        raise SystemExit("No intraday training data produced.")

    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)

    # Time-based split: last 20% of rows per symbol (chronological).
    X_eval = X.copy()
    X_eval["__idx"] = np.arange(len(X_eval))
    eval_rows: list[int] = []
    train_rows: list[int] = []
    for sym, grp in X_eval.groupby("symbol"):
        idxs = grp["__idx"].values
        cut = int(len(idxs) * 0.8)
        train_rows.extend(list(idxs[:cut]))
        eval_rows.extend(list(idxs[cut:]))

    X_train = X.iloc[train_rows].reset_index(drop=True)
    y_train = y.iloc[train_rows].reset_index(drop=True)
    X_val = X.iloc[eval_rows].reset_index(drop=True)
    y_val = y.iloc[eval_rows].reset_index(drop=True)

    bundle = fit_intraday_model(X_train, y_train, X_val=X_val, y_val=y_val)
    Xv = encode_features_for_bundle(bundle, X_val)
    pred = bundle.model_mu.predict(Xv.to_numpy(dtype=float))
    pred = np.clip(pred.astype(float), -0.05, 0.05)

    mae = float(np.mean(np.abs(pred - y_val.values)))
    dir_acc = float(np.mean((pred > 0) == (y_val.values > 0)))
    pred_vol = float(np.std(pred))
    act_vol = float(np.std(y_val.values))
    vol_ratio = float(pred_vol / act_vol) if act_vol > 1e-12 else float("nan")

    path = save_bundle(settings, bundle)
    print(f"Saved intraday model: {path}")
    print(f"Rows: {len(X)}  Features: {len(bundle.feature_cols)}  Symbols: {len(bundle.symbol_map)}")
    print(f"VAL: MAE={mae:.6f}  dir_acc={dir_acc:.4f}  vol_ratio={vol_ratio:.3f}")


if __name__ == "__main__":
    main()

