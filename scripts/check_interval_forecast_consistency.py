from __future__ import annotations

import pandas as pd

from radar.api.chart_series import get_chart_bundle
from radar.config.settings import get_settings


def _first_forecast_return(series: dict) -> float | None:
    pts = series.get("points") or []
    fc = series.get("forecast", {}).get("points") or []
    if not pts or not fc:
        return None
    last_close = float(pts[-1]["close"])
    first = float(fc[0]["close"])
    if last_close <= 0:
        return None
    return first / last_close - 1.0


def main() -> None:
    settings = get_settings("config")
    symbols = list(getattr(settings.universe, "traded", []))
    if not symbols:
        raise SystemExit("No settings.universe.traded symbols configured.")

    rows: list[dict] = []
    for sym in symbols:
        bundle = get_chart_bundle(sym, config_dir="config")
        s5 = bundle["intraday"]
        s1h = bundle["intraday_1h"]
        s1d = bundle["daily"]
        out = {"symbol": sym}
        for iv, s in (("5m", s5), ("1h", s1h), ("1d", s1d)):
            meta = s.get("meta", {})
            out[f"{iv}_p_up"] = meta.get("ai_p_up")
            out[f"{iv}_daily_ret"] = meta.get("ai_return_1d")
            out[f"{iv}_fc_first_ret"] = _first_forecast_return(s)
        rows.append(out)

    df = pd.DataFrame(rows)
    # Invariance check: daily_ret should match across intervals (same AI target).
    df["daily_ret_span"] = (
        df[["5m_daily_ret", "1h_daily_ret", "1d_daily_ret"]]
        .astype(float)
        .max(axis=1)
        - df[["5m_daily_ret", "1h_daily_ret", "1d_daily_ret"]].astype(float)
        .min(axis=1)
    )
    worst = df.sort_values("daily_ret_span", ascending=False).head(20)
    print(worst.to_string(index=False))

    bad = df[df["daily_ret_span"] > 1e-6]
    if len(bad):
        print("\nInconsistencies found:", len(bad))
        print(bad[["symbol", "5m_daily_ret", "1h_daily_ret", "1d_daily_ret", "daily_ret_span"]].to_string(index=False))
        raise SystemExit(2)

    print("\nOK: ai_return_1d matches across 5m/1h/1d for all symbols.")


if __name__ == "__main__":
    main()
