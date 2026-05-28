from __future__ import annotations

from typing import Any

import pandas as pd

from radar.config.settings import Settings


def apply_portfolio_limits(
    predictions: list[dict[str, Any]],
    settings: Settings,
) -> list[dict[str, Any]]:
    """
    Enforce top-N symbols and max single-name weight on BUY signals.

    Lower-ranked BUY signals are demoted to WAIT when limits bind.
    """
    if not predictions:
        return predictions

    cfg = settings.ensemble
    out = [dict(p) for p in predictions]
    buy_indices = [
        i for i, p in enumerate(out)
        if p.get("signal") == 1 and "error" not in p
    ]
    if not buy_indices:
        return out

    ranked = sorted(
        buy_indices,
        key=lambda i: (
            float(out[i].get("confluence_score") or out[i].get("p_up") or 0),
            float(out[i].get("p_up") or 0),
        ),
        reverse=True,
    )

    max_names = cfg.top_n_symbols if cfg.top_n_symbols > 0 else len(ranked)
    allowed = set(ranked[:max_names])
    max_weight = cfg.max_single_name_weight

    for i in buy_indices:
        if i not in allowed:
            out[i]["signal"] = 0
            out[i]["action"] = "WAIT"
            out[i]["confidence"] = "none"
            out[i]["portfolio_blocked"] = True
            out[i]["portfolio_reason"] = "top_n_limit"
        else:
            size = float(out[i].get("position_size") or max_weight)
            out[i]["position_size"] = min(size, max_weight)

    gross = sum(float(out[i].get("position_size") or 0) for i in allowed)
    max_gross = cfg.max_gross_exposure
    if gross > max_gross and gross > 0:
        scale = max_gross / gross
        for i in allowed:
            out[i]["position_size"] = float(out[i].get("position_size") or max_weight) * scale
            out[i]["portfolio_scaled"] = True

    return out
