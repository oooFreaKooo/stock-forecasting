from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from radar.config.settings import Settings


def _html_table(data: dict[str, Any], title: str) -> str:
    rows = ""
    for key, value in data.items():
        if isinstance(value, dict):
            rows += f"<tr><td colspan='2'><strong>{key}</strong></td></tr>"
            for k, v in value.items():
                if isinstance(v, float):
                    v = f"{v:.6f}"
                rows += f"<tr><td style='padding-left:20px'>{k}</td><td>{v}</td></tr>"
        else:
            if isinstance(value, float):
                value = f"{value:.6f}"
            rows += f"<tr><td>{key}</td><td>{value}</td></tr>"
    return f"""
    <h2>{title}</h2>
    <table border="1" cellpadding="6" cellspacing="0">
        {rows}
    </table>
    """


def write_report(
    backtest_results: dict[str, Any],
    fold_metrics: list[dict],
    settings: Settings,
) -> tuple[Path, Path]:
    """Write JSON and HTML reports."""
    reports_dir = Path(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"walkforward_{timestamp}.json"
    html_path = reports_dir / f"walkforward_{timestamp}.html"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backtest": backtest_results,
        "fold_metrics": fold_metrics,
        "config": {
            "signal_threshold": settings.backtest.signal_threshold,
            "transaction_cost_bps": settings.model.transaction_cost_bps,
            "traded": settings.universe.traded,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    html = f"""<!DOCTYPE html>
<html>
<head><title>Radar Walk-Forward Report</title></head>
<body>
<h1>Hybrid AI Investment Radar — Walk-Forward Report</h1>
<p>Generated: {payload["generated_at"]}</p>
{_html_table(backtest_results.get("pooled", {}), "Pooled OOS Metrics")}
<h2>By Symbol</h2>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Symbol</th><th>Expectancy</th><th>Win Rate</th><th>N Trades</th><th>Max DD</th></tr>
"""
    for symbol, metrics in backtest_results.get("by_symbol", {}).items():
        html += (
            f"<tr><td>{symbol}</td>"
            f"<td>{metrics.get('expectancy', 0):.6f}</td>"
            f"<td>{metrics.get('win_rate', 0):.4f}</td>"
            f"<td>{metrics.get('n_trades', 0)}</td>"
            f"<td>{metrics.get('max_drawdown', 0):.4f}</td></tr>"
        )

    html += "</table><h2>Fold Classification Metrics</h2><table border='1' cellpadding='6'>"
    html += "<tr><th>Fold</th><th>AUC</th><th>Brier</th><th>Accuracy</th></tr>"
    for fm in fold_metrics:
        html += (
            f"<tr><td>{fm.get('fold_id')}</td>"
            f"<td>{fm.get('auc', float('nan')):.4f}</td>"
            f"<td>{fm.get('brier', float('nan')):.4f}</td>"
            f"<td>{fm.get('accuracy', float('nan')):.4f}</td></tr>"
        )
    html += "</table></body></html>"

    html_path.write_text(html)
    return json_path, html_path
