import pandas as pd

from radar.backtest.gated_signals import gates_from_row


def test_gates_from_row_maps_gate_columns():
    row = pd.Series(
        {
            "gate_probability": 0,
            "gate_forecast": 1,
            "gate_memory": 1,
            "gate_event": 1,
            "gate_agreement": 0,
            "gate_horizon": 1,
            "gate_momentum": 1,
            "gate_vol": 1,
            "gate_confluence": 1,
        }
    )
    gates = gates_from_row(row)
    assert gates["probability"] is False
    assert gates["forecast"] is True
    assert gates["agreement"] is False
