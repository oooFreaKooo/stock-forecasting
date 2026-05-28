from __future__ import annotations

from typing import Optional

import numpy as np


def fractional_kelly_size(
    p_up: float,
    confluence: float,
    max_weight: float = 0.35,
    edge_assumption: float = 0.02,
) -> float:
    """
    Simple fractional Kelly position size from probability and confluence.

    Caps at max_weight; returns 0 when p_up <= 0.5.
    """
    if p_up <= 0.5:
        return 0.0
    edge = (p_up - 0.5) * 2 * edge_assumption
    kelly = edge / max(edge_assumption, 1e-6)
    size = kelly * 0.25 * (0.5 + 0.5 * confluence)
    return float(np.clip(size, 0.0, max_weight))
