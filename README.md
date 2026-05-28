# Hybrid AI Investment Radar

Phase 1 implementation: supervised directional prediction with anchored walk-forward validation and expectancy backtesting.

## Quick Start

```bash
# Create virtual environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# macOS: LightGBM requires libomp (one-time)
brew install libomp
source scripts/env.sh

# 1. Fetch market data (yfinance)
python -m radar.cli.fetch_data

# 2. Build feature panel
python -m radar.cli.build_features

# 3. Train walk-forward folds
python -m radar.cli.train --config config/walkforward.yaml

# 4. Run expectancy backtest + report
python -m radar.cli.backtest --report
```

## Universe

- **Traded:** AAPL, MSFT, NVDA, GOOGL, AMZN
- **Context:** SPY, QQQ, SOXX, ^VIX

## Architecture

1. **Layer 1 (Phase 1):** LightGBM directional classifier with isotonic calibration
2. **Layer 2 (Phase 2):** Semantic regime memory — ChromaDB index of macro regime vectors with similarity retrieval
3. **Layer 3 (Phase 3):** RL position sizing (PPO) — optimizes size and stops from Layer 1 + memory outputs

## Phase 2 — Semantic Memory

After building base features, index macro regimes and enrich the panel:

```bash
# Build regime vectors + ChromaDB index + attach memory features
python -m radar.cli.build_memory_index

# Or build features then enrich in one step (if index already exists)
python -m radar.cli.build_features --with-memory

# Retrain with memory-augmented features
python -m radar.cli.train --config config/walkforward.yaml
```

Memory features added per date (no look-ahead):
- `regime_sim_top1`, `regime_sim_mean` — similarity to historical macro regimes
- `regime_neighbor_win_rate`, `regime_neighbor_avg_return` — outcomes on similar past days
- `regime_vol_cluster` — VIX percentile bucket (low/med/high vol)

## Phase 3 — RL Position Sizing

RL optimizes position size and stop-loss multiplier using Layer 1 probabilities + memory — never raw price prediction:

```bash
# Train PPO sizing policy on OOS prediction stream
python -m radar.cli.train_rl

# Evaluate on held-out chronological split
python -m radar.cli.evaluate_rl
```

State: `p_up`, `p_down`, vol regime, setup quality, exposure, drawdown, regime similarity.
Actions: position target (0–100%) and stop ATR multiplier.
Reward: risk-adjusted return with drawdown, volatility, and turnover penalties.

## Phase 4 — Macro & Events

Cross-asset macro features (rates curve, credit stress, USD trend) and macro event calendar:

```bash
# Fetch includes macro symbols from config/macro.yaml (^TNX, HYG, TLT, UUP, etc.)
python -m radar.cli.fetch_data

# Build FOMC/CPI/NFP event calendar
python -m radar.cli.build_event_calendar

# Rebuild features (macro + event flags joined)
python -m radar.cli.build_features

# Ablation: baseline vs macro+events walk-forward AUC
python -m radar.cli.ablation
```

RL sizing caps exposure on event days (`max_size_on_event_day: 0.25` in `config/rl.yaml`).

## Phase 5 — NLP & Alt Data

Daily sentiment aggregates (RSS + VADER), GDELT geo flags, and IV proxy features:

```bash
python -m radar.cli.build_sentiment_index
```

## Phase 6 — Ensemble & Orchestrator

Stacking meta-learner (LightGBM + XGBoost + Logistic) with multi-horizon agreement gating:

```bash
python -m radar.cli.train_ensemble

# Full pipeline (events → features → memory → sentiment → train → ensemble)
python -m radar.cli.run_pipeline --full
```

## Validation

Strict anchored expanding walk-forward splits — no random cross-validation or shuffling.

Expectancy: `E = (P_w × A_w) - (P_l × A_l)`

## Tests

```bash
pytest
```

## Hybrid Prediction + Chart

Generate a forecast line chart and high-precision trade signal:

```bash
# Single symbol chart + signal
python -m radar.cli.predict --symbol AAPL

# All traded symbols
python -m radar.cli.predict --all

# Compare gated vs simple hit rate on OOS data
python -m radar.cli.predict --eval

# Backtest with gated high-precision signals
python -m radar.cli.backtest --gated --report
```

**Optional — Amazon Chronos** (better forecast line, needs download on first run):

```bash
pip install -e ".[forecast]"
# set forecast.engine: chronos in config/forecast.yaml
```

### How hit rate improves

The model alone is ~50% accurate on all days. The **gated hybrid** system trades less often by requiring:
- High model confidence (optimized threshold)
- Forecast direction agreement
- Favorable memory regime (similar past days won)
- No macro event day

This typically raises win rate on **taken trades** at the cost of fewer trades.
