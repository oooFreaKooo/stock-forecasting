# Radar Dashboard (Nuxt 4 + UI Thing)

## Prerequisites

- Node.js 22+
- Yarn 4 (`corepack enable`)
- Python radar API running on port 8000

## Setup

From repo root (API extras required for the dashboard backend):

```bash
source scripts/env.sh
pip install -e ".[dev,api]"
```

```bash
cd dashboard
yarn install
```

## Run

From `dashboard/`, `yarn dev` **restarts the Python API** (`source scripts/env.sh && bash scripts/restart-api.sh`) and then starts Nuxt:

```bash
cd dashboard
yarn dev
```

Open [http://localhost:3000](http://localhost:3000)

If the API fails to start, check `/tmp/radar-api.log`. The dashboard still polls health and can trigger `scripts/ensure-api.sh` when the API goes offline later in the session.

## Features

- **Auto-start API** — dashboard starts/reloads the Python backend when offline (dev only)
- **News sidebar** — right-side UI Thing sidebar with RSS headlines, sentiment, and refresh
- **Refresh Predictions** — fetch gated v2 signals for all symbols
- **Optimize Signals** — re-run OOS optimizer (best hit rate config)
- **Reload Metrics** — show simple vs gated hit rates
- **Forecast charts** — ApexCharts historical + prediction line per symbol

## Package manager

This project uses **yarn** exclusively (`packageManager: yarn@4.15.0`).
