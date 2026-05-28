# Radar Dashboard (Nuxt 4 + UI Thing)

## Prerequisites

- Node.js 22+
- Yarn 4 (`corepack enable`)
- Python radar API running on port 8000

## Setup

```bash
cd dashboard
yarn install
```

## Run

**Terminal 1 — Nuxt dashboard** (auto-starts the Python API when offline):

```bash
cd dashboard
yarn dev
```

Open [http://localhost:3000](http://localhost:3000)

Optional — start the API manually from repo root:

```bash
source scripts/env.sh && bash scripts/start-api.sh
```

The dashboard polls API health and runs `scripts/ensure-api.sh` in the background when the Python API is down or outdated. Python code changes reload automatically in dev (`RADAR_API_RELOAD=1`).

## Features

- **Auto-start API** — dashboard starts/reloads the Python backend when offline (dev only)
- **News sidebar** — right-side UI Thing sidebar with RSS headlines, sentiment, and refresh
- **Refresh Predictions** — fetch gated v2 signals for all symbols
- **Optimize Signals** — re-run OOS optimizer (best hit rate config)
- **Reload Metrics** — show simple vs gated hit rates
- **Forecast charts** — ApexCharts historical + prediction line per symbol

## Package manager

This project uses **yarn** exclusively (`packageManager: yarn@4.15.0`).
