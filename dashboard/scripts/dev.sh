#!/usr/bin/env bash
# Start Radar API (restart) then Nuxt dev — run via: yarn dev
set -euo pipefail

DASHBOARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$DASHBOARD_DIR/.." && pwd)"

cd "$ROOT"
# shellcheck source=scripts/env.sh
source scripts/env.sh
bash scripts/restart-api.sh

cd "$DASHBOARD_DIR"
exec nuxt dev "$@"
