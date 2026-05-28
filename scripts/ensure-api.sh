#!/usr/bin/env bash
# Start the Radar API if down; optionally stop a running instance first (force=1).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/env.sh
source scripts/env.sh

FORCE="${1:-0}"
PORT="${RADAR_API_PORT:-8000}"
LOG_FILE="${RADAR_API_LOG:-/tmp/radar-api.log}"
PID_FILE="${RADAR_API_PID:-/tmp/radar-api.pid}"

if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  if [ "$FORCE" != "1" ]; then
    echo "Radar API already running on http://127.0.0.1:${PORT}"
    exit 0
  fi
  "$ROOT/scripts/stop-api.sh"
elif lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "Port ${PORT} in use but /health failed — stopping stale process..." >&2
  "$ROOT/scripts/stop-api.sh"
fi

if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "Port ${PORT} is still in use. Close the process using it and retry." >&2
  exit 1
fi

if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "API dependencies missing. From repo root run: pip install -e \".[api]\"" >&2
  exit 1
fi

# PyTorch/Chronos is unstable with uvicorn auto-reload (worker restarts mid-inference).
export RADAR_API_RELOAD=0
nohup radar-api >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Radar API ready on http://127.0.0.1:${PORT}"
    exit 0
  fi
  sleep 0.25
done

echo "Radar API failed to start. See ${LOG_FILE}" >&2
exit 1
