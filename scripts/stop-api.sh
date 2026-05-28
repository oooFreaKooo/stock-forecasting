#!/usr/bin/env bash
# Gracefully stop the Radar API on RADAR_API_PORT (SIGTERM only — no SIGKILL).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${RADAR_API_PORT:-8000}"
PID_FILE="${RADAR_API_PID:-/tmp/radar-api.pid}"

if [ -f "$PID_FILE" ]; then
  kill -15 "$(cat "$PID_FILE")" 2>/dev/null || true
  for _ in $(seq 1 40); do
    if ! lsof -ti:"$PORT" >/dev/null 2>&1; then
      exit 0
    fi
    sleep 0.25
  done
fi

pids="$(lsof -ti:"$PORT" 2>/dev/null || true)"
if [ -z "$pids" ]; then
  exit 0
fi

echo "$pids" | xargs kill -15 2>/dev/null || true
for _ in $(seq 1 40); do
  if ! lsof -ti:"$PORT" >/dev/null 2>&1; then
    exit 0
  fi
  sleep 0.25
done

echo "Radar API on port ${PORT} did not stop gracefully. Close the terminal running radar-api, then retry." >&2
exit 1
