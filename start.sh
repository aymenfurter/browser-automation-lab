#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Activate venv
source .venv/bin/activate

# Kill any existing instance on port 8080
lsof -ti:8080 | xargs kill 2>/dev/null || true
sleep 1

# Start the dashboard
exec uvicorn web.server:app --host 0.0.0.0 --port 8080
