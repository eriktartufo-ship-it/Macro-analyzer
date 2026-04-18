#!/bin/sh
set -e

echo "[entrypoint] Applying database migrations..."
alembic upgrade head

echo "[entrypoint] Seeding reference data (idempotent)..."
python seed/seed_asset_regime.py || echo "[entrypoint] Seed skipped or already applied"

echo "[entrypoint] Starting uvicorn on 0.0.0.0:8000..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
