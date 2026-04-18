.PHONY: setup dev test migrate seed fetch-historical clean

# Setup iniziale: crea .env, installa dipendenze
setup:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Creato .env — compila con i tuoi valori"; fi
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt
	@echo "Setup completato. Attiva il venv: source .venv/Scripts/activate"

# Avvia in dev mode
dev:
	cd backend && ../.venv/Scripts/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Esegue test suite completa
test:
	cd backend && ../.venv/Scripts/python -m pytest -v --tb=short

# Test con coverage
test-cov:
	cd backend && ../.venv/Scripts/python -m pytest --cov=app --cov-report=html --cov-report=term

# Applica migrazioni Alembic
migrate:
	cd backend && ../.venv/Scripts/alembic upgrade head

# Genera nuova migrazione
migration:
	cd backend && ../.venv/Scripts/alembic revision --autogenerate -m "$(msg)"

# Carica dati iniziali hardcoded
seed:
	cd backend && ../.venv/Scripts/python seed/seed_asset_regime.py

# Scarica dati storici da FRED (bootstrap iniziale)
fetch-historical:
	cd backend && ../.venv/Scripts/python -c "from app.services.indicators.fetcher import FredFetcher; f = FredFetcher(); print('Fetching...'); [f.fetch_and_transform(s) for s in ['real_gdp', 'cpi', 'unrate', 'ism_manufacturing', 'initial_claims', 'lei', 'fed_funds', 'yield_curve_10y2y']]; print('Done')"

# Trigger refresh manuale
refresh:
	cd backend && ../.venv/Scripts/python -c "from app.scheduler.jobs import daily_refresh; daily_refresh()"

# Crea database PostgreSQL
createdb:
	createdb macro_analyzer

# Pulisci cache e file temporanei
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .venv
