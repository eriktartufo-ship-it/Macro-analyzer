# Macro Analyzer

Sistema di classificazione macro-regime e scoring per asset class.
Analizza indicatori economici (FRED API) per determinare il regime macro corrente
e calcolare score ottimali per 15 asset class.

## Regimi

| Regime | Descrizione |
|--------|------------|
| Growth | Espansione economica forte |
| Slowdown | Decelerazione pre-recessiva |
| Recession | Contrazione economica |
| Recovery | Ripresa post-recessione |
| Stagflation | Crescita bassa + inflazione alta |
| Goldilocks | Condizioni ottimali (crescita moderata, inflazione bassa) |

## Setup rapido (5 comandi)

```bash
# 1. Installa PostgreSQL 16+ e crea il database
createdb macro_analyzer

# 2. Setup progetto (crea venv, installa dipendenze, crea .env)
make setup

# 3. Compila .env con la tua FRED API key
#    (ottienila gratis su https://fred.stlouisfed.org/docs/api/api_key.html)

# 4. Applica migrazioni e carica seed data
make migrate && make seed

# 5. Avvia il server
make dev
```

Il server parte su http://localhost:8000 — docs interattive su http://localhost:8000/docs

## API Endpoints

| Metodo | Path | Descrizione |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/regime/current` | Regime corrente con probabilita |
| GET | `/api/v1/regime/history?days=30` | Storico regimi |
| GET | `/api/v1/signals/current` | Score correnti per asset |
| GET | `/api/v1/scoreboard` | Dashboard completa |
| POST | `/api/v1/regime/classify` | Classifica da indicatori custom |
| POST | `/api/v1/refresh` | Trigger refresh manuale |
| GET | `/api/v1/assets` | Lista asset class |

## Comandi Make

```bash
make setup              # Setup iniziale
make dev                # Avvia server dev (porta 8000)
make test               # Esegui tutti i test
make test-cov           # Test con coverage
make migrate            # Applica migrazioni DB
make seed               # Carica dati performance storiche
make fetch-historical   # Scarica storico FRED
make refresh            # Trigger refresh manuale
```

## Struttura

```
Macro analyzer/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI endpoints
│   │   ├── models/          # SQLAlchemy ORM
│   │   ├── scheduler/       # APScheduler (refresh giornaliero 06:00 UTC)
│   │   └── services/
│   │       ├── indicators/  # FRED fetch + ROC, z-score, YoY
│   │       ├── regime/      # Classifier (6 regimi, pesi, confidence)
│   │       └── scoring/     # Score 0-100 per 15 asset class
│   ├── migrations/          # Alembic
│   ├── seed/                # Dati storici hardcoded
│   └── tests/               # pytest (TDD)
├── frontend/                # Dashboard (dark mode) — coming soon
├── .env.example
├── Makefile
└── requirements.txt
```

## Asset Class monitorati

US Equities Growth, US Equities Value, International DM Equities,
EM Equities, US Bonds Short, US Bonds Long, TIPS/Inflation Bonds,
Gold, Silver, Broad Commodities, Energy, Real Estate (REITs),
Cash/Money Market, Bitcoin, Crypto Broad

## Deploy con Docker + Traefik (VPS)

Il progetto include `Dockerfile` per backend e frontend + `docker-compose.yml`
che orchestra Postgres + backend (FastAPI + APScheduler) + frontend (nginx),
**pronto per essere esposto da Traefik con HTTPS automatico Let's Encrypt**.

### Prerequisiti sul VPS

- Docker 24+ e Docker Compose plugin
- Traefik già in esecuzione con:
  - network `traefik-net` esterno
  - entrypoints `web` (80) + `websecure` (443)
  - cert resolver `letsencrypt`
- Record DNS A del sottodominio (es. `macro.pieranlab.cloud`) che punta all'IP del VPS

### Setup

```bash
# 1. Clona il repo
git clone https://github.com/<tuo-user>/macro-analyzer.git
cd macro-analyzer

# 2. Crea .env e compila le chiavi API
cp .env.example .env
nano .env
# Variabili da impostare:
#   APP_DOMAIN=macro.pieranlab.cloud
#   FRED_API_KEY=...
#   GEMINI_API_KEY=...
#   GROQ_API_KEY=...
#   POSTGRES_PASSWORD=... (robusta)

# 3. Build + avvio in background
docker compose up -d --build

# 4. Verifica stato + cert Let's Encrypt
docker compose ps
docker compose logs -f frontend
docker compose logs -f backend
```

Traefik emette automaticamente il certificato TLS al primo accesso
e dirige `https://macro.pieranlab.cloud/` → frontend → API proxy → backend.

> Il frontend è collegato a due network: `traefik-net` (per essere raggiunto
> da Traefik) e `macro-internal` (per parlare con backend e postgres).
> Postgres e backend **non** sono esposti su `traefik-net`: restano isolati.

### Refresh automatico

Lo scheduler APScheduler parte insieme al backend e lancia `daily_refresh`
all'ora definita da `SCHEDULER_HOUR` / `SCHEDULER_MINUTE` (UTC).
Default **06:00 UTC**. Per mezzanotte UTC: `SCHEDULER_HOUR=0`, `SCHEDULER_MINUTE=0`.

Il backend deve restare sempre acceso: `restart: unless-stopped` in compose
garantisce il riavvio dopo reboot o crash.

### Aggiornamenti

```bash
git pull
docker compose up -d --build
```

Le migrazioni Alembic vengono applicate automaticamente all'avvio del container backend.

### Persistenza

Due volume Docker conservano lo stato tra i restart:

- `pgdata` — database PostgreSQL
- `fred-cache` — cache su disco delle serie FRED (velocizza i refresh successivi)

## Roadmap

- [x] Fase 1: Core regime classifier + scoring + API
- [x] Fase 2: Dedollarization layer + news scoring
- [x] Dashboard frontend (dark mode)
- [x] Docker compose per deploy
- [ ] Fase 3: YouTube transcription DB + sentiment analysis
- [ ] HTTPS + reverse-proxy (Caddy/Traefik) con auto-TLS
