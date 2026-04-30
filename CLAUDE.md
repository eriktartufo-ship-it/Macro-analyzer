# Macro Analyzer — Architettura e Convenzioni

> **Master Guidelines globali**: prima di ogni sessione, l'IA deve aver già caricato i
> principi trasversali da `d:/Antigravity/obsidian/Erik/01_Architettura_e_Regole/AI_GUIDELINES.md`
> e i suoi file collegati. Questo CLAUDE.md contiene solo le **specificità** del progetto
> Macro Analyzer e i **delta** rispetto agli standard globali — nessun doppione.

## Descrizione

Sistema di classificazione macro-regime che analizza indicatori economici
(FRED API, Yahoo Finance, Kenneth French, Fed RSS) per determinare il regime corrente
in 4 quadranti (Reflation, Stagflation, Deflation, Goldilocks) e calcolare score
per asset class.

## Struttura

```
Macro analyzer/
├── backend/
│   ├── app/
│   │   ├── models/          # SQLAlchemy ORM models
│   │   ├── services/
│   │   │   ├── indicators/  # Fetch FRED + transforms (ROC, z-score, Kalman)
│   │   │   ├── prices/      # Yahoo Finance + bond TR sintetici 1962+
│   │   │   ├── factors/     # Fama-French regime mapping
│   │   │   ├── fomc/        # Statement/minutes LLM analyzer
│   │   │   ├── regime/      # Classifier rule-based + HMM + MS-VAR + ensemble + MC
│   │   │   ├── scoring/     # Score finale per asset class + calibration Bayesiana
│   │   │   ├── backtest/    # Portfolio simulator + lead-time NBER
│   │   │   ├── dedollarization/
│   │   │   ├── news/
│   │   │   └── config_flags.py  # USE_CALIBRATED_SCORING, USE_DEDOLLAR_BONUS
│   │   ├── api/             # FastAPI endpoints (~36 routes)
│   │   └── scheduler/       # APScheduler refresh giornaliero
│   ├── migrations/          # Alembic migrations
│   ├── tests/               # pytest (TDD obbligatorio, 194/194 passano)
│   └── seed/                # Hardcoded asset_regime_performance + calibrated JSON
├── frontend/                # React+TS+Vite, dark/light theme, 12 pannelli Data
├── .env.example
├── Makefile                 # make setup/dev/test/migrate/seed
└── requirements.txt
```

## Regimi Macro (4 quadranti: crescita × inflazione)

| Regime | Condizioni principali |
| -------------- | ------------------------------------------------------------ |
| **Reflation** | GDP forte + PMI > 50 + inflation in salita + occupazione ok |
| **Stagflation** | GDP debole + inflation alta + unemployment in salita |
| **Deflation** | GDP negativo/decelerante + inflation bassa + LEI negativo |
| **Goldilocks** | GDP moderato + inflation bassa + unemployment basso |

Pesi 0–1 per condizione, probabilità normalizzate (Σ=1.0), confidence basata su
concordanza condizioni. Penalità cross-regime allineate al Fed target 2% (CPI > 2.5
penalizza deflation/goldilocks).

## Convenzioni naming (specifiche progetto)

- Indicatori: snake_case con codice FRED (es. `gdp_roc`, `cpi_yoy`, `unrate_level`)
- Suffissi derivati: `_roc`, `_zscore`, `_ma`, `_yoy`
- Tabelle DB: snake_case plurale (es. `regime_classifications`)
- API endpoints: kebab-case sotto `/api/v1/...`

(Per il resto delle convenzioni — type hints, NO path assoluti, NO secrets — vedi
`ai_architecture.md` sez. 4.)

## Come aggiungere un nuovo indicatore

1. Codice FRED in `backend/app/services/indicators/fred_codes.py`
2. Calcolo derivato in `indicators/transforms.py`
3. **Test PRIMA del codice** in `backend/tests/test_indicators.py`
4. Aggiunta al regime classifier se influenza la classificazione
5. Migration Alembic se serve nuova colonna

## Come aggiungere un nuovo asset class

1. Aggiunta a `ASSET_CLASSES` + `ASSET_REGIME_DATA` in `scoring/engine.py`
2. Mapping ticker Yahoo + backfill_proxy in `prices/asset_universe.py`
3. Sensitivity dedollar in `dedollarization/scorer.py` se rilevante
4. Lo scoring finale lo includerà automaticamente

## Refresh dati

| Categoria | Frequenza | Fonte |
|-----------|-----------|-------|
| GDP, CPI, PCE | Mensile/Trimestrale | FRED |
| Unemployment, Claims | Settimanale/Mensile | FRED |
| Yield curve, Fed Funds, ACM term premium | Giornaliero | FRED |
| LEI (CFNAIMA3) | Mensile | FRED |
| Asset prices (15 ETF + bond TR sintetici) | Daily | Yahoo / FRED via duration |
| Fama-French factors | Mensile | Kenneth French (Dartmouth) |
| FOMC statements/minutes | ~8/anno | Fed RSS feed |

Scheduler: APScheduler, run giornaliero alle 06:00 UTC.

## Fasi (storico)

- **Fase 1**: Core regime classifier + scoring + dashboard ✅
- **Fase 2**: Dedollarization + news scoring (Groq) ✅
- **Phase 1–6 (data driven evolution)**: Yahoo integration, calibrazione Bayesiana,
  ensemble (HMM+MSVAR+rule-based), backtest+lead time, Monte Carlo + shock,
  Kalman + Fama-French + Term premium + FOMC LLM ✅

## Linee guida di sviluppo — riferimenti globali + delta progetto

I 4 principi cardine (**Pensa prima**, **Semplicità**, **Modifiche mirate**,
**TDD obbligatorio**) sono in `ai_programming_rules.md` sez. 2–5. Non li ripeto
qui. **Spec-Driven Development** (sez. 1 dello stesso file): qualunque feature non
banale richiede un breve documento di specifica concordato prima di scrivere codice
— il "vibe coding" è vietato.

### Delta progetto sui principi globali

- **Test-coverage minimo 80%** (più alto della baseline globale).
- **Backfill storico preservato dal prune**: ogni record marcato `"historical": true`
  in `conditions_met` JSON sopravvive al prune giornaliero. Senza questo, il dataset
  HMM/transition matrix sparisce ogni notte (bug critico già fixato 2026-04-25).
- **Indicators rumorosi**: usare `services/indicators/kalman.py` prima di passarli
  al classifier per ridurre outlier (es. il bug `lei_roc=30%` da CFNAI z-score).

## FinOps + Defensive Engineering per LLM (vedi `ai_finops_resilience.md`)

Il progetto fa LLM calls per FOMC analysis (Phase 6a) e news scoring (Phase 2).
Regole obbligatorie:

- **Provider cascading**: Claude (preferito, qualità superiore) → Groq llama-3.3-70b
  fallback automatico (`services/fomc/analyzer.py` segue questo pattern).
- **Structured outputs**: ogni risposta LLM passa per parsing JSON tollerante a
  markdown fences + validazione campi con clamping (vedi `_validate_analysis`).
- **Cache aggressiva su disco** per ogni documento: stessa coppia (URL, version) mai
  ri-analizzata. Pattern: cache parquet/JSON in `.cache/<service>/`.
- **Graceful degradation**: se il provider fallisce, il sistema NON crasha. Le
  feature LLM si disabilitano e gli endpoint ritornano `null`/lista vuota (non 500).
- **Caveman prompting**: system prompts telegrafici, no convenevoli. Massima densità
  informativa per token.
- **Pydantic validation**: tutti i payload API e gli output LLM strutturati passano
  per Pydantic models (già fatto in `app/api/routes.py`).

## Auto-Improvement: checklist pre-consegna

Prima di considerare un task completato (vedi `ai_self_improvement.md`):

1. I test passano e coprono il caso d'uso reale, non solo il happy path?
2. Ho rispettato le convenzioni di naming + layered architecture?
3. Se il task tocca scoring/regime, ho aggiornato i test storici (`test_historical_coherence.py`)?
4. Se il task aggiunge un endpoint, ho coerentemente esteso il frontend type+client+panel?
5. Se ho introdotto un bias (calibration, dedollar, ecc.), è opt-in via flag (env var
   o toggle UI)?
6. **Ho aggiornato la memoria persistente** (sezione successiva)?

A task chiuso, **chiedere feedback all'utente** sulle scelte non banali. Quando
l'utente fornisce una correzione, **proporre proattivamente** di aggiornare il file
`.md` pertinente (qui o nei guidelines globali) per evitare ripetizioni future.

## Workflow sync — memoria persistente (vedi `ai_workflow_sync.md`)

Memoria del progetto:

- **Locale all'agente**: `C:\Users\erikp\.claude\projects\d--Antigravity\memory\`
- **Persistente cross-PC** (Synology Drive backup): `d:/Antigravity/Antigravity/memoria/`
  con `MEMORY.md` come index e `project_macro_analyzer.md` come logbook progetto.

Al termine di un task complesso, l'IA aggiorna il file di memoria del progetto con:

1. Obiettivo della sessione
2. Decisioni tecniche e ragionamenti (i "perché")
3. File chiave toccati e perché
4. TODO espliciti per la prossima sessione

All'inizio di una sessione, l'IA legge il logbook prima di proporre soluzioni.

## Stile grafico (vedi `ai_graphic_style.md`)

Il frontend Macro Analyzer è il **reference** dei guidelines globali (Liquid Glass,
dark/light, micro-animazioni, font Inter, layout responsive). Per nuovi pannelli:

- Pattern card-based con bordi sottili e glassmorphism
- ScrollShadow per tabelle wide
- SVG charts custom per cone/timeline/heatmap (no librerie pesanti tipo Chart.js)
- Toggle nell'Header per flag globali (es. dark theme, dedollar bonus)

## Deployment (vedi `ai_deployment.md`) — TODO futuro

Attualmente il progetto gira locale (Postgres + uvicorn + vite dev). Per ship a VPS
servono `Dockerfile` backend/frontend + `docker-compose.yml`. Non ancora implementato
— da pianificare quando si vuole esporre a utenti esterni.

## Riferimenti incrociati

| Tema | File globale |
|------|-------------|
| 4 principi base (pensa, semplicità, mirate, TDD) | `ai_programming_rules.md` |
| Layered architecture, naming, type hints | `ai_architecture.md` |
| FinOps LLM + defensive engineering | `ai_finops_resilience.md` |
| Self-improvement checklist | `ai_self_improvement.md` |
| Workflow sync Obsidian | `ai_workflow_sync.md` |
| Stile grafico (frontend) | `ai_graphic_style.md` |
| Docker/VPS (TODO futuro) | `ai_deployment.md` |

I file `ai_agency_standards.md` (multi-tenant B2B) e `ai_autonomous_agency.md`
(Dify/n8n) **non si applicano** a questo progetto: è uso personale, non prodotto
commerciale, non agente autonomo 24/7.
