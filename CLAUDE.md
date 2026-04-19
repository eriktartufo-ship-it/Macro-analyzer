# Macro Analyzer — Architettura e Convenzioni

## Descrizione

Sistema di classificazione macro-regime che analizza indicatori economici
(FRED API) per determinare il regime corrente in 4 quadranti
(Reflation, Stagflation, Deflation, Goldilocks) e calcolare score per asset class.

## Struttura

```
Macro analyzer/
├── backend/
│   ├── app/
│   │   ├── models/          # SQLAlchemy ORM models
│   │   ├── services/
│   │   │   ├── indicators/  # Fetch FRED + calcolo ROC, z-score
│   │   │   ├── regime/      # Classificatore regimi (pesi + confidence)
│   │   │   └── scoring/     # Score finale per asset class
│   │   ├── api/             # FastAPI endpoints
│   │   └── scheduler/       # APScheduler refresh giornaliero
│   ├── migrations/          # Alembic migrations
│   ├── tests/               # pytest (TDD obbligatorio)
│   └── seed/                # Dati hardcoded asset_regime_performance
├── frontend/                # Dashboard dark mode
├── .env.example
├── Makefile
└── requirements.txt
```

## Regimi Macro (4 quadranti: crescita x inflazione)

| Regime | Condizioni principali |
| -------------- | ------------------------------------------------------------ |
| **Reflation** | GDP forte + PMI > 50 + inflation in salita + occupazione ok |
| **Stagflation** | GDP debole + inflation alta + unemployment in salita |
| **Deflation** | GDP negativo/decelerante + inflation bassa + LEI negativo |
| **Goldilocks** | GDP moderato + inflation bassa + unemployment basso |

Ogni condizione ha un peso (0-1). Le probabilita dei regimi sono normalizzate (somma = 1.0).
Confidence score (0-1) basato su quante condizioni concordano.

## Convenzioni di naming

- Indicatori: snake_case con codice FRED (es. `gdp_roc`, `cpi_yoy`, `unrate_level`)
- Suffissi derivati: `_roc` (rate of change), `_zscore`, `_ma` (moving average), `_yoy` (year over year)
- Tabelle DB: snake_case plurale (es. `macro_indicators`, `regime_classifications`)
- API endpoints: `/api/v1/...` con kebab-case

## Come aggiungere un nuovo indicatore

1. Aggiungi il codice FRED in `backend/app/services/indicators/fred_codes.py`
2. Definisci il calcolo derivato (ROC, z-score) in `indicators/transforms.py`
3. Scrivi test in `backend/tests/test_indicators.py` (TDD: test PRIMA)
4. Aggiungi al regime classifier se influenza la classificazione
5. Esegui migration se serve nuova colonna

## Come aggiungere un nuovo asset class

1. Aggiungi alla tabella `asset_regime_performance` (seed data)
2. Definisci hit_rate, avg_return_12m, volatility, sharpe per ogni regime
3. Il scoring finale lo includera automaticamente

## Refresh dati

| Categoria | Frequenza | Fonte |
|-----------|-----------|-------|
| GDP, CPI, PCE | Mensile/Trimestrale | FRED |
| PMI, ISM | Mensile | FRED |
| Unemployment, Claims | Settimanale/Mensile | FRED |
| Yield curve, Fed Funds | Giornaliero | FRED |
| LEI | Mensile | FRED |
| Market data (per backtest) | Giornaliero | FRED/Yahoo |

Scheduler: APScheduler, run giornaliero alle 06:00 UTC.

## Regole tecniche

- NO path assoluti nel codice (tutto relativo)
- NO secrets nel codice (solo .env)
- Coverage minimo 80%
- Usare type hints ovunque
- Docstring solo dove la logica non è auto-esplicativa

## Linee guida di sviluppo

### 1. Pensa prima di programmare

Non dare nulla per scontato. Non nascondere i dubbi. Metti in evidenza i compromessi.

Prima di implementare:

- Dichiara esplicitamente i presupposti. In caso di incertezza, chiedi.
- Se esistono diverse interpretazioni, presentale: non scegliere in silenzio.
- Se esiste un approccio più semplice, dillo. Esprimi la tua opinione quando necessario.
- Se qualcosa non è chiaro, fermati. Individua ciò che non ti è chiaro. Chiedi.

### 2. Semplicità prima di tutto

Codice minimo indispensabile per risolvere il problema. Niente di speculativo.

- Nessuna funzionalità oltre a quella richiesta.
- Nessuna astrazione per codice monouso.
- Nessuna "flessibilità" o "configurabilità" non richiesta.
- Nessuna gestione degli errori per scenari impossibili.
- Se scrivi 200 righe di codice e potresti riscriverle in 50, riscrivile.
- Chiediti: "Un ingegnere senior direbbe che questo è troppo complicato?" Se sì, semplifica.

### 3. Modifiche mirate

Modifica solo ciò che è necessario. Pulisci solo il tuo codice.

Quando modifichi codice esistente:

- Non "migliorare" codice, commenti o formattazione adiacenti.
- Non effettuare il refactoring di codice funzionante.
- Mantieni lo stile esistente, anche se lo faresti diversamente.
- Se noti del codice inutilizzato non correlato, segnalalo, non eliminarlo.

Quando le tue modifiche creano righe orfane:

- Rimuovi importazioni/variabili/funzioni che le TUE modifiche hanno reso inutilizzate.
- Non rimuovere codice inutilizzato preesistente a meno che non ti venga richiesto.

Il test: ogni riga modificata deve essere direttamente riconducibile alla richiesta dell'utente.

### 4. Esecuzione orientata agli obiettivi (TDD obbligatorio)

Definisci i criteri di successo. Ripeti il ciclo fino alla verifica.

Trasforma le attività in obiettivi verificabili:

- "Aggiungi la validazione" → "Scrivi test per input non validi, quindi falli superare"
- "Correggi il bug" → "Scrivi un test che lo riproduca, quindi fallo superare"
- "Esegui il refactoring di X" → "Assicurati che i test vengano superati prima e dopo"

Per le attività a più fasi, definisci un breve piano:

```text
1. [Fase] → verifica: [controllo]
2. [Fase] → verifica: [controllo]
3. [Fase] → verifica: [controllo]
```

Criteri di successo rigorosi consentono di eseguire cicli in modo indipendente. Criteri deboli ("fallo funzionare") richiedono chiarimenti costanti.

## Fasi

- **Fase 1** (corrente): Core regime classifier + scoring + dashboard
- **Fase 2**: Dedollarization layer, news scoring (Claude API), YouTube transcription DB
