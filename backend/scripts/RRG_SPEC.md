# RRG — Relative Rotation Graph (spec, spec-driven prima del codice)

> Scaletta S43 #2. Il segnale flussi è REALE (diagnostico +2,72 t4,1) ma NON batte l'S&P
> long-only (`FLOW_TRADER_CRITERIA.md`, mensile+daily). Quindi il suo valore è **INFORMATIVO**:
> una vista che mostra DOVE stanno andando i soldi tra settori/asset-class. L'RRG è lo standard
> (Julius de Kempenaer / StockCharts) per questo: ogni asset in uno dei 4 quadranti, con la coda.

## Definizioni (convenzione JdK, centro = 100)

- **Benchmark** = **basket equal-weight dell'universo** (NON un singolo indice): l'RRG cross-asset
  mette equity/bond/oro/cripto insieme → un benchmark equity non ha senso per i bond. Il basket =
  "il mercato dei flussi"; RS relativo alla media dice se i soldi vanno VERSO o VIA da quell'asset.
  Dichiarato. (StockCharts usa un indice esterno; qui il basket è la scelta corretta cross-asset.)
- **RS** (relative strength) = livello_asset / livello_benchmark (livelli = NAV cumulati dai ret).
- **RS-Ratio** = 100 + z-score(RS) su finestra W → forza relativa normalizzata. >100 = più forte
  del basket. (Il valore assoluto della scala non conta per i quadranti — conta il segno vs 100.)
- **RS-Momentum** = 100 + z-score(ROC dell'RS-Ratio) su finestra W → sta MIGLIORANDO o PEGGIORANDO
  la forza relativa. >100 = accelera.
- **4 quadranti** (dal centro 100,100):
  | quadrante | ratio | momentum | significato |
  |---|---|---|---|
  | **Leading** | ≥100 | ≥100 | forte e accelera (i soldi ci sono e continuano) |
  | **Weakening** | ≥100 | <100 | forte ma rallenta (inizia a uscire) |
  | **Lagging** | <100 | <100 | debole e peggiora (i soldi sono usciti) |
  | **Improving** | <100 | ≥100 | debole ma accelera (i soldi iniziano a entrare) |
  Rotazione tipica (senso orario): Improving → Leading → Weakening → Lagging → Improving.
- **Tail** = ultimi N punti (ratio, momentum) per asset = la traiettoria recente (dove sta andando).

## Cadenza & dati

- **Settimanale** (RRG classico): ricampiono i livelli mensili/daily a fine settimana. Uso la
  matrice DAILY (già in cache) → livelli → resample settimanale. Universo = i 17 del diagnostico.
- **W** (finestra z-score) = 52 settimane (1 anno) di default. **N tail** = 8 settimane.
- **NO FUTURE LEAK**: z-score e ROC usano solo dati ≤ t (rolling trailing). L'as_of = ultima
  settimana disponibile.

## Endpoint

`GET /api/v1/flows/rrg?window_weeks=52&tail=8`
```json
{
  "as_of": "2026-05-29",
  "benchmark": "equal_weight_universe",
  "window_weeks": 52, "tail": 8,
  "points": [
    {"asset": "sector_technology", "label": "Tech", "quadrant": "leading",
     "ratio": 102.4, "momentum": 101.1,
     "tail": [{"date": "...", "ratio": 99.8, "momentum": 100.3}, ...]}  // N punti, vecchio→nuovo
  ]
}
```

## Frontend `FlowsRrgPanel.tsx`

- SVG scatter (pattern di `MonteCarloPanel/ConeChart`): assi RS-Ratio (x) × RS-Momentum (y),
  origine centrata a 100, 4 quadranti colorati soft (verde/giallo/rosso/blu = leading/weakening/
  lagging/improving), croce centrale a 100.
- Ogni asset = pallino nel quadrante + polyline della tail (opacità crescente verso il presente) +
  label. Tooltip su hover (asset, quadrante, ratio, momentum).
- Liquid Glass, dark/light via token, mobile-first (SVG viewBox responsive, `max-width:100%`).
- Legenda 4 quadranti. Registrato in `DataPage.tsx`.

## Cosa NON è (limiti)

- NON è un segnale di trading (provato: non batte l'S&P). È una LETTURA dei flussi per decidere.
- Benchmark = basket interno → un asset "Leading" è forte VS gli altri 16, non in assoluto.
- z-score normalizza → in un mercato dove tutto sale insieme, i quadranti restano informativi in
  RELATIVO (chi guida), non dicono "risk-on/off" assoluto (quello è la scaletta #3, cross-asset).
