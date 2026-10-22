import { useEffect, useState } from "react";
import type { MacroIndicatorsHistoryItem, RegimeExplain } from "../types";
import { api } from "../api/client";
import { MultiLineChart, type ChartPoint } from "./MultiLineChart";
import { regimeLabel } from "../lib/assets";

interface Props {
  explain: RegimeExplain;
}

const SIGNAL_COLORS: Record<string, string> = {
  strong: "var(--reflation)",
  moderate: "var(--goldilocks)",
  weak: "var(--deflation)",
  neutral: "var(--muted)",
};

const SIGNAL_LABELS: Record<string, string> = {
  strong: "Forte",
  moderate: "Media",
  weak: "Debole",
  neutral: "Spenta",
};

// Le condizioni arrivano ordinate per peso × intensità: le prime dicono il perché,
// le altre (fino a ~35) restano a richiesta.
const TOP_DRIVERS = 8;

const FORCE_TYPE_ICONS: Record<string, string> = {
  indicator: "DATI",
  news: "NOTIZIE",
  insider: "INSIDER",
  dedollarization: "DOLLARO",
};

interface IndicatorMeta {
  label: string;
  unit: string;
  type: "level" | "roc";
  explain: (v: number) => string;
}

const INDICATOR_META: Record<string, IndicatorMeta> = {
  gdp_roc: {
    label: "Crescita del PIL",
    unit: "%",
    type: "roc",
    explain: (v) =>
      v > 2 ? "Economia in espansione solida" :
      v > 0 ? "Crescita debole ma positiva" :
      "Economia in contrazione",
  },
  pmi: {
    label: "PMI manifatturiero",
    unit: "",
    type: "level",
    explain: (v) =>
      v > 55 ? "Manifatturiero in forte espansione" :
      v > 50 ? "Manifatturiero in leggera espansione" :
      v > 45 ? "Manifatturiero in leggera contrazione" :
      "Manifatturiero in forte contrazione",
  },
  cpi_yoy: {
    label: "Inflazione a 12 mesi (CPI)",
    unit: "%",
    type: "roc",
    explain: (v) =>
      v > 5 ? "Inflazione molto alta — pressione sui tassi" :
      v > 3 ? "Inflazione sopra il target Fed (2%)" :
      v > 2 ? "Inflazione vicina al target" :
      "Inflazione sotto il target — rischio deflazione",
  },
  unrate: {
    label: "Disoccupazione",
    unit: "%",
    type: "level",
    explain: (v) =>
      v < 4 ? "Mercato del lavoro molto forte" :
      v < 5 ? "Disoccupazione moderata" :
      v < 6 ? "Mercato del lavoro in deterioramento" :
      "Disoccupazione alta — stress economico",
  },
  unrate_roc: {
    label: "Disoccupazione (trend)",
    unit: "%",
    type: "roc",
    explain: (v) =>
      v < -1 ? "Disoccupazione in forte calo — economia in ripresa" :
      v < 0 ? "Disoccupazione in leggero miglioramento" :
      v < 0.5 ? "Disoccupazione stabile/in lieve aumento" :
      "Disoccupazione in rapido aumento — segnale recessivo",
  },
  yield_curve_10y2y: {
    label: "Curva dei tassi (10 anni − 2 anni)",
    unit: "%",
    type: "level",
    explain: (v) =>
      v > 1.5 ? "Curva ripida — aspettative di crescita forte" :
      v > 0.5 ? "Curva normale — nessun segnale d'allarme" :
      v > 0 ? "Curva piatta — rallentamento atteso" :
      "Curva invertita — forte segnale recessivo",
  },
  initial_claims_roc: {
    label: "Richieste sussidi (trend)",
    unit: "%",
    type: "roc",
    explain: (v) =>
      v < -10 ? "Sussidi in forte calo — mercato lavoro in ripresa" :
      v < 0 ? "Sussidi in leggero calo" :
      v < 10 ? "Sussidi stabili/in lieve aumento" :
      "Sussidi in forte aumento — licenziamenti in corso",
  },
  lei_roc: {
    label: "Indicatori anticipatori (LEI)",
    unit: "%",
    type: "roc",
    explain: (v) =>
      v > 5 ? "LEI in forte salita — crescita attesa nei prossimi 6-12 mesi" :
      v > 0 ? "LEI in miglioramento — prospettive positive" :
      v > -3 ? "LEI in calo — rallentamento atteso" :
      "LEI in forte calo — recessione probabile",
  },
  fed_funds_rate: {
    label: "Tasso della Fed",
    unit: "%",
    type: "level",
    explain: (v) =>
      v > 5 ? "Politica molto restrittiva — freno all'economia" :
      v > 3 ? "Politica restrittiva — condizioni finanziarie strette" :
      v > 1.5 ? "Politica neutrale" :
      "Politica accomodante — stimolo all'economia",
  },
  breakeven_10y: {
    label: "Inflazione attesa a 10 anni",
    unit: "%",
    type: "level",
    explain: (v) =>
      v > 2.8 ? "Mercato prezza inflazione elevata nel lungo periodo" :
      v > 2.3 ? "Aspettative inflazione sopra il target Fed" :
      v > 1.8 ? "Aspettative inflazione vicine al target 2%" :
      "Mercato prezza inflazione bassa — rischio deflazione",
  },
  vix: {
    label: "VIX (Volatilità)",
    unit: "",
    type: "level",
    explain: (v) =>
      v > 30 ? "Stress acuto — paura diffusa sui mercati" :
      v > 20 ? "Volatilità elevata — incertezza sopra la norma" :
      v > 15 ? "Volatilità normale" :
      "Volatilità compressa — calma e propensione al rischio",
  },
  nfci: {
    label: "Condizioni finanziarie (NFCI)",
    unit: "",
    type: "level",
    explain: (v) =>
      v > 0.5 ? "Condizioni finanziarie molto strette — credito razionato" :
      v > 0 ? "Condizioni finanziarie strette" :
      v > -0.5 ? "Condizioni finanziarie accomodanti" :
      "Condizioni finanziarie molto larghe — liquidità abbondante",
  },
};

function fmt(value: number, unit: string): string {
  return `${value.toFixed(2)}${unit}`;
}

function trendIcon(type: "level" | "roc", value: number): string {
  if (type === "roc") {
    if (value > 1) return "\u2191";
    if (value < -1) return "\u2193";
    return "\u2192";
  }
  return "";
}

export function AnalysisPanel({ explain }: Props) {
  const hasIndicators = Object.keys(explain.indicators).length > 0;
  const traj = explain.trajectory;
  const hasForces = !!traj && traj.forces.length > 0;

  const [history, setHistory] = useState<MacroIndicatorsHistoryItem[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [allDrivers, setAllDrivers] = useState(false);
  const drivers = allDrivers ? explain.top_drivers : explain.top_drivers.slice(0, TOP_DRIVERS);

  useEffect(() => {
    let alive = true;
    api
      .macroIndicatorsHistory(365)
      .then((rows) => {
        if (alive) setHistory(rows);
      })
      .catch(() => {
        if (alive) setHistory([]);
      });
    return () => {
      alive = false;
    };
  }, []);

  const toggle = (key: string) => setExpanded((cur) => (cur === key ? null : key));

  return (
    <div className="card">
      <h2>Perché questo regime: le condizioni</h2>

      {/* Driving forces */}
      {hasForces && (
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
            COSA LO STA SPINGENDO
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {traj!.forces.map((f, i) => {
              const forceChipClass =
                f.type === "news"
                  ? "chip chip-info"
                  : f.type === "dedollarization"
                  ? "chip"
                  : "chip chip-warn";
              const forceChipStyle =
                f.type === "dedollarization"
                  ? { background: "rgba(159, 78, 237, 0.15)", color: "var(--stagflation)" }
                  : undefined;
              return (
                <div key={i} className="card-row">
                  <span
                    className={forceChipClass}
                    style={{ fontSize: 11, padding: "4px 8px", fontWeight: 700, ...forceChipStyle }}
                    data-nowrap
                  >
                    {FORCE_TYPE_ICONS[f.type] ?? f.type}
                  </span>
                  <span style={{ fontSize: 14 }}>
                    {f.description}
                  </span>
                  <span data-nowrap style={{ fontSize: 13, color: "var(--muted)" }}>
                    {"\u2192"}
                  </span>
                  <span
                    data-nowrap
                    className={`regime-${f.pushes_toward}`}
                    style={{ fontSize: 13, fontWeight: 700 }}
                  >
                    {regimeLabel(f.pushes_toward)}
                  </span>
                  <span
                    data-nowrap
                    className="card-row-value"
                    style={{
                      color: f.strength > 0 ? "var(--reflation)" : "var(--deflation)",
                    }}
                  >
                    {f.strength > 0 ? "+" : ""}
                    {(f.strength * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Top drivers */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
          LE CONDIZIONI CHE PESANO DI PIÙ
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {drivers.map((d) => (
            <div
              key={d.name}
              className="card-row"
              style={{
                borderLeft: `4px solid ${SIGNAL_COLORS[d.signal]}`,
              }}
            >
              <span
                data-nowrap
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  color: SIGNAL_COLORS[d.signal],
                  textTransform: "uppercase",
                  minWidth: 70,
                }}
              >
                {SIGNAL_LABELS[d.signal]}
              </span>
              <span style={{ fontSize: 14 }}>{d.description}</span>
              <span data-nowrap className="card-row-value" style={{ color: "var(--muted)" }}>
                {(d.raw_score * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
        {explain.top_drivers.length > TOP_DRIVERS && (
          <button className="btn-more" onClick={() => setAllDrivers((v) => !v)}>
            {allDrivers ? "Mostra solo le più importanti" : `Mostra tutte le ${explain.top_drivers.length} condizioni`}
          </button>
        )}
      </div>

      {/* Raw indicators with explanations */}
      {hasIndicators && (
        <div>
          <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
            INDICATORI MACRO
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {Object.entries(explain.indicators).map(([key, value]) => {
              const meta = INDICATOR_META[key];
              if (!meta) return null;
              const arrow = trendIcon(meta.type, value);
              const isExpanded = expanded === key;
              const points: ChartPoint[] = history
                .filter((h) => typeof h.indicators[key] === "number")
                .map((h) => ({ date: h.date, values: { [key]: h.indicators[key] } }));
              const canChart = points.length > 1;

              return (
                <div
                  key={key}
                  onClick={() => canChart && toggle(key)}
                  style={{
                    padding: "8px 12px",
                    background: "var(--bg)",
                    borderRadius: 6,
                    cursor: canChart ? "pointer" : "default",
                  }}
                  role={canChart ? "button" : undefined}
                  aria-expanded={canChart ? isExpanded : undefined}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 13, fontWeight: 500 }}>
                      {canChart && (
                        <span style={{ marginRight: 6, fontSize: 11, color: "var(--muted)" }}>
                          {isExpanded ? "▼" : "▶"}
                        </span>
                      )}
                      {meta.label}
                      {meta.type === "roc" && (
                        <span style={{ fontSize: 10, color: "var(--muted)", marginLeft: 6 }}>
                          TREND
                        </span>
                      )}
                    </span>
                    <span style={{ fontWeight: 700, fontVariantNumeric: "tabular-nums", fontSize: 14 }}>
                      {arrow} {fmt(value, meta.unit)}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 3 }}>
                    {meta.explain(value)}
                  </div>
                  {isExpanded && canChart && (
                    <div onClick={(e) => e.stopPropagation()} style={{ marginTop: 6 }}>
                      <MultiLineChart
                        title={`${meta.label} — ultimi 12 mesi`}
                        points={points}
                        series={[{ key, label: meta.label, color: "var(--accent)" }]}
                        yFormat={(v) => `${v.toFixed(2)}${meta.unit}`}
                        height={180}
                        compact
                        showLegend={false}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
