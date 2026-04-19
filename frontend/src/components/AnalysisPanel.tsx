import type { RegimeExplain } from "../types";

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
  strong: "Strong",
  moderate: "Moderate",
  weak: "Weak",
  neutral: "Neutral",
};

const FORCE_TYPE_ICONS: Record<string, string> = {
  indicator: "MACRO",
  news: "NEWS",
  dedollarization: "DEDOL",
};

interface IndicatorMeta {
  label: string;
  unit: string;
  type: "level" | "roc";
  explain: (v: number) => string;
}

const INDICATOR_META: Record<string, IndicatorMeta> = {
  gdp_roc: {
    label: "GDP Growth",
    unit: "%",
    type: "roc",
    explain: (v) =>
      v > 2 ? "Economia in espansione solida" :
      v > 0 ? "Crescita debole ma positiva" :
      "Economia in contrazione",
  },
  pmi: {
    label: "PMI Manufacturing",
    unit: "",
    type: "level",
    explain: (v) =>
      v > 55 ? "Manifatturiero in forte espansione" :
      v > 50 ? "Manifatturiero in leggera espansione" :
      v > 45 ? "Manifatturiero in leggera contrazione" :
      "Manifatturiero in forte contrazione",
  },
  cpi_yoy: {
    label: "Inflazione (CPI YoY)",
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
    label: "Yield Curve (10Y-2Y)",
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
      v > 0 ? "LEI in miglioramento — outlook positivo" :
      v > -3 ? "LEI in calo — rallentamento atteso" :
      "LEI in forte calo — recessione probabile",
  },
  fed_funds_rate: {
    label: "Fed Funds Rate",
    unit: "%",
    type: "level",
    explain: (v) =>
      v > 5 ? "Politica molto restrittiva — freno all'economia" :
      v > 3 ? "Politica restrittiva — condizioni finanziarie strette" :
      v > 1.5 ? "Politica neutrale" :
      "Politica accomodante — stimolo all'economia",
  },
  breakeven_10y: {
    label: "10Y Breakeven Inflation",
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
      "Volatilità compressa — calma e risk-on",
  },
  nfci: {
    label: "Chicago Fed NFCI",
    unit: "",
    type: "level",
    explain: (v) =>
      v > 0.5 ? "Condizioni finanziarie molto strette — credit crunch" :
      v > 0 ? "Condizioni finanziarie strette" :
      v > -0.5 ? "Condizioni finanziarie accomodanti" :
      "Condizioni finanziarie molto loose — liquidità abbondante",
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

  return (
    <div className="card">
      <h2>Analysis — Why {explain.regime}?</h2>

      {/* Driving forces */}
      {hasForces && (
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
            DRIVING FORCES
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
                <div key={i} className="row-item" style={{ padding: "8px 12px" }}>
                  <span
                    className={forceChipClass}
                    style={{ fontSize: 10, padding: "2px 8px", ...forceChipStyle }}
                    data-nowrap
                  >
                    {FORCE_TYPE_ICONS[f.type] ?? f.type}
                  </span>
                  <span className="row-text" style={{ fontSize: 13 }}>
                    {f.description}
                  </span>
                  <span data-nowrap style={{ fontSize: 12, color: "var(--muted)" }}>
                    {"\u2192"}
                  </span>
                  <span
                    data-nowrap
                    className={`regime-${f.pushes_toward}`}
                    style={{ fontSize: 12, fontWeight: 600, textTransform: "capitalize" }}
                  >
                    {f.pushes_toward}
                  </span>
                  <span
                    data-nowrap
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      fontVariantNumeric: "tabular-nums",
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
          KEY DRIVERS
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {explain.top_drivers.map((d) => (
            <div
              key={d.name}
              className="row-item"
              style={{
                padding: "8px 12px",
                borderLeft: `3px solid ${SIGNAL_COLORS[d.signal]}`,
              }}
            >
              <span
                data-nowrap
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: SIGNAL_COLORS[d.signal],
                  textTransform: "uppercase",
                  minWidth: 60,
                }}
              >
                {SIGNAL_LABELS[d.signal]}
              </span>
              <span className="row-text" style={{ fontSize: 13 }}>{d.description}</span>
              <span data-nowrap style={{ fontSize: 12, color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
                {(d.raw_score * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Raw indicators with explanations */}
      {hasIndicators && (
        <div>
          <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
            MACRO INDICATORS
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {Object.entries(explain.indicators).map(([key, value]) => {
              const meta = INDICATOR_META[key];
              if (!meta) return null;
              const arrow = trendIcon(meta.type, value);
              return (
                <div
                  key={key}
                  style={{
                    padding: "8px 12px",
                    background: "var(--bg)",
                    borderRadius: 6,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 13, fontWeight: 500 }}>
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
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
