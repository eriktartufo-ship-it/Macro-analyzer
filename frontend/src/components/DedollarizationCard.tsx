import type { Dedollarization } from "../types";

interface Props {
  data: Dedollarization;
  rawIndicators?: Record<string, number>;
}

interface ComponentMeta {
  label: string;
  explain: (score: number, raw?: number) => string;
  rawKey?: string;
  rawUnit?: string;
}

const COMPONENT_META: Record<string, ComponentMeta> = {
  usd_weakness: {
    label: "USD Weakness (12m)",
    rawKey: "dxy_roc_12m",
    rawUnit: "%",
    explain: (_score, raw) => {
      if (raw === undefined) return "";
      if (raw < -8) return `USD ${raw.toFixed(1)}% in 12 mesi — forte svalutazione`;
      if (raw < -3) return `USD ${raw.toFixed(1)}% in 12 mesi — dollaro in indebolimento`;
      if (raw < 3) return `USD ${raw > 0 ? "+" : ""}${raw.toFixed(1)}% — sostanzialmente stabile`;
      return `USD +${raw.toFixed(1)}% in 12 mesi — dollaro in rafforzamento`;
    },
  },
  gold_strength: {
    label: "Gold Strength (12m)",
    rawKey: "gold_roc_12m",
    rawUnit: "%",
    explain: (_score, raw) => {
      if (raw === undefined) return "";
      if (raw > 20) return `Oro +${raw.toFixed(0)}% in 12 mesi — forte domanda di safe haven`;
      if (raw > 8) return `Oro +${raw.toFixed(0)}% — domanda hedge in crescita`;
      if (raw > 0) return `Oro +${raw.toFixed(1)}% — crescita moderata`;
      return `Oro ${raw.toFixed(1)}% — nessuna domanda di hedge`;
    },
  },
  gold_oil_ratio: {
    label: "Gold / Oil Ratio",
    rawKey: "gold_oil_ratio",
    rawUnit: "x",
    explain: (_score, raw) => {
      if (raw === undefined) return "";
      if (raw > 35) return `Ratio ${raw.toFixed(0)}x — oro molto forte vs energia reale`;
      if (raw > 25) return `Ratio ${raw.toFixed(0)}x — gold premium moderato`;
      return `Ratio ${raw.toFixed(0)}x — equilibrio tra oro e petrolio`;
    },
  },
  debt_burden: {
    label: "Debt Burden",
    rawKey: "debt_gdp",
    rawUnit: "%",
    explain: (_score, raw) => {
      if (raw === undefined) return "";
      if (raw > 120) return `Debito/PIL ${raw.toFixed(0)}% — livello insostenibile`;
      if (raw > 100) return `Debito/PIL ${raw.toFixed(0)}% — sopra soglia critica`;
      if (raw > 80) return `Debito/PIL ${raw.toFixed(0)}% — elevato ma gestibile`;
      return `Debito/PIL ${raw.toFixed(0)}% — fiscalmente solido`;
    },
  },
  real_rate_signal: {
    label: "Real Rate Signal",
    rawKey: "real_rate",
    rawUnit: "%",
    explain: (_score, raw) => {
      if (raw === undefined) return "";
      if (raw < -2) return `Tasso reale ${raw.toFixed(1)}% — forte debasement valutario`;
      if (raw < 0) return `Tasso reale ${raw.toFixed(2)}% — inflazione erode il dollaro`;
      if (raw < 1.5) return `Tasso reale +${raw.toFixed(2)}% — leggermente positivo`;
      return `Tasso reale +${raw.toFixed(1)}% — dollaro supportato`;
    },
  },
  monetary_debasement: {
    label: "Monetary Debasement (12m)",
    rawKey: "m2_roc_12m",
    rawUnit: "%",
    explain: (_score, raw) => {
      if (raw === undefined) return "";
      if (raw > 12) return `M2 +${raw.toFixed(0)}% — espansione aggressiva`;
      if (raw > 7) return `M2 +${raw.toFixed(1)}% — crescita sopra la norma`;
      if (raw > 3) return `M2 +${raw.toFixed(1)}% — crescita moderata`;
      return `M2 +${raw.toFixed(1)}% — supply restrittiva`;
    },
  },
};

const STRUCTURAL_META: Record<string, { label: string; explain: (v: number) => string }> = {
  usd_secular: {
    label: "USD Decline (5Y trend)",
    explain: (v) =>
      v > 0.7 ? "Trend secolare di indebolimento USD confermato" :
      v > 0.5 ? "Leggero declino strutturale in corso" :
      "USD strutturalmente stabile/forte negli ultimi 5 anni",
  },
  gold_secular: {
    label: "Gold Rise (5Y trend)",
    explain: (v) =>
      v > 0.7 ? "Oro in forte trend rialzista pluriennale" :
      v > 0.5 ? "Oro in crescita moderata su 5 anni" :
      "Oro stagnante su base pluriennale",
  },
  debt_trajectory: {
    label: "Debt Trajectory (5Y)",
    explain: (v) =>
      v > 0.7 ? "Debito in rapida espansione — +20pp in 5 anni" :
      v > 0.5 ? "Debito in crescita moderata" :
      "Debito stabile o in discesa",
  },
  m2_cumulative: {
    label: "M2 Cumulative (5Y)",
    explain: (v) =>
      v > 0.7 ? "Espansione monetaria cumulata molto forte su 5 anni" :
      v > 0.5 ? "Crescita M2 sopra media su 5 anni" :
      "Crescita M2 contenuta su 5 anni",
  },
};

function scoreColor(score: number): string {
  if (score >= 0.7) return "var(--deflation)";
  if (score >= 0.5) return "var(--goldilocks)";
  return "var(--reflation)";
}

function scoreLabel(score: number): string {
  if (score >= 0.75) return "Extreme";
  if (score >= 0.6) return "High";
  if (score >= 0.4) return "Moderate";
  if (score >= 0.25) return "Low";
  return "Minimal";
}

function accelLabel(accel: number): { text: string; color: string } {
  if (accel > 0.3) return { text: "Accelerating fast", color: "var(--deflation)" };
  if (accel > 0.1) return { text: "Accelerating", color: "var(--goldilocks)" };
  if (accel > -0.1) return { text: "Stable pace", color: "var(--muted)" };
  if (accel > -0.3) return { text: "Decelerating", color: "var(--reflation)" };
  return { text: "Reversing", color: "var(--reflation)" };
}

const DECADE_META: Record<string, { label: string; explain: (v: number) => string }> = {
  usd_decade: {
    label: "USD Decline (10Y trend)",
    explain: (v) =>
      v > 0.7 ? "Dollaro in declino strutturale confermato su un decennio" :
      v > 0.5 ? "Tendenza decennale di indebolimento" :
      v > 0.3 ? "Dollaro sostanzialmente stabile su 10 anni" :
      "Dollaro strutturalmente forte nell'ultimo decennio",
  },
  gold_decade: {
    label: "Gold Rise (10Y trend)",
    explain: (v) =>
      v > 0.7 ? "Oro in super-ciclo rialzista decennale" :
      v > 0.5 ? "Oro in crescita strutturale su 10 anni" :
      v > 0.3 ? "Oro con crescita modesta su 10 anni" :
      "Oro stagnante/in calo nell'ultimo decennio",
  },
  debt_decade: {
    label: "Debt Growth (10Y)",
    explain: (v) =>
      v > 0.7 ? "Esplosione del debito: +30pp o più in un decennio" :
      v > 0.5 ? "Debito in forte crescita su 10 anni" :
      v > 0.3 ? "Crescita del debito moderata" :
      "Debito stabile o in discesa su 10 anni",
  },
  m2_decade: {
    label: "M2 Expansion (10Y)",
    explain: (v) =>
      v > 0.7 ? "Espansione monetaria strutturale molto forte" :
      v > 0.5 ? "M2 in crescita sopra media su un decennio" :
      v > 0.3 ? "Crescita M2 moderata" :
      "Supply monetaria contenuta su 10 anni",
  },
};

export function DedollarizationCard({ data, rawIndicators }: Props) {
  const color = scoreColor(data.combined_score);
  const accel = accelLabel(data.acceleration);
  const sortedComponents = Object.entries(data.components).sort((a, b) => b[1] - a[1]);
  const sortedStructural = Object.entries(data.structural).sort((a, b) => b[1] - a[1]);
  const sortedDecade = Object.entries(data.decade).sort((a, b) => b[1] - a[1]);
  const hasStructural = sortedStructural.length > 0;
  const hasDecade = sortedDecade.length > 0;

  return (
    <div className="card">
      <h2>Dedollarization Index</h2>

      {/* Header scores */}
      <div className="grid grid-4" style={{ gap: 12, marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", marginBottom: 4 }}>
            Cyclical (1Y)
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: scoreColor(data.score) }}>
            {(data.score * 100).toFixed(0)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", marginBottom: 4 }}>
            Structural (5Y)
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: scoreColor(data.structural_score) }}>
            {(data.structural_score * 100).toFixed(0)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", marginBottom: 4 }}>
            Decade (10Y)
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: scoreColor(data.decade_score) }}>
            {(data.decade_score * 100).toFixed(0)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", marginBottom: 4 }}>
            Acceleration
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: accel.color }}>
            {data.acceleration > 0 ? "+" : ""}{(data.acceleration * 100).toFixed(0)}%
          </div>
          <div style={{ fontSize: 11, color: accel.color }}>{accel.text}</div>
        </div>
      </div>

      {/* Combined score */}
      <div style={{ marginBottom: 20, padding: "8px 12px", background: "var(--bg)", borderRadius: 6 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: 13, fontWeight: 500 }}>Combined Score</span>
          <span style={{ fontSize: 18, fontWeight: 700, color }}>
            {(data.combined_score * 100).toFixed(0)}% — {scoreLabel(data.combined_score)}
          </span>
        </div>
        <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
          Media pesata: ciclico 25% + strutturale 5Y 30% + decennale 10Y 30% + accelerazione 15%
        </div>
      </div>

      {/* Structural components (5Y) */}
      {hasStructural && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
            TREND STRUTTURALE (5 ANNI)
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {sortedStructural.map(([key, value]) => {
              const meta = STRUCTURAL_META[key];
              if (!meta) return null;
              return (
                <div key={key} style={{ padding: "6px 10px", background: "var(--bg)", borderRadius: 6 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                    <span style={{ fontSize: 13 }}>{meta.label}</span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: scoreColor(value) }}>
                      {(value * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="prob-bar" style={{ height: 5, marginBottom: 3 }}>
                    <div className="prob-fill" style={{ width: `${(value * 100).toFixed(0)}%`, background: scoreColor(value) }} />
                  </div>
                  <div style={{ fontSize: 12, color: "var(--muted)" }}>{meta.explain(value)}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Decade components (10Y) */}
      {hasDecade && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
            VISIONE DECENNALE (10 ANNI)
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {sortedDecade.map(([key, value]) => {
              const meta = DECADE_META[key];
              if (!meta) return null;
              return (
                <div key={key} style={{ padding: "6px 10px", background: "var(--bg)", borderRadius: 6 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                    <span style={{ fontSize: 13 }}>{meta.label}</span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: scoreColor(value) }}>
                      {(value * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="prob-bar" style={{ height: 5, marginBottom: 3 }}>
                    <div className="prob-fill" style={{ width: `${(value * 100).toFixed(0)}%`, background: scoreColor(value) }} />
                  </div>
                  <div style={{ fontSize: 12, color: "var(--muted)" }}>{meta.explain(value)}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Cyclical components (12m) */}
      <div>
        <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
          SEGNALI CICLICI (12 MESI)
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {sortedComponents.map(([key, value]) => {
            const meta = COMPONENT_META[key];
            if (!meta) return null;
            const rawValue = rawIndicators?.[meta.rawKey ?? ""];
            return (
              <div key={key} style={{ padding: "6px 10px", background: "var(--bg)", borderRadius: 6 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 3 }}>
                  <span style={{ fontSize: 13, flex: 1 }}>{meta.label}</span>
                  {rawValue !== undefined && (
                    <span style={{ fontSize: 12, color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
                      {rawValue > 0 ? "+" : ""}{rawValue.toFixed(1)}{meta.rawUnit}
                    </span>
                  )}
                  <span style={{ fontSize: 13, fontWeight: 600, color: scoreColor(value), width: 40, textAlign: "right" }}>
                    {(value * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="prob-bar" style={{ height: 5, marginBottom: 3 }}>
                  <div className="prob-fill" style={{ width: `${(value * 100).toFixed(0)}%`, background: scoreColor(value) }} />
                </div>
                <div style={{ fontSize: 12, color: "var(--muted)" }}>{meta.explain(value, rawValue)}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
