import type { Reliability } from "../types";

/**
 * Badge di affidabilità del segnale di flusso (S43 #4): il momentum settoriale è robusto in
 * mercato calmo e una trappola in stress (misurato: calmo t+3.4, stress crolla). Dice all'utente
 * quando fidarsi delle viste flussi.
 */
export function ReliabilityBadge({ r }: { r?: Reliability }) {
  if (!r || r.reliable === null) return null;
  const calm = r.reliable === true;
  const col = calm ? "#10b981" : "#f59e0b";
  const dist = r.dist_pct === null ? "" :
    `S&P ${r.dist_pct >= 0 ? "+" : ""}${r.dist_pct.toFixed(1)}% vs media ${r.sma_days}gg`;
  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: 8, margin: "0 0 12px",
      padding: "8px 12px", borderRadius: 8,
      background: `color-mix(in srgb, ${col} 12%, transparent)`,
      border: `1px solid color-mix(in srgb, ${col} 40%, transparent)`,
    }}>
      <span style={{ fontSize: 15, lineHeight: 1.2 }}>{calm ? "✓" : "⚠"}</span>
      <div style={{ fontSize: 12.5, lineHeight: 1.45 }}>
        <strong style={{ color: col }}>
          {calm ? "Segnale affidabile — mercato in trend" : "Cautela — mercato in stress"}
        </strong>
        <span style={{ color: "var(--muted)" }}>
          {" "}({dist}).{" "}
          {calm
            ? "Il momentum dei flussi tende a persistere."
            : "In stress il momentum settoriale è spesso una trappola: leggi i flussi con prudenza."}
        </span>
      </div>
    </div>
  );
}
