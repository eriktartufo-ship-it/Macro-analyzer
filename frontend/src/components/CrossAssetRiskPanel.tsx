import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { RiskClass, RiskOnOff } from "../types";

const RISK_COL = "#10b981";   // risk-on / positivo
const OFF_COL = "#ef4444";    // risk-off / negativo
const LOOKBACKS = [4, 13, 26];
const GAUGE_CLAMP = 20;       // pp: estremi del termometro

const REGIME_LABEL: Record<string, string> = {
  risk_on: "RISK-ON", neutral: "NEUTRALE", risk_off: "RISK-OFF",
};
const REGIME_COL: Record<string, string> = {
  risk_on: RISK_COL, neutral: "#f59e0b", risk_off: OFF_COL,
};

function Gauge({ gauge, regime }: { gauge: number; regime: string }) {
  const pos = Math.min(1, Math.max(0, (gauge + GAUGE_CLAMP) / (2 * GAUGE_CLAMP)));
  return (
    <div style={{ margin: "8px 0 18px" }}>
      <div style={{ position: "relative", height: 14, borderRadius: 999,
        background: `linear-gradient(90deg, ${OFF_COL} 0%, #f59e0b 50%, ${RISK_COL} 100%)`,
        opacity: 0.85 }}>
        <div style={{ position: "absolute", left: `${pos * 100}%`, top: -4, transform: "translateX(-50%)",
          width: 4, height: 22, borderRadius: 3, background: "var(--text)",
          boxShadow: "0 0 0 2px var(--bg)" }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11,
        color: "var(--muted)", marginTop: 6 }}>
        <span>← Risk-off (rifugi)</span>
        <strong style={{ color: REGIME_COL[regime], fontSize: 13 }}>
          {REGIME_LABEL[regime]} · {gauge >= 0 ? "+" : ""}{gauge.toFixed(1)}pp
        </strong>
        <span>Risk-on (rischio) →</span>
      </div>
    </div>
  );
}

function ClassBar({ c, maxAbs, isRefuge }: { c: RiskClass; maxAbs: number; isRefuge: boolean }) {
  const m = c.momentum_pct;
  const na = m === null;
  const pct = na ? 0 : Math.min(1, Math.abs(m) / maxAbs);
  const pos = m !== null && m >= 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "5px 0" }}>
      <div style={{ width: 96, flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontWeight: 600, color: "var(--text)", fontSize: 13 }}>{c.label}</span>
        {isRefuge && <span style={{ fontSize: 9.5, fontWeight: 700, color: "#fff",
          background: RISK_COL, borderRadius: 4, padding: "1px 5px" }}>RIFUGIO</span>}
      </div>
      <div style={{ flex: 1, display: "flex", alignItems: "center", height: 20 }}>
        {/* barra divergente dal centro */}
        <div style={{ flex: 1, display: "flex", justifyContent: "flex-end" }}>
          {!na && !pos && (
            <div style={{ width: `${pct * 100}%`, height: 12, background: OFF_COL,
              borderRadius: "3px 0 0 3px", opacity: 0.85 }} />
          )}
        </div>
        <div style={{ width: 1, height: 16, background: "var(--border)" }} />
        <div style={{ flex: 1, display: "flex", justifyContent: "flex-start" }}>
          {!na && pos && (
            <div style={{ width: `${pct * 100}%`, height: 12, background: RISK_COL,
              borderRadius: "0 3px 3px 0", opacity: 0.85 }} />
          )}
        </div>
      </div>
      <div style={{ width: 76, flexShrink: 0, textAlign: "right", fontSize: 12.5,
        fontVariantNumeric: "tabular-nums",
        color: na ? "var(--muted)" : pos ? RISK_COL : OFF_COL, fontWeight: 600 }}>
        {na ? `~${c.weeks_available}w` : `${m >= 0 ? "+" : ""}${m.toFixed(1)}%`}
      </div>
    </div>
  );
}

export function CrossAssetRiskPanel() {
  const [data, setData] = useState<RiskOnOff | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [lookback, setLookback] = useState(13);

  useEffect(() => {
    let alive = true;
    setLoading(true); setErr(null);
    api.flowsRiskOnOff(lookback)
      .then((r) => { if (alive) { setData(r); setLoading(false); } })
      .catch((e) => { if (alive) { setErr(String(e)); setLoading(false); } });
    return () => { alive = false; };
  }, [lookback]);

  const maxAbs = useMemo(() => Math.max(1e-6,
    ...(data?.classes ?? []).map((c) => Math.abs(c.momentum_pct ?? 0))), [data]);

  return (
    <div className="card">
      <h2>Flussi — Risk-on / Risk-off cross-asset</h2>
      <p style={{ color: "var(--muted)", fontSize: 13, marginTop: -4, maxWidth: 660 }}>
        Il mercato è in modalità <strong style={{ color: RISK_COL }}>rischio</strong> (azioni/cripto
        guidano) o <strong style={{ color: OFF_COL }}>rifugio</strong> (oro/bond/cash)? E quale rifugio
        sta reggendo ora. Momentum reale sul periodo — fotografia del posizionamento, non previsione.
      </p>

      <div style={{ display: "inline-flex", gap: 2, background: "var(--surface-2, var(--bg))",
        border: "1px solid var(--border)", borderRadius: 999, padding: 2, marginBottom: 14 }}>
        {LOOKBACKS.map((w) => (
          <button key={w} onClick={() => setLookback(w)}
            style={{ border: "none", cursor: "pointer", borderRadius: 999, padding: "5px 14px",
              fontSize: 12.5, fontWeight: 600,
              background: lookback === w ? RISK_COL : "transparent",
              color: lookback === w ? "#fff" : "var(--muted)" }}>
            {w}w
          </button>
        ))}
      </div>

      {loading && <p style={{ color: "var(--muted)" }}>Calcolo il posizionamento…</p>}
      {err && <p style={{ color: OFF_COL }}>Errore: {err}</p>}

      {data && !loading && (
        <>
          <p style={{ color: "var(--muted)", fontSize: 12, margin: "0 0 4px" }}>
            Aggiornato al {data.as_of} · momentum a {data.lookback_weeks} settimane
            {data.best_refuge_label && <> · rifugio migliore: <strong style={{ color: "var(--text)" }}>
              {data.best_refuge_label}</strong></>}
          </p>
          <Gauge gauge={data.gauge} regime={data.regime} />
          <div>
            {data.classes.map((c) => (
              <ClassBar key={c.asset} c={c} maxAbs={maxAbs} isRefuge={c.asset === data.best_refuge} />
            ))}
          </div>
          <p style={{ color: "var(--muted)", fontSize: 11, marginTop: 10 }}>
            Cripto ha storia breve: se le settimane disponibili sono meno del periodo, resta fuori dal
            gauge (mostrata come <em>~Nw</em>).
          </p>
        </>
      )}
    </div>
  );
}
