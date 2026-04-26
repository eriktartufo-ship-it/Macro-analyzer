import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useDedollarBonus } from "../hooks/useDedollarBonus";
import { ScrollShadow } from "./ScrollShadow";
import type { DedollarComparison } from "../types";

function formatAsset(name: string): string {
  return name.replace(/_/g, " ");
}

function deltaColor(delta: number): string {
  if (Math.abs(delta) < 0.5) return "var(--muted)";
  return delta > 0 ? "var(--reflation)" : "var(--deflation)";
}

function bar(value: number, maxAbs: number, color: string) {
  const pct = Math.min(100, (Math.abs(value) / Math.max(maxAbs, 1)) * 100);
  return (
    <div style={{ position: "relative", height: 6, background: "var(--bg)", borderRadius: 3, marginTop: 3 }}>
      <div
        style={{
          position: "absolute",
          height: "100%",
          width: `${pct}%`,
          background: color,
          left: value < 0 ? `${50 - pct / 2}%` : "50%",
          right: value > 0 ? `${50 - pct / 2}%` : "auto",
          borderRadius: 3,
        }}
      />
      <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "var(--border)" }} />
    </div>
  );
}

export function DedollarComparisonPanel() {
  const [data, setData] = useState<DedollarComparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dedollarFlag] = useDedollarBonus();  // re-render quando toggle cambia

  useEffect(() => {
    let alive = true;
    api
      .dedollarComparison()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(e instanceof Error ? e.message : "Errore"));
    return () => {
      alive = false;
    };
  }, [dedollarFlag]);

  if (error)
    return (
      <div className="card">
        <h2>Asset scores — pure data-driven vs with dedollar bonus</h2>
        <div className="error" style={{ fontSize: 13 }}>{error}</div>
      </div>
    );
  if (!data)
    return (
      <div className="card">
        <h2>Asset scores — pure data-driven vs with dedollar bonus</h2>
        <div className="loading">Caricamento…</div>
      </div>
    );

  const maxAbsDelta = Math.max(0.01, ...data.items.map((it) => Math.abs(it.dedollar_delta)));
  // Stato effettivo applicato all'app: toggle UI ha priorita' su env server-side
  const userToggleActive = dedollarFlag;
  const flagColor = userToggleActive ? "var(--reflation)" : "var(--muted)";

  return (
    <div className="card">
      <h2>Asset scores — pure data-driven vs with dedollar bonus</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Il <strong>pure score</strong> e' calcolato solo da regime probabilities × performance hardcoded
        (data-driven puro). L'<strong>adjusted score</strong> aggiunge il bonus dedollar (sensitivity ×
        combined score). Il <strong>delta</strong> mostra esattamente quanto la dedollarizzazione sposta
        ogni asset.
      </div>

      <div style={{
        display: "flex", gap: 18, marginBottom: 14, padding: 10,
        background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 6, flexWrap: "wrap",
      }}>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Regime corrente</div>
          <div style={{ fontSize: 16, fontWeight: 600, textTransform: "capitalize" }}>{data.regime}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Dedollar combined</div>
          <div style={{ fontSize: 16, fontWeight: 600 }}>{data.dedollar_combined_score.toFixed(3)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Toggle UI corrente</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: flagColor }}>
            {userToggleActive ? "ADJUSTED (dedollar ON)" : "PURE (dedollar OFF)"}
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Come cambiare</div>
          <div style={{ fontSize: 11 }}>
            usa il toggle <code>$</code> nell'header — applicato istantaneamente a
            scoreboard, scenari, monte carlo e backtest.
            {data.use_dedollar_bonus_active && (
              <div style={{ marginTop: 4, color: "var(--muted)" }}>
                (env server <code>USE_DEDOLLAR_BONUS=1</code> attivo: usato come default
                quando il toggle non e' settato)
              </div>
            )}
          </div>
        </div>
      </div>

      <ScrollShadow innerClassName="table-wrap">
        <table className="table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>Asset</th>
              <th className="num">Sensitivity</th>
              <th className="num">Pure score</th>
              <th className="num">Adjusted</th>
              <th className="num">Δ dedollar</th>
              <th>Impatto</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((it) => {
              const color = deltaColor(it.dedollar_delta);
              return (
                <tr key={it.asset}>
                  <td><strong>{formatAsset(it.asset)}</strong></td>
                  <td className="num" style={{ color: it.sensitivity > 0 ? "var(--reflation)" : it.sensitivity < 0 ? "var(--deflation)" : "var(--muted)" }}>
                    {it.sensitivity > 0 ? "+" : ""}{it.sensitivity.toFixed(0)}
                  </td>
                  <td className="num" style={{ color: "var(--muted)" }}>{it.pure_score.toFixed(1)}</td>
                  <td className="num"><strong>{it.adjusted_score.toFixed(1)}</strong></td>
                  <td className="num" style={{ color, fontWeight: 600 }}>
                    {it.dedollar_delta > 0 ? "+" : ""}{it.dedollar_delta.toFixed(2)}
                  </td>
                  <td style={{ width: 140 }}>
                    {bar(it.dedollar_delta, maxAbsDelta, color)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </ScrollShadow>

      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 12 }}>
        <strong>Lettura.</strong> Sensitivity positiva = asset benefit della dedollarizzazione (gold,
        silver, BTC, EM). Negativa = penalizzato (USD, US bonds, cash). Δ dedollar = sensitivity ×
        combined_score. Quando il flag <code>USE_DEDOLLAR_BONUS</code> e' OFF (default), il sistema
        calcola gli asset score puramente dal regime + performance storiche, senza il bias dedollar.
        L'utente puo' vedere qui esattamente di quanto cambierebbero gli score se attivasse il bias.
      </div>
    </div>
  );
}
