import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useDedollarBonus } from "../hooks/useDedollarBonus";
import { ScrollShadow } from "./ScrollShadow";
import type { DedollarComparison } from "../types";
import { assetName, fmt, fmtSigned, regimeLabel } from "../lib/assets";

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
        <h2>Score degli asset senza e con il bonus dollaro</h2>
        <div className="error" style={{ fontSize: 13 }}>{error}</div>
      </div>
    );
  if (!data)
    return (
      <div className="card">
        <h2>Score degli asset senza e con il bonus dollaro</h2>
        <div className="loading">Caricamento…</div>
      </div>
    );

  const maxAbsDelta = Math.max(0.01, ...data.items.map((it) => Math.abs(it.dedollar_delta)));
  // Stato effettivo applicato all'app: toggle UI ha priorita' su env server-side
  const userToggleActive = dedollarFlag;
  const flagColor = userToggleActive ? "var(--reflation)" : "var(--muted)";

  return (
    <div className="card">
      <h2>Score degli asset senza e con il bonus dollaro</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Lo <strong>score puro</strong> viene solo dal regime e dai rendimenti storici. Lo{" "}
        <strong>score con bonus</strong> aggiunge l'effetto della de-dollarizzazione (sensibilità
        dell'asset × punteggio complessivo). La <strong>differenza</strong> dice quanto il dollaro sposta
        ogni asset.
      </div>

      <div style={{
        display: "flex", gap: 18, marginBottom: 14, padding: 10,
        background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 6, flexWrap: "wrap",
      }}>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Regime attuale</div>
          <div style={{ fontSize: 16, fontWeight: 600 }}>{regimeLabel(data.regime)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>De-dollarizzazione</div>
          <div style={{ fontSize: 16, fontWeight: 600 }}>{fmt(data.dedollar_combined_score, 3)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Nel sito adesso</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: flagColor }}>
            {userToggleActive ? "Score con bonus" : "Score puro"}
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Come cambiare</div>
          <div style={{ fontSize: 11 }}>
            Impostazioni › Bonus dollaro: vale subito per classifica, scenari, Monte Carlo e backtest.
            {data.use_dedollar_bonus_active && (
              <div style={{ marginTop: 4, color: "var(--muted)" }}>
                (il server ha il bonus acceso di base: vale quando non scegli tu)
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
              <th className="num">Sensibilità</th>
              <th className="num">Score puro</th>
              <th className="num">Con bonus</th>
              <th className="num">Differenza</th>
              <th>Effetto</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((it) => {
              const color = deltaColor(it.dedollar_delta);
              return (
                <tr key={it.asset}>
                  <td><strong>{assetName(it.asset)}</strong></td>
                  <td className="num" style={{ color: it.sensitivity > 0 ? "var(--reflation)" : it.sensitivity < 0 ? "var(--deflation)" : "var(--muted)" }}>
                    {it.sensitivity > 0 ? "+" : ""}{it.sensitivity.toFixed(0)}
                  </td>
                  <td className="num" style={{ color: "var(--muted)" }}>{fmt(it.pure_score, 1)}</td>
                  <td className="num"><strong>{fmt(it.adjusted_score, 1)}</strong></td>
                  <td className="num" style={{ color, fontWeight: 600 }}>
                    {fmtSigned(it.dedollar_delta, 2)}
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
        <strong>Come si legge.</strong> Sensibilità positiva = l'asset guadagna se il dollaro perde peso
        (oro, argento, bitcoin, emergenti); negativa = ci perde (bond USA, liquidità). Differenza =
        sensibilità × punteggio di de-dollarizzazione. Con il bonus spento (la scelta di base) gli score
        vengono solo dal regime e dai rendimenti storici: qui vedi di quanto cambierebbero accendendolo.
      </div>
    </div>
  );
}
