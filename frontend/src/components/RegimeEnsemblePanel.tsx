import { useState } from "react";
import { api } from "../api/client";
import type { EnsembleResult } from "../types";

const REGIMES = ["reflation", "stagflation", "deflation", "goldilocks"] as const;

const REGIME_LABEL: Record<string, string> = {
  reflation: "Reflation",
  stagflation: "Stagflation",
  deflation: "Deflation",
  goldilocks: "Goldilocks",
};

const MODEL_LABEL: Record<string, string> = {
  rule_based: "Rule-based",
  hmm_market: "HMM-Market",
  msvar: "MS-VAR",
};

const MODEL_DESCRIPTION: Record<string, string> = {
  rule_based: "Conditions su 11+ indicatori macro (CPI, GDP, unrate, VIX, NFCI…)",
  hmm_market: "HMM su features di mercato indipendenti (yield curve, credit spread, copper/gold, momentum)",
  msvar: "Markov-Switching su S&P 500 returns mensili (Hamilton 1989)",
};

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function regimeColor(regime: string): string {
  const map: Record<string, string> = {
    reflation: "var(--reflation)",
    stagflation: "var(--stagflation, #d97706)",
    deflation: "var(--deflation)",
    goldilocks: "var(--goldilocks, #06b6d4)",
  };
  return map[regime] || "var(--text)";
}

function probBar(p: number, color: string) {
  return (
    <div
      style={{
        height: 6,
        background: "var(--bg)",
        borderRadius: 3,
        overflow: "hidden",
        marginTop: 4,
      }}
    >
      <div
        style={{
          height: "100%",
          width: `${Math.max(2, p * 100)}%`,
          background: color,
        }}
      />
    </div>
  );
}

/** Estrae la distribuzione state→regime dai metadata di una view (msvar, hmm_market).
 *  Risolve il bug pre-fix dove il mapping hard 1-a-1 etichettava arbitrariamente
 *  stati con correlazioni deboli. Ora ogni stato ha una distribuzione completa
 *  sui 4 regimi (clip + smoothing prior alpha=0.05). */
function parseStateRegimeDist(
  metadata: Record<string, unknown> | null,
): Record<string, Record<string, number>> | null {
  if (!metadata) return null;
  const raw = metadata["state_to_regime_distribution"];
  if (!raw || typeof raw !== "object") return null;
  const out: Record<string, Record<string, number>> = {};
  for (const [stateId, regimeMapRaw] of Object.entries(raw as Record<string, unknown>)) {
    if (!regimeMapRaw || typeof regimeMapRaw !== "object") continue;
    const inner: Record<string, number> = {};
    for (const [regime, prob] of Object.entries(regimeMapRaw as Record<string, unknown>)) {
      if (typeof prob === "number") inner[regime] = prob;
    }
    out[stateId] = inner;
  }
  return Object.keys(out).length > 0 ? out : null;
}

/** Heatmap stato × regime: opacità proporzionale a P(regime|stato).
 *  Mostra la "vera" composizione di ogni stato latente, sostituendo l'etichetta
 *  hard arbitraria con la distribuzione informativa. */
function StateRegimeHeatmap({
  dist,
  dominantMap,
}: {
  dist: Record<string, Record<string, number>>;
  dominantMap?: Record<string, string>;
}) {
  const stateIds = Object.keys(dist).sort();
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>
        Distribuzione P(regime | stato latente) — opacità ∝ massa
      </div>
      <table
        style={{
          fontSize: 11,
          borderCollapse: "collapse",
          width: "100%",
        }}
      >
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "2px 6px", color: "var(--muted)", fontWeight: 500 }}>
              Stato
            </th>
            {REGIMES.map((r) => (
              <th
                key={r}
                style={{
                  textAlign: "center",
                  padding: "2px 6px",
                  color: regimeColor(r),
                  fontWeight: 600,
                }}
              >
                {REGIME_LABEL[r]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {stateIds.map((sid) => {
            const row = dist[sid] ?? {};
            const dominant = dominantMap?.[sid];
            return (
              <tr key={sid}>
                <td style={{ padding: "3px 6px", color: "var(--muted)" }}>
                  state {sid}
                  {dominant && (
                    <span style={{ marginLeft: 4, fontSize: 10, color: regimeColor(dominant) }}>
                      → {REGIME_LABEL[dominant]?.[0] ?? "?"}
                    </span>
                  )}
                </td>
                {REGIMES.map((r) => {
                  const p = row[r] ?? 0;
                  return (
                    <td
                      key={r}
                      style={{
                        padding: "3px 6px",
                        textAlign: "center",
                        background: regimeColor(r),
                        color: p > 0.4 ? "white" : "var(--text)",
                        opacity: 0.15 + p * 0.85,
                        fontWeight: p > 0.3 ? 600 : 400,
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {(p * 100).toFixed(0)}%
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function RegimeEnsemblePanel() {
  const [data, setData] = useState<EnsembleResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    api
      .regimeEnsemble()
      .then((d) => setData(d))
      .catch((e) => setError(e instanceof Error ? e.message : "Errore"))
      .finally(() => setLoading(false));
  };

  return (
    <div className="card">
      <h2>Regime ensemble — 3 modelli indipendenti</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Combina rule-based, HMM-Market e MS-VAR. JS divergence pairwise misura il
        disaccordo: alto disaccordo = incertezza, prudenza nell'allocation.
      </div>

      <button
        onClick={load}
        disabled={loading}
        style={{
          padding: "6px 14px",
          background: "var(--accent)",
          color: "white",
          border: "none",
          borderRadius: 6,
          cursor: loading ? "wait" : "pointer",
          marginBottom: 14,
        }}
      >
        {loading ? "Computing…" : data ? "Aggiorna ensemble" : "Esegui ensemble"}
      </button>

      {error && (
        <div className="error" style={{ fontSize: 13 }}>
          {error}
        </div>
      )}

      {data && (
        <>
          <div
            style={{
              padding: 12,
              background: "var(--bg)",
              borderRadius: 8,
              border: data.high_disagreement
                ? "1px solid var(--deflation)"
                : "1px solid var(--border)",
              marginBottom: 14,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <div>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>Dominante ensemble:</span>{" "}
                <strong style={{ fontSize: 16, color: regimeColor(data.dominant_regime) }}>
                  {REGIME_LABEL[data.dominant_regime] ?? data.dominant_regime}
                </strong>{" "}
                <span style={{ fontSize: 13 }}>
                  ({pct(data.ensemble_probabilities[data.dominant_regime])})
                </span>
              </div>
              <div>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>Confidence:</span>{" "}
                <strong style={{ color: data.confidence > 0.6 ? "var(--reflation)" : "var(--deflation)" }}>
                  {pct(data.confidence)}
                </strong>{" "}
                <span style={{ fontSize: 11, color: "var(--muted)" }}>
                  · disagreement {data.disagreement_score.toFixed(3)}
                </span>
              </div>
            </div>
            {data.high_disagreement && (
              <div style={{ marginTop: 8, fontSize: 12, color: "var(--deflation)", fontWeight: 600 }}>
                ⚠ Alto disaccordo tra modelli — prudenza nell'interpretazione
              </div>
            )}
            <div style={{ display: "flex", gap: 12, marginTop: 10, flexWrap: "wrap" }}>
              {REGIMES.map((r) => (
                <div key={r} style={{ minWidth: 110, flex: 1 }}>
                  <div style={{ fontSize: 11, color: "var(--muted)" }}>{REGIME_LABEL[r]}</div>
                  <div style={{ fontSize: 14, fontWeight: 600 }}>
                    {pct(data.ensemble_probabilities[r] ?? 0)}
                  </div>
                  {probBar(data.ensemble_probabilities[r] ?? 0, regimeColor(r))}
                </div>
              ))}
            </div>
          </div>

          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8 }}>Per modello</div>
          <div style={{ display: "grid", gap: 12 }}>
            {data.views.map((v) => (
              <div
                key={v.name}
                style={{
                  padding: 10,
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  background: v.error ? "rgba(220,38,38,0.08)" : "transparent",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <div>
                    <strong>{MODEL_LABEL[v.name] ?? v.name}</strong>
                    <span style={{ fontSize: 11, color: "var(--muted)", marginLeft: 8 }}>
                      peso {((data.weights[v.name] ?? 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  {v.error && (
                    <span style={{ fontSize: 11, color: "var(--deflation)" }}>err: {v.error}</span>
                  )}
                </div>
                <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6 }}>
                  {MODEL_DESCRIPTION[v.name]}
                </div>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  {REGIMES.map((r) => (
                    <div key={r} style={{ minWidth: 90 }}>
                      <div style={{ fontSize: 10, color: "var(--muted)" }}>{REGIME_LABEL[r]}</div>
                      <div style={{ fontSize: 13, fontWeight: 600 }}>
                        {pct(v.probabilities[r] ?? 0)}
                      </div>
                      {probBar(v.probabilities[r] ?? 0, regimeColor(r))}
                    </div>
                  ))}
                </div>
                {(() => {
                  const dist = parseStateRegimeDist(v.metadata);
                  if (!dist) return null;
                  const dominantMapRaw = v.metadata?.["state_to_regime"];
                  const dominantMap =
                    dominantMapRaw && typeof dominantMapRaw === "object"
                      ? Object.fromEntries(
                          Object.entries(dominantMapRaw as Record<string, unknown>).map(
                            ([k, val]) => [k, typeof val === "string" ? val : ""],
                          ),
                        )
                      : undefined;
                  return <StateRegimeHeatmap dist={dist} dominantMap={dominantMap} />;
                })()}
              </div>
            ))}
          </div>

          {data.notes.length > 0 && (
            <div style={{ marginTop: 12, fontSize: 11, color: "var(--muted)" }}>
              {data.notes.map((n, i) => (
                <div key={i}>• {n}</div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
