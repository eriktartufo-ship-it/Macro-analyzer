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
