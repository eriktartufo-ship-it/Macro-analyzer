import { useEffect, useState } from "react";

interface FailurePattern {
  name: string;
  n: number;
  mean_alpha: number;
  description: string;
  affected_regimes: string[];
  suggested_action: string;
}

interface RecentFailure {
  id: number;
  date_predicted: string | null;
  regime: string;
  confidence: number;
  horizon: string;
  alpha: number;
  severity: "minor" | "moderate" | "severe";
  regime_was_boundary: boolean;
  confidence_was_low: boolean;
  findings: string[];
  suggestions: string[];
}

interface LearningsResponse {
  summary: {
    horizon: string;
    n_evaluated: number;
    n_failures: number;
    failure_rate: number;
    mean_failure_alpha: number | null;
    patterns: FailurePattern[];
    top_actions: string[];
    diagnostic: string;
  };
  recent_failures: RecentFailure[];
}

const BASE = "/api/v1";

async function fetchLearnings(horizon: string): Promise<LearningsResponse> {
  const r = await fetch(`${BASE}/prediction-log/learnings?horizon=${horizon}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/**
 * NEW council 2026-05-27 — Failures & Learnings panel.
 *
 * Reverse engineering automatic dei trade failed. Mostra:
 * - Aggregate failure rate + mean alpha
 * - Patterns identificati (regime systematic, low confidence, boundary cases)
 * - Top 3 suggested actions
 * - Recent failures con findings + suggestions individuali
 */
export function FailureLearningsPanel() {
  const [horizon, setHorizon] = useState<"1m" | "3m" | "6m">("3m");
  const [data, setData] = useState<LearningsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchLearnings(horizon)
      .then((d) => !cancelled && setData(d))
      .catch((e: unknown) =>
        !cancelled && setError(e instanceof Error ? e.message : String(e)),
      )
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [horizon]);

  const severityColor = (s: string): string => {
    if (s === "severe") return "var(--deflation)";
    if (s === "moderate") return "var(--stagflation)";
    return "var(--muted)";
  };

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <h2 style={{ margin: 0 }}>🔬 Failures & Learnings</h2>
        <div style={{ display: "flex", gap: 6 }}>
          {(["1m", "3m", "6m"] as const).map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              className={horizon === h ? "chip chip-active" : "chip"}
              style={{ cursor: "pointer", border: 0 }}
            >
              {h}
            </button>
          ))}
        </div>
      </div>

      <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: 12 }}>
        Reverse engineering automatico dei trade failed. Pattern-detection per
        migliorare il modello iterativamente.
      </div>

      {error && (
        <div style={{ padding: 10, background: "rgba(255,80,80,0.1)", borderRadius: 6 }}>
          Error: {error}
        </div>
      )}

      {loading && !data && <div style={{ opacity: 0.6 }}>Loading...</div>}

      {data && (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
              gap: 10,
              marginBottom: 12,
            }}
          >
            <div className="metric-box">
              <div className="metric-label">Failures</div>
              <div className="metric-value">
                {data.summary.n_failures}/{data.summary.n_evaluated}
              </div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Failure rate</div>
              <div className="metric-value">
                {(data.summary.failure_rate * 100).toFixed(0)}%
              </div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Mean failure alpha</div>
              <div className="metric-value" style={{ color: "var(--deflation)" }}>
                {data.summary.mean_failure_alpha == null
                  ? "—"
                  : `${(data.summary.mean_failure_alpha * 100).toFixed(1)}%`}
              </div>
            </div>
          </div>

          <div style={{ fontSize: "0.85rem", marginBottom: 12 }}>
            <strong>Diagnostic:</strong> {data.summary.diagnostic}
          </div>

          {data.summary.top_actions.length > 0 && (
            <>
              <h3 style={{ fontSize: "0.95rem", margin: "12px 0 6px" }}>
                Top 3 suggested actions
              </h3>
              <ol style={{ fontSize: "0.9rem", paddingLeft: 20, marginTop: 0 }}>
                {data.summary.top_actions.map((a, i) => (
                  <li key={i} style={{ marginBottom: 4 }}>{a}</li>
                ))}
              </ol>
            </>
          )}

          {data.summary.patterns.length > 0 && (
            <>
              <h3 style={{ fontSize: "0.95rem", margin: "12px 0 6px" }}>
                Failure patterns identified
              </h3>
              {data.summary.patterns.map((p) => (
                <div
                  key={p.name}
                  style={{
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    padding: 10,
                    marginBottom: 8,
                    fontSize: "0.85rem",
                  }}
                >
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>
                    <code>{p.name}</code> ·{" "}
                    <span style={{ color: "var(--muted)" }}>
                      {p.n} occurrences, mean α {(p.mean_alpha * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ color: "var(--text)", marginBottom: 4 }}>
                    {p.description}
                  </div>
                  <div style={{ color: "var(--accent)" }}>
                    → {p.suggested_action}
                  </div>
                </div>
              ))}
            </>
          )}

          {data.recent_failures.length > 0 && (
            <>
              <h3 style={{ fontSize: "0.95rem", margin: "16px 0 6px" }}>
                Recent failures (post-mortem)
              </h3>
              {data.recent_failures.slice(0, 5).map((f) => (
                <details
                  key={f.id}
                  style={{
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    padding: 8,
                    marginBottom: 6,
                    fontSize: "0.85rem",
                  }}
                >
                  <summary style={{ cursor: "pointer" }}>
                    <span style={{ color: severityColor(f.severity), fontWeight: 600 }}>
                      [{f.severity}]
                    </span>{" "}
                    {f.date_predicted ?? "—"} ·{" "}
                    <span className={`regime-${f.regime}`}>{f.regime}</span> ·{" "}
                    α {(f.alpha * 100).toFixed(1)}% · conf{" "}
                    {(f.confidence * 100).toFixed(0)}%
                  </summary>
                  <div style={{ paddingLeft: 12, paddingTop: 8 }}>
                    {f.findings.length > 0 && (
                      <>
                        <strong>Findings:</strong>
                        <ul style={{ marginTop: 4, marginBottom: 8 }}>
                          {f.findings.map((x, i) => (
                            <li key={i}>{x}</li>
                          ))}
                        </ul>
                      </>
                    )}
                    {f.suggestions.length > 0 && (
                      <>
                        <strong>Suggestions:</strong>
                        <ul style={{ marginTop: 4, marginBottom: 0, color: "var(--accent)" }}>
                          {f.suggestions.map((x, i) => (
                            <li key={i}>{x}</li>
                          ))}
                        </ul>
                      </>
                    )}
                  </div>
                </details>
              ))}
            </>
          )}
        </>
      )}
    </div>
  );
}
