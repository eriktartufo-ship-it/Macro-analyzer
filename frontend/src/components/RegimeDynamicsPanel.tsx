import { useEffect, useState } from "react";
import { api } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";
import type { HMMPrediction, TransitionMatrix } from "../types";

const REGIMES = ["reflation", "stagflation", "deflation", "goldilocks"] as const;

const REGIME_LABEL: Record<string, string> = {
  reflation: "Reflation",
  stagflation: "Stagflation",
  deflation: "Deflation",
  goldilocks: "Goldilocks",
};

function pct(v: number): string {
  if (!Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function heatColor(v: number): string {
  const intensity = Math.min(1, Math.max(0, v));
  const alpha = 0.12 + intensity * 0.55;
  return `rgba(56, 189, 248, ${alpha.toFixed(3)})`;
}

export function RegimeDynamicsPanel() {
  const [matrix, setMatrix] = useState<TransitionMatrix | null>(null);
  const [matrixError, setMatrixError] = useState<string | null>(null);
  const [hmm, setHmm] = useState<HMMPrediction | null>(null);
  const [hmmError, setHmmError] = useState<string | null>(null);
  const [loadingHmm, setLoadingHmm] = useState(false);
  const [horizonDays, setHorizonDays] = useState(30);
  const [projectSteps, setProjectSteps] = useState(6);

  useEffect(() => {
    let alive = true;
    setMatrix(null);
    setMatrixError(null);
    api
      .transitionMatrix(horizonDays, projectSteps)
      .then((d) => {
        if (alive) setMatrix(d);
      })
      .catch((e) => {
        if (alive) setMatrixError(e instanceof Error ? e.message : "Errore matrice");
      });
    return () => {
      alive = false;
    };
  }, [horizonDays, projectSteps]);

  const runHmm = () => {
    setLoadingHmm(true);
    setHmmError(null);
    api
      .hmmPrediction(4)
      .then((d) => setHmm(d))
      .catch((e) => setHmmError(e instanceof Error ? e.message : "Errore HMM"))
      .finally(() => setLoadingHmm(false));
  };

  return (
    <>
      <div className="card">
        <h2>Transition matrix empirica</h2>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
          Probabilita osservate P(r<sub>t+h</sub> | r<sub>t</sub>) sui record di
          RegimeClassification in DB. Richiede backfill storico per essere statisticamente
          robusta.
        </div>
        <div style={{ display: "flex", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
          <label style={{ fontSize: 12 }}>
            Horizon (giorni):{" "}
            <input
              type="number"
              min={7}
              max={365}
              step={1}
              value={horizonDays}
              onChange={(e) => setHorizonDays(Math.max(1, Number(e.target.value) || 30))}
              style={{ width: 70 }}
            />
          </label>
          <label style={{ fontSize: 12 }}>
            Project steps:{" "}
            <input
              type="number"
              min={0}
              max={24}
              step={1}
              value={projectSteps}
              onChange={(e) => setProjectSteps(Math.max(0, Number(e.target.value) || 0))}
              style={{ width: 60 }}
            />
          </label>
        </div>

        {matrixError && (
          <div className="error" style={{ fontSize: 13 }}>
            {matrixError}
          </div>
        )}
        {!matrix && !matrixError && <div className="loading">Caricamento matrice…</div>}

        {matrix && (
          <>
            <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
              {matrix.total_observations} transizioni osservate · periodo{" "}
              {matrix.date_from ?? "—"} → {matrix.date_to ?? "—"} · horizon {matrix.horizon_days}d
            </div>
            <ScrollShadow innerClassName="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>from \ to</th>
                    {REGIMES.map((r) => (
                      <th key={r} className="num">
                        {REGIME_LABEL[r]}
                      </th>
                    ))}
                    <th className="num">Self-trans</th>
                    <th className="num">Avg dur (d)</th>
                  </tr>
                </thead>
                <tbody>
                  {REGIMES.map((rFrom) => (
                    <tr key={rFrom}>
                      <td style={{ fontWeight: 600 }}>{REGIME_LABEL[rFrom]}</td>
                      {REGIMES.map((rTo) => {
                        const p = matrix.probabilities[rFrom]?.[rTo] ?? 0;
                        const c = matrix.counts[rFrom]?.[rTo] ?? 0;
                        return (
                          <td
                            key={rTo}
                            className="num"
                            style={{ background: heatColor(p) }}
                          >
                            {pct(p)}
                            <div style={{ fontSize: 10, color: "var(--muted)" }}>
                              n={c}
                            </div>
                          </td>
                        );
                      })}
                      <td className="num">
                        {pct(matrix.self_transition_probability[rFrom] ?? 0)}
                      </td>
                      <td className="num">
                        {(matrix.avg_duration_days[rFrom] ?? 0).toFixed(0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollShadow>

            {matrix.projected_probabilities && projectSteps > 0 && (() => {
              const projected = matrix.projected_probabilities;
              return (
                <div style={{ marginTop: 14 }}>
                  <div style={{ fontWeight: 600, marginBottom: 6 }}>
                    Proiezione a {projectSteps * matrix.horizon_days} giorni
                  </div>
                  <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                    {REGIMES.map((r) => (
                      <div key={r} style={{ minWidth: 120 }}>
                        <div style={{ fontSize: 12, color: "var(--muted)" }}>
                          {REGIME_LABEL[r]}
                        </div>
                        <div style={{ fontSize: 18, fontWeight: 700 }}>
                          {pct(projected[r] ?? 0)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}
          </>
        )}
      </div>

      <div className="card">
        <h2>HMM regime classifier (Gaussian, 4 stati)</h2>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
          Distribuzione posteriore sullo stato corrente via Baum-Welch sulle feature
          (gdp_roc, cpi_yoy, unrate, pmi, yield_curve_10y2y, lei_roc) standardizzate.
          Training on-demand sulla history in DB.
        </div>
        <button
          onClick={runHmm}
          disabled={loadingHmm}
          style={{
            padding: "6px 14px",
            background: "var(--accent)",
            color: "white",
            border: "none",
            borderRadius: 6,
            cursor: loadingHmm ? "wait" : "pointer",
            marginBottom: 12,
          }}
        >
          {loadingHmm ? "Training…" : hmm ? "Ri-esegui HMM" : "Esegui HMM"}
        </button>

        {hmmError && (
          <div className="error" style={{ fontSize: 13 }}>
            {hmmError}
          </div>
        )}
        {hmm && (
          <>
            <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
              n_training={hmm.n_training} · log-likelihood={hmm.log_likelihood.toFixed(2)}{" "}
              · stato corrente: {hmm.current_state} →{" "}
              {REGIME_LABEL[hmm.state_to_regime[hmm.current_state]] ?? "—"}
            </div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 14 }}>
              {REGIMES.map((r) => (
                <div key={r} style={{ minWidth: 120 }}>
                  <div style={{ fontSize: 12, color: "var(--muted)" }}>{REGIME_LABEL[r]}</div>
                  <div style={{ fontSize: 18, fontWeight: 700 }}>
                    {pct(hmm.probabilities[r] ?? 0)}
                  </div>
                </div>
              ))}
            </div>
            <details>
              <summary style={{ cursor: "pointer", fontSize: 12, color: "var(--muted)" }}>
                Mappatura stato → regime e stats feature
              </summary>
              <div style={{ marginTop: 8, fontSize: 12 }}>
                <div style={{ marginBottom: 8 }}>
                  {Object.entries(hmm.state_to_regime).map(([s, r]) => (
                    <span
                      key={s}
                      style={{
                        display: "inline-block",
                        marginRight: 10,
                        padding: "2px 8px",
                        border: "1px solid var(--border)",
                        borderRadius: 10,
                      }}
                    >
                      state {s} → {REGIME_LABEL[r]}
                    </span>
                  ))}
                </div>
                <table className="table" style={{ fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th className="num">μ</th>
                      <th className="num">σ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(hmm.feature_means).map(([f, mu]) => (
                      <tr key={f}>
                        <td>{f.replace(/_/g, " ")}</td>
                        <td className="num">{mu.toFixed(3)}</td>
                        <td className="num">
                          {(hmm.feature_stds[f] ?? 0).toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </>
        )}
      </div>
    </>
  );
}
