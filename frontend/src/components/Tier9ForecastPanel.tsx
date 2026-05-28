import { useState } from "react";
import { api } from "../api/client";
import type { MLForecastResponse } from "../types";

/**
 * Tier 9 — ML Forecast Panel.
 *
 * Mostra forecast regime + asset allocation per 1/3/6/12 mesi.
 * Source: /regime/ml-forecast con transition matrix empirica.
 */
export function Tier9ForecastPanel() {
  const [data, setData] = useState<MLForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [horizons] = useState("1,3,6,12");

  // Performance fix 2026-05-28: on-demand load (era auto-fetch al mount,
  // lento per transition matrix estimation interna)
  const load = () => {
    setLoading(true);
    setError(null);
    api
      .mlForecast(horizons, 5)
      .then((d) => {
        setData(d);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  };

  if (!data && !loading) {
    return (
      <div className="card">
        <h3 style={{ margin: 0, marginBottom: 8 }}>Forecast 1-12 mesi (T9)</h3>
        <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginBottom: 12 }}>
          ML forecast con transition matrix empirica. On-demand per ridurre
          page load latency.
        </p>
        <button onClick={load} className="btn">Esegui forecast</button>
        {error && <p style={{ color: "var(--deflation)", marginTop: 8 }}>Errore: {error}</p>}
      </div>
    );
  }
  if (loading) {
    return <div className="card">Computing ML forecast...</div>;
  }
  if (!data) return null;

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <h3 style={{ margin: 0 }}>Forecast 1-12 mesi (T9)</h3>
        {data.ml_blend_active && (
          <span style={{
            padding: "2px 8px",
            borderRadius: 4,
            background: "var(--accent-bg)",
            fontSize: 11,
            fontWeight: 600,
          }}>
            ML blend ON
          </span>
        )}
      </div>

      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase" }}>
          Stato corrente
        </div>
        <div style={{ fontSize: 18, fontWeight: 700, textTransform: "capitalize" }}>
          {data.current_regime}
        </div>
        <div style={{ fontSize: 12 }}>
          {Object.entries(data.current_probs)
            .sort((a, b) => b[1] - a[1])
            .map(([r, p]) => `${r}: ${(p * 100).toFixed(0)}%`)
            .join(" · ")}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
        {data.forecasts.map((fc) => (
          <div key={fc.horizon_months} style={{
            padding: 12,
            border: "1px solid var(--border)",
            borderRadius: 8,
            background: "var(--card-bg-subtle)",
          }}>
            <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase" }}>
              Horizon {fc.horizon_months}m
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, textTransform: "capitalize", marginBottom: 4 }}>
              {fc.expected_regime}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
              entropy {fc.entropy.toFixed(2)}
            </div>
            <div style={{ fontSize: 11, marginBottom: 6 }}>
              {Object.entries(fc.probs)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 2)
                .map(([r, p]) => `${r.slice(0, 4)}: ${(p * 100).toFixed(0)}%`)
                .join(" · ")}
            </div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 4 }}>
              Top-5 asset
            </div>
            <ol style={{ margin: 0, paddingLeft: 18, fontSize: 12 }}>
              {fc.top_assets.slice(0, 5).map((a) => (
                <li key={a.asset}>
                  {a.asset.replace(/_/g, " ")}{" "}
                  <span style={{ color: "var(--text-muted)" }}>
                    ({a.weighted_score.toFixed(0)})
                  </span>
                </li>
              ))}
            </ol>
          </div>
        ))}
      </div>
    </div>
  );
}
