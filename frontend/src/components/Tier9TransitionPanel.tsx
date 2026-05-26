import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { TransitionCheckResponse } from "../types";

/**
 * Tier 9 — Transition Check Panel.
 *
 * Espone l'output del defensive transition mode + financial-stress veto:
 * - Stato regime + confidence (live)
 * - 3 trigger transition + veto
 * - Allocazione defensive suggerita se transition attiva
 */
export function Tier9TransitionPanel() {
  const [data, setData] = useState<TransitionCheckResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .transitionCheck()
      .then((d) => {
        if (alive) {
          setData(d);
          setError(null);
        }
      })
      .catch((e) => {
        if (alive) setError(String(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  if (loading) {
    return <div className="card">Caricamento transition check...</div>;
  }
  if (error || !data) {
    return (
      <div className="card">
        <h3>Transition Check</h3>
        <p style={{ color: "var(--text-muted)" }}>
          Endpoint /regime/transition-check non disponibile (flag OFF). Errore: {error}
        </p>
      </div>
    );
  }

  const isTransition = data.is_transition;
  const bgColor = isTransition ? "var(--warn-bg)" : "var(--success-bg)";
  const accentColor = isTransition ? "var(--warn-text)" : "var(--reflation)";

  return (
    <div className="card" style={{ background: bgColor }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <h3 style={{ margin: 0 }}>Transition Check (T9)</h3>
        <span style={{
          padding: "4px 12px",
          borderRadius: 6,
          background: accentColor,
          color: "white",
          fontWeight: 700,
          fontSize: 12,
        }}>
          {isTransition ? "TRANSITION" : "STABLE"}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase" }}>
            Max Prob
          </div>
          <div style={{ fontSize: 20, fontWeight: 700 }}>
            {(data.max_prob * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            threshold 38%
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase" }}>
            Entropy
          </div>
          <div style={{ fontSize: 20, fontWeight: 700 }}>{data.entropy.toFixed(2)}</div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            max 2.00 (uniforme)
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase" }}>
            Confidence
          </div>
          <div style={{ fontSize: 20, fontWeight: 700 }}>
            {(data.confidence * 100).toFixed(0)}%
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
            threshold 30%
          </div>
        </div>
      </div>

      <p style={{ marginBottom: 12, color: accentColor, fontWeight: 500 }}>
        {data.description}
      </p>

      {data.triggers_fired.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 6 }}>
            Trigger fired ({data.triggers_count}/3)
          </div>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
            {data.triggers_fired.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </div>
      )}

      {data.defensive_allocation && (
        <div style={{ marginTop: 16, padding: 12, background: "rgba(255,255,255,0.5)", borderRadius: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>
            Defensive Allocation suggerita
          </div>
          <table style={{ width: "100%", fontSize: 13 }}>
            <tbody>
              {Object.entries(data.defensive_allocation)
                .sort((a, b) => b[1] - a[1])
                .map(([asset, weight]) => (
                  <tr key={asset}>
                    <td style={{ padding: "2px 0", textTransform: "capitalize" }}>
                      {asset.replace(/_/g, " ")}
                    </td>
                    <td style={{ textAlign: "right", fontWeight: 600 }}>
                      {(weight * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
