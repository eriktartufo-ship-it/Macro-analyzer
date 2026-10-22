import { useEffect, useState } from "react";
import { api } from "../api/client";
import type {
  ActiveModelInfo,
  PredictionLogRecent,
  PredictionLogStats,
} from "../types";
import { fmtPctSigned, regimeLabel } from "../lib/assets";

const HORIZON_LABEL = { "1m": "1 mese", "3m": "3 mesi", "6m": "6 mesi" } as const;

/**
 * CRITICAL #1+#2 council 2026-05-27 — Live Track Record panel.
 *
 * Mostra il track record realizzato del modello vs benchmark 60/40 a 1m/3m/6m
 * orizzonti. Inizialmente vuoto (l'onestà è la feature: si vede che è appena
 * partito), poi accumula evidence over time.
 *
 * Model snapshot version chiarisce quale "modello sigillato" sta facendo le
 * predictions (es. v1.0-sealed-2026-05-27).
 */
export function LiveTrackRecordPanel() {
  const [horizon, setHorizon] = useState<"1m" | "3m" | "6m">("3m");
  const [stats, setStats] = useState<PredictionLogStats | null>(null);
  const [recent, setRecent] = useState<PredictionLogRecent[]>([]);
  const [model, setModel] = useState<ActiveModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([
      api.predictionLogStats(horizon),
      api.predictionLogRecent(10),
      api.activeModel(),
    ])
      .then(([s, r, m]) => {
        if (cancelled) return;
        setStats(s);
        setRecent(r);
        setModel(m);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [horizon]);

  const fmtPct = (v: number | null | undefined): string => fmtPctSigned(v, 2);

  const fmtAlphaColor = (alpha: number | null): string => {
    if (alpha == null) return "var(--text-muted)";
    if (alpha > 0) return "var(--reflation)";
    if (alpha < 0) return "var(--deflation)";
    return "var(--text-muted)";
  };

  const n = stats?.summary.n_evaluated ?? 0;
  // L'impronta salvata quando il modello è stato sigillato: se diversa da quella attuale,
  // le previsioni recenti non vengono più dal modello sigillato e va detto.
  const sealedHash = model?.snapshots.find((s) => s.version === model.active_version)?.classifier_hash ?? null;

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <h2 style={{ margin: 0 }}>Track record delle previsioni</h2>
        <div style={{ display: "flex", gap: 6 }}>
          {(["1m", "3m", "6m"] as const).map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              className={horizon === h ? "chip chip-active" : "chip"}
              style={{ cursor: "pointer", border: 0 }}
            >
              {HORIZON_LABEL[h]}
            </button>
          ))}
        </div>
      </div>

      <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: 12 }}>
        Modello congelato: <code>{model?.active_version ?? "—"}</code> · impronta del classificatore{" "}
        <code>{model?.current_classifier_hash?.slice(0, 8) ?? "—"}</code>
        {sealedHash && model && sealedHash !== model.current_classifier_hash && (
          <>
            {" "}
            <span style={{ color: "var(--warn-text)" }}>
              (al sigillo era <code>{sealedHash.slice(0, 8)}</code>: il classificatore è cambiato dopo il sigillo)
            </span>
          </>
        )}
        . Ogni previsione si confronta con un portafoglio 60/40 (60% azioni, 40% bond) a {HORIZON_LABEL[horizon]}.
      </div>

      {error && (
        <div style={{ padding: 12, background: "rgba(255,80,80,0.1)", borderRadius: 8 }}>
          Errore: {error}
        </div>
      )}

      {loading && !stats && <div style={{ opacity: 0.6 }}>Caricamento…</div>}

      {stats && (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
              gap: 10,
              marginBottom: 16,
            }}
          >
            <div className="metric-box">
              <div className="metric-label">Previsioni verificate</div>
              <div className="metric-value">{n}</div>
              {n === 0 && (
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  In attesa della prima verifica
                </div>
              )}
            </div>
            <div className="metric-box">
              <div className="metric-label">Extra-rendimento medio sul 60/40</div>
              <div
                className="metric-value"
                style={{ color: fmtAlphaColor(stats.summary.mean_alpha ?? null) }}
              >
                {fmtPct(stats.summary.mean_alpha)}
              </div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Quante battono il 60/40</div>
              <div className="metric-value">
                {stats.summary.hit_rate == null
                  ? "—"
                  : `${(stats.summary.hit_rate * 100).toFixed(0)}%`}
              </div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Rendimento medio ottenuto</div>
              <div className="metric-value">
                {fmtPct(stats.summary.mean_realized_return)}
              </div>
            </div>
          </div>

          {Object.keys(stats.by_regime).length > 0 && (
            <>
              <h3 style={{ fontSize: "0.95rem", margin: "12px 0 6px" }}>
                Per regime previsto
              </h3>
              <div style={{ overflowX: "auto" }}>
                <table className="log-table" style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "left", padding: 6 }}>Regime</th>
                      <th style={{ textAlign: "right", padding: 6 }}>Previsioni</th>
                      <th style={{ textAlign: "right", padding: 6 }}>
                        Extra medio
                      </th>
                      <th style={{ textAlign: "right", padding: 6 }}>Battono il 60/40</th>
                      <th style={{ textAlign: "right", padding: 6 }}>Peggiore</th>
                      <th style={{ textAlign: "right", padding: 6 }}>Migliore</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(stats.by_regime).map(([regime, r]) => (
                      <tr
                        key={regime}
                        style={{ borderTop: "1px solid var(--border)" }}
                      >
                        <td
                          style={{ padding: 6 }}
                          className={`regime-${regime}`}
                        >
                          {regimeLabel(regime)}
                        </td>
                        <td style={{ padding: 6, textAlign: "right" }}>
                          {r.n}
                        </td>
                        <td
                          style={{
                            padding: 6,
                            textAlign: "right",
                            color: fmtAlphaColor(r.mean_alpha),
                          }}
                        >
                          {fmtPct(r.mean_alpha)}
                        </td>
                        <td style={{ padding: 6, textAlign: "right" }}>
                          {(r.hit_rate * 100).toFixed(0)}%
                        </td>
                        <td style={{ padding: 6, textAlign: "right" }}>
                          {fmtPct(r.worst_alpha)}
                        </td>
                        <td style={{ padding: 6, textAlign: "right" }}>
                          {fmtPct(r.best_alpha)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}

      {recent.length > 0 && (
        <>
          <h3 style={{ fontSize: "0.95rem", margin: "16px 0 6px" }}>
            Ultime previsioni
          </h3>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: 6 }}>
            «—» = non ancora verificabile: l'orizzonte non è ancora passato.
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="log-table" style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ fontSize: "0.85rem" }}>
                  <th style={{ textAlign: "left", padding: 6 }}>Data</th>
                  <th style={{ textAlign: "left", padding: 6 }}>Regime</th>
                  <th style={{ textAlign: "right", padding: 6 }}>Fiducia</th>
                  <th style={{ textAlign: "right", padding: 6 }}>Extra 1 mese</th>
                  <th style={{ textAlign: "right", padding: 6 }}>3 mesi</th>
                  <th style={{ textAlign: "right", padding: 6 }}>6 mesi</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((p) => (
                  <tr
                    key={p.id}
                    style={{
                      borderTop: "1px solid var(--border)",
                      fontSize: "0.85rem",
                    }}
                  >
                    <td style={{ padding: 6 }}>{p.date_predicted ?? "—"}</td>
                    <td
                      style={{ padding: 6 }}
                      className={`regime-${p.regime}`}
                    >
                      {regimeLabel(p.regime)}
                    </td>
                    <td style={{ padding: 6, textAlign: "right" }}>
                      {(p.confidence * 100).toFixed(0)}%
                    </td>
                    <td
                      style={{
                        padding: 6,
                        textAlign: "right",
                        color: fmtAlphaColor(p.alpha_1m),
                      }}
                    >
                      {fmtPct(p.alpha_1m)}
                    </td>
                    <td
                      style={{
                        padding: 6,
                        textAlign: "right",
                        color: fmtAlphaColor(p.alpha_3m),
                      }}
                    >
                      {fmtPct(p.alpha_3m)}
                    </td>
                    <td
                      style={{
                        padding: 6,
                        textAlign: "right",
                        color: fmtAlphaColor(p.alpha_6m),
                      }}
                    >
                      {fmtPct(p.alpha_6m)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
