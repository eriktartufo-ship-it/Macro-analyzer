import { useEffect, useState } from "react";
import { api } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";
import type { FactorRegimeReport, FactorRegimeStat } from "../types";

const REGIME_LABEL: Record<string, string> = {
  reflation: "Reflation",
  stagflation: "Stagflation",
  deflation: "Deflation",
  goldilocks: "Goldilocks",
};

const FACTOR_DESCRIPTION: Record<string, string> = {
  "Mkt-RF": "Market premium (S&P 500 - T-bill)",
  SMB: "Small Minus Big (size factor)",
  HML: "High Minus Low B/M (value vs growth)",
  Mom: "Momentum (12-1 month winners minus losers)",
};

function pct(v: number | null, digits = 1): string {
  if (v === null || !Number.isFinite(v)) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(digits)}%`;
}

function num(v: number | null, digits = 2): string {
  if (v === null || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

/** Heatmap color: -3 → rosso intenso, 0 → neutro, +3 → verde intenso */
function sharpeColor(sharpe: number | null): string {
  if (sharpe === null || !Number.isFinite(sharpe)) return "transparent";
  const clipped = Math.max(-2, Math.min(2, sharpe));
  const intensity = Math.abs(clipped) / 2;
  const alpha = 0.10 + intensity * 0.55;
  if (clipped > 0) return `rgba(16, 185, 129, ${alpha.toFixed(3)})`;
  return `rgba(239, 68, 68, ${alpha.toFixed(3)})`;
}

function indexBy(stats: FactorRegimeStat[]): Record<string, Record<string, FactorRegimeStat>> {
  const out: Record<string, Record<string, FactorRegimeStat>> = {};
  for (const s of stats) {
    if (!out[s.factor]) out[s.factor] = {};
    out[s.factor][s.regime] = s;
  }
  return out;
}

export function FactorRegimePanel() {
  const [data, setData] = useState<FactorRegimeReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.40);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .factorRegimeMapping(threshold)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(e instanceof Error ? e.message : "Errore factor mapping"))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [threshold]);

  if (error)
    return (
      <div className="card">
        <h2>Fama-French factor performance × regime</h2>
        <div className="error" style={{ fontSize: 13 }}>{error}</div>
      </div>
    );

  return (
    <div className="card">
      <h2>Fama-French factor performance × regime</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Performance annualizzata dei 4 fattori (Mkt-RF, SMB, HML, Mom) per regime macro.
        Heatmap su Sharpe (verde = factor funziona, rosso = factor perde). Fonte: Kenneth
        French Data Library 1926+.
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
        <label style={{ fontSize: 12 }}>
          Threshold prob_regime:{" "}
          <input
            type="number" min={0.20} max={0.80} step={0.05} value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            style={{ width: 60 }}
          />
        </label>
        {data && (
          <span style={{ fontSize: 11, color: "var(--muted)", alignSelf: "center" }}>
            Periodo {data.common_period[0]} → {data.common_period[1]} ({data.n_months_analyzed} mesi)
          </span>
        )}
      </div>

      {loading && <div className="loading">Computing factor mapping…</div>}

      {data && !loading && (
        <ScrollShadow innerClassName="table-wrap">
          <table className="table" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>Factor</th>
                <th>Description</th>
                {data.regimes.map((r) => (
                  <th key={r} className="num">{REGIME_LABEL[r] ?? r}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.factor_keys.map((factor) => {
                const byRegime = indexBy(data.stats)[factor] ?? {};
                return (
                  <tr key={factor}>
                    <td><strong>{factor}</strong></td>
                    <td style={{ fontSize: 11, color: "var(--muted)" }}>
                      {FACTOR_DESCRIPTION[factor]}
                    </td>
                    {data.regimes.map((regime) => {
                      const s = byRegime[regime];
                      if (!s || s.n_observations < 6) {
                        return (
                          <td key={regime} className="num" style={{ color: "var(--muted)" }}>
                            n={s?.n_observations ?? 0}
                          </td>
                        );
                      }
                      return (
                        <td key={regime} className="num" style={{
                          background: sharpeColor(s.sharpe), padding: 6,
                        }}>
                          <div style={{ fontWeight: 600 }}>{pct(s.mean_annual)}</div>
                          <div style={{ fontSize: 10, color: "var(--muted)" }}>
                            σ {pct(s.vol_annual, 0)} · SR {num(s.sharpe)}
                          </div>
                          <div style={{ fontSize: 10, color: "var(--muted)" }}>
                            n={s.n_observations} · win {pct(s.win_rate, 0)}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </ScrollShadow>
      )}

      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 12 }}>
        <strong>Lettura.</strong> Verde = il factor produce alpha in quel regime, rosso =
        il factor sottoperforma. Esempi storici: Mom funziona in stagflation '70s
        (commodity trends), HML funziona in early-cycle reflation (value rotation 2020-21),
        Mkt-RF brilla in goldilocks (bull steady), tutti i factor crollano in deflation
        (whipsaw 2008/2020). Permette ranking sub-equity (size, value, momentum)
        regime-conditional dentro il book azionario.
      </div>
    </div>
  );
}
