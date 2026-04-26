import { useEffect, useState } from "react";
import { api } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";
import type { TermPremiumPoint, TermPremiumReport } from "../types";

const REGIME_LABEL: Record<string, string> = {
  reflation: "Reflation",
  stagflation: "Stagflation",
  deflation: "Deflation",
  goldilocks: "Goldilocks",
};

function pct(v: number | null, digits = 2): string {
  if (v === null || !Number.isFinite(v)) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(digits)}%`;
}

function pctOnly(v: number | null, digits = 0): string {
  if (v === null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function DualLineChart({ points }: { points: TermPremiumPoint[] }) {
  if (points.length === 0) return null;
  const W = 720;
  const H = 280;
  const PAD = { top: 16, right: 100, bottom: 30, left: 50 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const startTs = new Date(points[0].date).getTime();
  const endTs = new Date(points[points.length - 1].date).getTime();
  const xRange = endTs - startTs || 1;

  const allValues = points.flatMap((p) => [p.fitted_yield, p.term_premium, p.expected_path]);
  const minY = Math.min(...allValues, -1);
  const maxY = Math.max(...allValues);
  const yRange = maxY - minY || 1;

  const xScale = (d: string) => PAD.left + ((new Date(d).getTime() - startTs) / xRange) * innerW;
  const yScale = (v: number) => PAD.top + (1 - (v - minY) / yRange) * innerH;

  const buildPath = (key: keyof TermPremiumPoint) =>
    points
      .map((p, i) => `${i === 0 ? "M" : "L"}${xScale(p.date)},${yScale(p[key] as number)}`)
      .join(" ");

  // Zero line per term premium reference
  const yZero = yScale(0);

  return (
    <svg width={W} height={H} style={{ background: "var(--bg)", borderRadius: 6 }}>
      {[0, 0.25, 0.5, 0.75, 1].map((p) => {
        const v = minY + p * yRange;
        const y = yScale(v);
        return (
          <g key={p}>
            <line x1={PAD.left} y1={y} x2={PAD.left + innerW} y2={y}
              stroke="var(--border)" strokeDasharray="2 2" opacity={0.3} />
            <text x={PAD.left - 6} y={y + 3} fontSize="10" fill="var(--muted)" textAnchor="end">
              {v.toFixed(1)}%
            </text>
          </g>
        );
      })}

      {/* Linea zero evidenziata per term premium */}
      {minY < 0 && maxY > 0 && (
        <line x1={PAD.left} y1={yZero} x2={PAD.left + innerW} y2={yZero}
          stroke="var(--text)" strokeWidth={0.6} opacity={0.5} />
      )}

      {[0, 0.25, 0.5, 0.75, 1].map((p) => {
        const t = startTs + p * xRange;
        const x = PAD.left + p * innerW;
        return (
          <text key={p} x={x} y={H - 8} fontSize="10" fill="var(--muted)" textAnchor="middle">
            {new Date(t).getFullYear()}
          </text>
        );
      })}

      {/* Linee */}
      <path d={buildPath("fitted_yield")} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
      <path d={buildPath("expected_path")} fill="none" stroke="var(--reflation, #10b981)"
        strokeWidth={1.4} strokeDasharray="4 2" />
      <path d={buildPath("term_premium")} fill="none" stroke="var(--deflation, #ef4444)" strokeWidth={1.6} />

      {/* Legend */}
      <g transform={`translate(${PAD.left + innerW + 10}, ${PAD.top})`}>
        <line x1={0} y1={3} x2={14} y2={3} stroke="var(--accent)" strokeWidth={1.6} />
        <text x={18} y={6} fontSize="10" fill="var(--text)">fitted 10Y</text>
        <line x1={0} y1={20} x2={14} y2={20} stroke="var(--reflation, #10b981)"
          strokeWidth={1.4} strokeDasharray="4 2" />
        <text x={18} y={23} fontSize="10" fill="var(--text)">expected path</text>
        <line x1={0} y1={37} x2={14} y2={37} stroke="var(--deflation, #ef4444)" strokeWidth={1.6} />
        <text x={18} y={40} fontSize="10" fill="var(--text)">term premium</text>
      </g>
    </svg>
  );
}

export function TermPremiumPanel() {
  const [data, setData] = useState<TermPremiumReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.40);
  const [years, setYears] = useState(15);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .termPremiumReport(threshold, years * 365)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(e instanceof Error ? e.message : "Errore term premium"))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [threshold, years]);

  if (error)
    return (
      <div className="card">
        <h2>Term premium decomposition (Adrian-Crump-Moench)</h2>
        <div className="error" style={{ fontSize: 13 }}>{error}</div>
      </div>
    );

  return (
    <div className="card">
      <h2>Term premium decomposition (Adrian-Crump-Moench)</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Decompone <strong>fitted 10Y yield</strong> in <strong>expected path</strong>{" "}
        (cosa il mercato si aspetta dalla Fed) + <strong>term premium</strong> (compensation
        per duration risk). Fonte: NY Fed ACM model. TP &gt; 0 = mercato vuole compensation;
        TP &lt; 0 = flight to quality / QE-driven scarcity.
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 12, flexWrap: "wrap" }}>
        <label style={{ fontSize: 12 }}>
          Years window:{" "}
          <input
            type="number" min={2} max={35} step={1} value={years}
            onChange={(e) => setYears(Number(e.target.value))}
            style={{ width: 60 }}
          />
        </label>
        <label style={{ fontSize: 12 }}>
          Threshold prob:{" "}
          <input
            type="number" min={0.20} max={0.80} step={0.05} value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            style={{ width: 60 }}
          />
        </label>
        {data && (
          <span style={{ fontSize: 11, color: "var(--muted)", alignSelf: "center" }}>
            {data.common_period[0]} → {data.common_period[1]} ({data.n_observations} mesi)
          </span>
        )}
      </div>

      {loading && <div className="loading">Computing decomposition…</div>}

      {data && !loading && (
        <>
          <div style={{ overflowX: "auto", marginBottom: 16 }}>
            <DualLineChart points={data.points} />
          </div>

          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
            Per regime — quale parte del 10Y guida il livello?
          </div>
          <ScrollShadow innerClassName="table-wrap">
            <table className="table" style={{ fontSize: 12 }}>
              <thead>
                <tr>
                  <th>Regime</th>
                  <th className="num">n</th>
                  <th className="num">Fitted 10Y</th>
                  <th className="num">Expected path</th>
                  <th className="num">Term premium</th>
                  <th className="num">% TP &gt; 0</th>
                </tr>
              </thead>
              <tbody>
                {data.by_regime.map((s) => (
                  <tr key={s.regime}>
                    <td><strong>{REGIME_LABEL[s.regime] ?? s.regime}</strong></td>
                    <td className="num">{s.n_observations}</td>
                    {s.n_observations < 6 ? (
                      <td colSpan={4} className="num" style={{ color: "var(--muted)" }}>
                        insufficient data
                      </td>
                    ) : (
                      <>
                        <td className="num">{pct(s.mean_fitted)}</td>
                        <td className="num" style={{ color: "var(--reflation, #10b981)" }}>
                          {pct(s.mean_expected_path)}
                        </td>
                        <td className="num" style={{
                          color:
                            s.mean_term_premium && s.mean_term_premium > 0
                              ? "var(--deflation, #ef4444)"
                              : "var(--reflation, #10b981)",
                          fontWeight: 600,
                        }}>
                          {pct(s.mean_term_premium)}
                        </td>
                        <td className="num">{pctOnly(s.pct_term_premium_positive)}</td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollShadow>

          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 12 }}>
            <strong>Lettura.</strong> In <em>stagflation</em> il TP cresce (mercato richiede
            compensation per inflation/duration risk → '70s, 2022). In <em>deflation</em> il
            TP puo' essere alto (uncertainty) ma l'expected path crolla (mercato prevede tagli
            Fed). In <em>reflation</em> early-cycle e <em>goldilocks</em> il TP e' basso o
            negativo (low risk premium, QE/scarcity). Quando l'utente vede yield 10Y =
            5%, sapere se e' guidato dal path (Fed hawkish) o dal TP (risk premium) cambia
            la strategia di duration.
          </div>
        </>
      )}
    </div>
  );
}
