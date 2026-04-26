import { useState } from "react";
import { api } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";
import type { BacktestResult, BacktestStrategy, LeadTimeReport } from "../types";

const STRATEGY_COLOR: Record<string, string> = {
  macro_score_weighted: "var(--accent)",
  "60_40": "var(--reflation)",
  spy_buyhold: "var(--goldilocks, #06b6d4)",
  equal_weight: "var(--muted)",
};

function pct(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(digits)}%`;
}

function num(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function NavChart({ strategies }: { strategies: BacktestStrategy[] }) {
  if (strategies.length === 0) return null;

  const W = 720;
  const H = 280;
  const PADDING = { top: 20, right: 100, bottom: 30, left: 50 };
  const allDates = strategies[0].nav.map((p) => p.date);
  const startDate = new Date(allDates[0]);
  const endDate = new Date(allDates[allDates.length - 1]);
  const xRange = endDate.getTime() - startDate.getTime();

  // Min/max NAV across all strategies
  let minY = Infinity, maxY = -Infinity;
  strategies.forEach((s) => {
    s.nav.forEach((p) => {
      if (p.value < minY) minY = p.value;
      if (p.value > maxY) maxY = p.value;
    });
  });
  const yRange = maxY - minY || 1;

  const xScale = (date: string) => {
    const t = new Date(date).getTime() - startDate.getTime();
    return PADDING.left + (t / xRange) * (W - PADDING.left - PADDING.right);
  };
  const yScale = (v: number) => {
    return PADDING.top + (1 - (v - minY) / yRange) * (H - PADDING.top - PADDING.bottom);
  };

  return (
    <svg width={W} height={H} style={{ background: "var(--bg)", borderRadius: 6 }}>
      {/* Y-axis grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((p) => {
        const v = minY + p * yRange;
        const y = yScale(v);
        return (
          <g key={p}>
            <line x1={PADDING.left} y1={y} x2={W - PADDING.right} y2={y}
              stroke="var(--border)" strokeDasharray="2 2" opacity={0.3} />
            <text x={PADDING.left - 6} y={y + 3} fontSize="10" fill="var(--muted)" textAnchor="end">
              {v.toFixed(1)}x
            </text>
          </g>
        );
      })}

      {/* X-axis labels (anni) */}
      {[0, 0.25, 0.5, 0.75, 1].map((p) => {
        const t = startDate.getTime() + p * xRange;
        const x = PADDING.left + p * (W - PADDING.left - PADDING.right);
        return (
          <text key={p} x={x} y={H - 10} fontSize="10" fill="var(--muted)" textAnchor="middle">
            {new Date(t).getFullYear()}
          </text>
        );
      })}

      {/* Lines per strategy */}
      {strategies.map((s) => {
        const path = s.nav.map((p, i) => {
          const x = xScale(p.date);
          const y = yScale(p.value);
          return `${i === 0 ? "M" : "L"}${x},${y}`;
        }).join(" ");
        return (
          <path
            key={s.name}
            d={path}
            fill="none"
            stroke={STRATEGY_COLOR[s.name] ?? "var(--text)"}
            strokeWidth={1.8}
          />
        );
      })}

      {/* Legend */}
      {strategies.map((s, i) => (
        <g key={s.name} transform={`translate(${W - PADDING.right + 10}, ${PADDING.top + i * 18})`}>
          <line x1={0} y1={6} x2={12} y2={6} stroke={STRATEGY_COLOR[s.name] ?? "var(--text)"} strokeWidth={2} />
          <text x={16} y={9} fontSize="10" fill="var(--text)">
            {s.name.replace(/_/g, " ")}
          </text>
        </g>
      ))}
    </svg>
  );
}

export function BacktestPanel() {
  const [data, setData] = useState<BacktestResult | null>(null);
  const [leadTime, setLeadTime] = useState<LeadTimeReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startYear, setStartYear] = useState(2003);
  const [topN, setTopN] = useState(5);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const [bt, lt] = await Promise.all([
        api.backtestRun({ startYear, endYear: 2026, topN, threshold: 30.0, costBps: 10.0 }),
        api.backtestLeadTime(0.35, 12),
      ]);
      setData(bt);
      setLeadTime(lt);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Errore backtest");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Backtest portfolio + recession lead time</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Strategy macro score-weighted (top-N asset, monthly rebalance, 10bp cost) vs 3 benchmarks
        (60/40, SPY buy-hold, equal-weight). Senza lookahead bias: pesi al mese t basati su info t-1.
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
        <label style={{ fontSize: 12 }}>
          Start year:{" "}
          <input type="number" min={1990} max={2025} value={startYear}
            onChange={(e) => setStartYear(Number(e.target.value))}
            style={{ width: 70 }} />
        </label>
        <label style={{ fontSize: 12 }}>
          Top N:{" "}
          <input type="number" min={1} max={15} value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
            style={{ width: 50 }} />
        </label>
        <button
          onClick={run} disabled={loading}
          style={{
            padding: "6px 14px", background: "var(--accent)", color: "white",
            border: "none", borderRadius: 6, cursor: loading ? "wait" : "pointer",
          }}
        >
          {loading ? "Running…" : data ? "Riesegui backtest" : "Esegui backtest"}
        </button>
      </div>

      {error && <div className="error" style={{ fontSize: 13 }}>{error}</div>}

      {data && (
        <>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
            Periodo {data.common_start} → {data.common_end} ({data.n_months} mesi)
          </div>

          <div style={{ marginBottom: 16, overflowX: "auto" }}>
            <NavChart strategies={data.strategies} />
          </div>

          <ScrollShadow innerClassName="table-wrap">
            <table className="table" style={{ fontSize: 12 }}>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th className="num">CAGR</th>
                  <th className="num">Vol</th>
                  <th className="num">Sharpe</th>
                  <th className="num">MaxDD</th>
                  <th className="num">Calmar</th>
                  <th className="num">Win%</th>
                  <th className="num">NAV finale</th>
                  <th className="num">Alpha vs 60/40</th>
                  <th className="num">Beta</th>
                  <th className="num">Corr</th>
                </tr>
              </thead>
              <tbody>
                {data.strategies.map((s) => (
                  <tr key={s.name} style={{
                    background: s.name === "macro_score_weighted" ? "rgba(56,189,248,0.08)" : undefined,
                  }}>
                    <td>
                      <span style={{
                        display: "inline-block", width: 8, height: 8,
                        background: STRATEGY_COLOR[s.name] ?? "var(--text)",
                        borderRadius: 4, marginRight: 6,
                      }} />
                      <strong>{s.name.replace(/_/g, " ")}</strong>
                    </td>
                    <td className="num">{pct(s.stats.annualized_return)}</td>
                    <td className="num">{pct(s.stats.annualized_volatility, 1)}</td>
                    <td className="num">{num(s.stats.sharpe)}</td>
                    <td className="num" style={{ color: "var(--deflation)" }}>
                      {pct(s.stats.max_drawdown, 1)}
                    </td>
                    <td className="num">{num(s.stats.calmar)}</td>
                    <td className="num">{pct(s.stats.win_rate, 0)}</td>
                    <td className="num"><strong>{num(s.stats.final_nav)}x</strong></td>
                    <td className="num" style={{
                      color: s.alpha_vs_60_40.alpha > 0 ? "var(--reflation)" : "var(--deflation)",
                    }}>
                      {pct(s.alpha_vs_60_40.alpha)}
                    </td>
                    <td className="num">{num(s.alpha_vs_60_40.beta)}</td>
                    <td className="num">{num(s.alpha_vs_60_40.correlation)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollShadow>
        </>
      )}

      {leadTime && (
        <div style={{ marginTop: 18 }}>
          <h3 style={{ marginBottom: 6 }}>Recession lead time (NBER)</h3>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
            Soglia P(deflation o stagflation) ≥ {leadTime.threshold} · lookback{" "}
            {leadTime.lookback_months}m · {leadTime.n_recessions_analyzed} recessioni · hit rate{" "}
            <strong style={{ color: leadTime.hit_rate > 0.6 ? "var(--reflation)" : "var(--deflation)" }}>
              {pct(leadTime.hit_rate, 0)}
            </strong>{" "}
            · lead medio{" "}
            <strong>
              {leadTime.avg_lead_months !== null ? `${leadTime.avg_lead_months.toFixed(1)}m` : "—"}
            </strong>{" "}
            (mediano{" "}
            {leadTime.median_lead_months !== null ? `${leadTime.median_lead_months.toFixed(1)}m` : "—"})
          </div>
          <ScrollShadow innerClassName="table-wrap">
            <table className="table" style={{ fontSize: 12 }}>
              <thead>
                <tr>
                  <th>Recession start</th>
                  <th>End</th>
                  <th className="num">Durata (m)</th>
                  <th>Signal date</th>
                  <th className="num">Lead (mesi)</th>
                  <th className="num">P stress pre</th>
                  <th className="num">P stress during</th>
                </tr>
              </thead>
              <tbody>
                {leadTime.recessions.map((r) => (
                  <tr key={r.recession_start}>
                    <td>{r.recession_start}</td>
                    <td>{r.recession_end}</td>
                    <td className="num">{r.duration_months}</td>
                    <td>{r.signal_date ?? <span style={{ color: "var(--muted)" }}>nessun segnale</span>}</td>
                    <td className="num" style={{
                      color: r.lead_months !== null && r.lead_months > 0 ? "var(--reflation)" : "var(--muted)",
                      fontWeight: r.lead_months !== null && r.lead_months > 0 ? 600 : 400,
                    }}>
                      {r.lead_months !== null ? r.lead_months.toFixed(1) : "—"}
                    </td>
                    <td className="num">{pct(r.pre_recession_max_prob, 0)}</td>
                    <td className="num">{pct(r.max_prob_during, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollShadow>
        </div>
      )}
    </div>
  );
}
