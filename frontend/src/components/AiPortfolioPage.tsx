import { useEffect, useState } from "react";
import { api } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";
import type {
  AiPortfolioDecision,
  AiPortfolioPerformanceResponse,
  AiPortfolioPosition,
} from "../types";

const ACTION_COLORS: Record<string, string> = {
  OPEN_TRANCHE: "var(--reflation)",
  CLOSE_TRANCHE: "var(--stagflation)",
  FULL_CLOSE: "var(--deflation)",
  STOP_LOSS: "var(--deflation)",
  TAKE_PROFIT: "var(--reflation)",
  HOLD: "var(--muted)",
  SKIP_NO_SIGNAL: "var(--muted)",
};

const MOMENTUM_COLORS: Record<string, string> = {
  BULLISH: "var(--reflation)",
  NEUTRAL: "var(--muted)",
  BEARISH: "var(--deflation)",
};

function pct(v: number | null | undefined, digits = 2): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function money(v: number | null | undefined): string {
  if (v == null) return "—";
  return `$${v.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}

function formatAsset(a: string): string {
  return a.replace(/_/g, " ");
}

export function AiPortfolioPage() {
  const [positions, setPositions] = useState<AiPortfolioPosition[]>([]);
  const [decisions, setDecisions] = useState<AiPortfolioDecision[]>([]);
  const [perf, setPerf] = useState<AiPortfolioPerformanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [p, d, perfRes] = await Promise.all([
        api.aiPortfolioPositions(),
        api.aiPortfolioDecisions(30),
        api.aiPortfolioPerformance(180),
      ]);
      setPositions(p);
      setDecisions(d);
      setPerf(perfRes);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Errore caricamento");
    } finally {
      setLoading(false);
    }
  };

  const manualScan = async () => {
    setScanning(true);
    try {
      await api.aiPortfolioManualScan();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Scan fallito");
    } finally {
      setScanning(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const current = perf?.current;
  const noData = !current && positions.length === 0 && decisions.length === 0;

  return (
    <div>
      {error && <div className="error">{error}</div>}

      {/* Hero metrics */}
      <div className="card" style={{ marginTop: 12 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: 12,
            marginBottom: 12,
          }}
        >
          <div>
            <h2 style={{ margin: 0 }}>🤖 AI Portfolio — Paper Trading</h2>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
              Capitale base $100k · Scan giornaliero autonomo · DCA tranche dinamiche
            </div>
          </div>
          <button
            onClick={manualScan}
            disabled={scanning}
            style={{
              background: "var(--surface-sunk)",
              border: "1px solid var(--stroke)",
              color: "var(--text)",
              borderRadius: 6,
              padding: "6px 12px",
              fontSize: 12,
              cursor: scanning ? "default" : "pointer",
              opacity: scanning ? 0.5 : 1,
            }}
            title="Forza un nuovo scan ora (bootstrap iniziale o test)"
          >
            {scanning ? "Scanning…" : "🔄 Manual Scan"}
          </button>
        </div>

        {noData && !loading && (
          <div
            style={{
              padding: "16px 12px",
              background: "var(--surface-sunk)",
              borderRadius: 6,
              fontSize: 13,
              color: "var(--muted)",
            }}
          >
            Nessun dato ancora. Premi <strong>🔄 Manual Scan</strong> per avviare il primo daily
            scan e creare le prime decisioni AI.
          </div>
        )}

        {current && (
          <div className="grid grid-3" style={{ gap: 12 }}>
            <MetricCard
              label="NAV corrente"
              value={money(current.nav)}
              sub={`Peak ${money(current.peak_nav)} · ${current.date}`}
            />
            <MetricCard
              label="Return cumulativo"
              value={pct(current.cumulative_return_pct)}
              sub={`DD ${pct(current.drawdown_pct)}`}
              accent={
                current.cumulative_return_pct >= 0
                  ? "var(--reflation)"
                  : "var(--deflation)"
              }
            />
            <MetricCard
              label="Posizioni / Deployed"
              value={`${current.n_positions} / ${pct(current.deployed_pct, 1)}`}
              sub={
                current.top_position_asset
                  ? `Top: ${formatAsset(current.top_position_asset)} ${pct(current.top_position_weight_pct, 1)}`
                  : "—"
              }
            />
          </div>
        )}

        {/* Benchmark comparison */}
        {current && (
          <div className="grid grid-2" style={{ gap: 12, marginTop: 12 }}>
            <BenchmarkRow
              label="vs 60/40"
              modelReturn={current.cumulative_return_pct}
              benchReturn={current.benchmark_60_40_return_pct}
            />
            <BenchmarkRow
              label="vs S&P 500"
              modelReturn={current.cumulative_return_pct}
              benchReturn={current.benchmark_sp500_return_pct}
            />
          </div>
        )}
      </div>

      {/* Positions table */}
      <div className="card" style={{ marginTop: 16 }}>
        <h2 style={{ margin: 0, marginBottom: 8 }}>Posizioni attive</h2>
        <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 10 }}>
          DCA tranche · PnL stimato (drift score + ASSET_REGIME_DATA × probs)
        </div>
        {positions.length === 0 && (
          <div style={{ fontSize: 13, color: "var(--muted)", fontStyle: "italic" }}>
            Nessuna posizione aperta — l'AI attende segnali validi.
          </div>
        )}
        {positions.length > 0 && (
          <ScrollShadow innerClassName="table-wrap">
            <table className="table table-responsive">
              <thead>
                <tr>
                  <th>Asset</th>
                  <th className="num">Peso %</th>
                  <th className="num">Target %</th>
                  <th className="num">Tranches</th>
                  <th className="num">PnL est.</th>
                  <th>Regime entry</th>
                  <th className="num">Score entry</th>
                  <th className="num">Giorni</th>
                  <th>Ultimo</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr key={p.asset_class}>
                    <td>{formatAsset(p.asset_class)}</td>
                    <td className="num">{pct(p.current_weight_pct, 1)}</td>
                    <td className="num">{pct(p.target_weight_pct, 1)}</td>
                    <td className="num">
                      {p.tranches_filled}/{p.tranches_total}
                    </td>
                    <td
                      className="num"
                      style={{
                        color:
                          p.estimated_pnl_pct >= 0
                            ? "var(--reflation)"
                            : "var(--deflation)",
                      }}
                    >
                      {pct(p.estimated_pnl_pct, 1)}
                    </td>
                    <td>{p.entry_regime}</td>
                    <td className="num">{p.avg_entry_score.toFixed(1)}</td>
                    <td className="num">{p.days_held}</td>
                    <td style={{ fontSize: 11, color: "var(--muted)" }}>
                      {p.last_action}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollShadow>
        )}
      </div>

      {/* Decisions log */}
      <div className="card" style={{ marginTop: 16 }}>
        <h2 style={{ margin: 0, marginBottom: 8 }}>
          Log decisioni AI{" "}
          <span style={{ fontSize: 12, color: "var(--muted)", fontWeight: 500 }}>
            ultimi 30 giorni
          </span>
        </h2>
        {decisions.length === 0 ? (
          <div style={{ fontSize: 13, color: "var(--muted)", fontStyle: "italic" }}>
            Nessuna decisione ancora loggata.
          </div>
        ) : (
          <ScrollShadow innerClassName="table-wrap">
            <table className="table table-responsive">
              <thead>
                <tr>
                  <th>Data</th>
                  <th>Asset</th>
                  <th>Action</th>
                  <th className="num">Score</th>
                  <th className="num">Conf</th>
                  <th>Regime</th>
                  <th>Mom.</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {decisions.map((d, i) => (
                  <tr key={i}>
                    <td style={{ fontSize: 11, color: "var(--muted)" }}>{d.date}</td>
                    <td style={{ fontSize: 12 }}>{formatAsset(d.asset_class)}</td>
                    <td>
                      <span
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          padding: "2px 6px",
                          background: ACTION_COLORS[d.action] || "var(--muted)",
                          color: "white",
                          borderRadius: 4,
                          textTransform: "uppercase",
                        }}
                      >
                        {d.action.replace(/_/g, " ")}
                      </span>
                      {d.tranches_total > 0 && d.tranche_index > 0 && (
                        <span style={{ marginLeft: 6, fontSize: 10, color: "var(--muted)" }}>
                          {d.tranche_index}/{d.tranches_total}
                        </span>
                      )}
                    </td>
                    <td className="num">{d.score.toFixed(1)}</td>
                    <td className="num">{(d.confidence * 100).toFixed(0)}%</td>
                    <td style={{ fontSize: 11 }}>{d.regime}</td>
                    <td>
                      <span style={{ color: MOMENTUM_COLORS[d.momentum_signal] || "var(--muted)", fontSize: 10, fontWeight: 600 }}>
                        {d.momentum_signal}
                      </span>
                    </td>
                    <td style={{ fontSize: 11, color: "var(--muted)" }}>{d.reason_short}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollShadow>
        )}
      </div>

      {/* Equity curve simplified */}
      {perf && perf.history.length > 1 && (
        <div className="card" style={{ marginTop: 16 }}>
          <h2 style={{ margin: 0, marginBottom: 12 }}>Equity curve vs benchmarks</h2>
          <EquityChart history={perf.history} />
        </div>
      )}
    </div>
  );
}

function MetricCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
}) {
  return (
    <div
      style={{
        padding: "12px 14px",
        background: "var(--surface-sunk)",
        borderRadius: 8,
        borderTop: accent ? `2px solid ${accent}` : `1px solid var(--stroke)`,
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: 0.6,
          color: "var(--muted)",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 20, fontWeight: 700, color: accent || "var(--text)" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function BenchmarkRow({
  label,
  modelReturn,
  benchReturn,
}: {
  label: string;
  modelReturn: number;
  benchReturn: number | null;
}) {
  const alpha = benchReturn == null ? null : modelReturn - benchReturn;
  const color =
    alpha == null ? "var(--muted)" : alpha >= 0 ? "var(--reflation)" : "var(--deflation)";
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "10px 12px",
        background: "var(--surface-sunk)",
        borderRadius: 6,
        fontSize: 12,
      }}
    >
      <span style={{ color: "var(--muted)" }}>{label}</span>
      <span>
        <strong style={{ color }}>{alpha == null ? "—" : pct(alpha)}</strong>
        <span style={{ marginLeft: 8, fontSize: 11, color: "var(--muted)" }}>
          ({pct(benchReturn)})
        </span>
      </span>
    </div>
  );
}

function EquityChart({ history }: { history: AiPortfolioPerformanceResponse["history"] }) {
  // Mini SVG chart: 3 linee (portfolio, 60/40, SP500) cumulative %
  const W = 720;
  const H = 200;
  const PAD = 30;

  const allVals = history.flatMap((h) => [
    h.cumulative_return_pct,
    h.benchmark_60_40_return_pct ?? 0,
    h.benchmark_sp500_return_pct ?? 0,
  ]);
  const minV = Math.min(0, ...allVals);
  const maxV = Math.max(0.001, ...allVals);
  const range = maxV - minV || 0.001;

  const xScale = (i: number) => PAD + ((W - 2 * PAD) * i) / Math.max(history.length - 1, 1);
  const yScale = (v: number) => H - PAD - ((H - 2 * PAD) * (v - minV)) / range;

  const linePath = (key: "cumulative_return_pct" | "benchmark_60_40_return_pct" | "benchmark_sp500_return_pct") => {
    return history
      .map((h, i) => {
        const v = h[key];
        if (v == null) return null;
        return `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} style={{ display: "block" }}>
      {/* Zero line */}
      <line
        x1={PAD}
        y1={yScale(0)}
        x2={W - PAD}
        y2={yScale(0)}
        stroke="var(--stroke)"
        strokeDasharray="2 4"
      />
      <text x={W - PAD + 4} y={yScale(0) + 4} fontSize={10} fill="var(--muted)">
        0%
      </text>

      <path d={linePath("benchmark_sp500_return_pct")} fill="none" stroke="var(--stagflation)" strokeWidth={1.5} strokeDasharray="4 3" />
      <path d={linePath("benchmark_60_40_return_pct")} fill="none" stroke="var(--muted)" strokeWidth={1.5} strokeDasharray="2 3" />
      <path d={linePath("cumulative_return_pct")} fill="none" stroke="var(--reflation)" strokeWidth={2} />

      {/* Legend */}
      <g transform={`translate(${PAD}, 10)`} fontSize={11}>
        <rect width={12} height={3} y={0} fill="var(--reflation)" />
        <text x={18} y={6} fill="var(--text)">AI Portfolio</text>
        <rect width={12} height={3} y={14} fill="var(--muted)" />
        <text x={18} y={20} fill="var(--text)">60/40</text>
        <rect width={12} height={3} y={28} fill="var(--stagflation)" />
        <text x={18} y={34} fill="var(--text)">S&amp;P 500</text>
      </g>
    </svg>
  );
}
