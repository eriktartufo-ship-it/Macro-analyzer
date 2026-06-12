import { useEffect, useState } from "react";
import { api } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";
import type {
  AiPortfolioDecision,
  AiPortfolioLearning,
  AiPortfolioPerformanceResponse,
  AiPortfolioPosition,
  AiPortfolioReasoning,
} from "../types";

type StrategyType = "model_driven" | "data_driven";

const STRATEGY_META: Record<StrategyType, { label: string; emoji: string; subtitle: string; color: string }> = {
  model_driven: {
    label: "Model-Driven",
    emoji: "🧠",
    subtitle: "Classifier regime + score + momentum",
    color: "var(--reflation)",
  },
  data_driven: {
    label: "Data-Driven",
    emoji: "📊",
    subtitle: "Raw indicators FRED + ROC + soglie",
    color: "var(--goldilocks)",
  },
};

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
  const [activeStrategy, setActiveStrategy] = useState<StrategyType>("model_driven");
  const [leaderboard, setLeaderboard] = useState<Record<string, unknown> | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Per ogni strategia tengo lo stato
  const [posByStrat, setPosByStrat] = useState<Record<StrategyType, AiPortfolioPosition[]>>({
    model_driven: [],
    data_driven: [],
  });
  const [decByStrat, setDecByStrat] = useState<Record<StrategyType, AiPortfolioDecision[]>>({
    model_driven: [],
    data_driven: [],
  });
  const [perfByStrat, setPerfByStrat] = useState<Record<StrategyType, AiPortfolioPerformanceResponse | null>>({
    model_driven: null,
    data_driven: null,
  });
  const [reasoningByStrat, setReasoningByStrat] = useState<Record<StrategyType, AiPortfolioReasoning | null>>({
    model_driven: null,
    data_driven: null,
  });
  const [learnByStrat, setLearnByStrat] = useState<Record<StrategyType, AiPortfolioLearning[]>>({
    model_driven: [],
    data_driven: [],
  });

  const loadFor = async (strategy: StrategyType) => {
    try {
      const [pos, dec, perf, learns] = await Promise.all([
        api.aiPortfolioPositions(strategy),
        api.aiPortfolioDecisions(30, strategy),
        api.aiPortfolioPerformance(180, strategy),
        api.aiPortfolioLearnings(20, strategy).catch(() => [] as AiPortfolioLearning[]),
      ]);
      setPosByStrat((s) => ({ ...s, [strategy]: pos }));
      setDecByStrat((s) => ({ ...s, [strategy]: dec }));
      setPerfByStrat((s) => ({ ...s, [strategy]: perf }));
      setLearnByStrat((s) => ({ ...s, [strategy]: learns }));
      try {
        const r = await api.aiPortfolioReasoning(false, strategy);
        setReasoningByStrat((s) => ({ ...s, [strategy]: r }));
      } catch {
        setReasoningByStrat((s) => ({ ...s, [strategy]: null }));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Errore caricamento");
    }
  };

  const loadAll = async () => {
    setError(null);
    await Promise.all([loadFor("model_driven"), loadFor("data_driven")]);
    try {
      const lb = await api.aiPortfolioLeaderboard();
      setLeaderboard(lb);
    } catch {
      setLeaderboard(null);
    }
  };

  const manualScan = async () => {
    setScanning(true);
    try {
      await api.aiPortfolioManualScan();
      await loadAll();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Scan fallito");
    } finally {
      setScanning(false);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  const positions = posByStrat[activeStrategy];
  const decisions = decByStrat[activeStrategy];
  const reasoning = reasoningByStrat[activeStrategy];
  const learnings = learnByStrat[activeStrategy];
  const meta = STRATEGY_META[activeStrategy];

  const lbHead = (leaderboard as any)?.head_to_head as
    | { winner: string; spread_pct: number; spread_text: string }
    | null;
  const mdCurrent = perfByStrat.model_driven?.current;
  const ddCurrent = perfByStrat.data_driven?.current;

  return (
    <div>
      {error && <div className="error">{error}</div>}

      {/* Header + scan button */}
      <div className="card" style={{ marginTop: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, marginBottom: 8 }}>
          <div>
            <h2 style={{ margin: 0 }}>🏇 AI Portfolio Horserace</h2>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
              2 strategie competono, $100k capitale separato ciascuna. Stesso universe 24 asset.
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
          >
            {scanning ? "Scanning…" : "🔄 Scan entrambe"}
          </button>
        </div>

        {/* Leaderboard head-to-head */}
        {lbHead && mdCurrent && ddCurrent && (
          <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
            <LeaderCard
              strategy="model_driven"
              current={mdCurrent}
              isWinner={lbHead.winner === "model_driven"}
            />
            <div style={{ display: "flex", alignItems: "center", padding: "0 8px" }}>
              <div style={{ fontSize: 11, color: "var(--muted)", textAlign: "center" }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text)" }}>
                  {lbHead.winner === "tie" ? "=" : "→"}
                </div>
                <div>Spread {lbHead.spread_text}</div>
                <div style={{ fontSize: 9 }}>{lbHead.winner === "tie" ? "tie" : `${lbHead.winner.replace("_", " ")} avanti`}</div>
              </div>
            </div>
            <LeaderCard
              strategy="data_driven"
              current={ddCurrent}
              isWinner={lbHead.winner === "data_driven"}
            />
          </div>
        )}

        {/* Equity chart 4-line */}
        {perfByStrat.model_driven && perfByStrat.data_driven && (
          <div style={{ marginTop: 16 }}>
            <EquityChart4
              model={perfByStrat.model_driven.history}
              data={perfByStrat.data_driven.history}
            />
          </div>
        )}
      </div>

      {/* Strategy selector tabs */}
      <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
        {(["model_driven", "data_driven"] as StrategyType[]).map((s) => {
          const m = STRATEGY_META[s];
          const active = activeStrategy === s;
          return (
            <button
              key={s}
              onClick={() => setActiveStrategy(s)}
              style={{
                flex: 1,
                background: active ? m.color : "var(--surface-sunk)",
                color: active ? "white" : "var(--text)",
                border: `1px solid ${active ? m.color : "var(--stroke)"}`,
                borderRadius: 8,
                padding: "10px 12px",
                cursor: "pointer",
                fontSize: 13,
                fontWeight: 600,
                textAlign: "left",
              }}
            >
              {m.emoji} {m.label}
              <div style={{ fontSize: 10, fontWeight: 400, opacity: 0.8, marginTop: 2 }}>{m.subtitle}</div>
            </button>
          );
        })}
      </div>

      {/* AI Reasoning per strategia selezionata */}
      {reasoning && (
        <div className="card" style={{ marginTop: 12, borderLeft: `3px solid ${meta.color}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
            <h2 style={{ margin: 0 }}>
              🧠 Analisi AI · {meta.label} · {reasoning.date}
            </h2>
            <RegimeChallengeBadge agree={reasoning.ai_agrees} alt={reasoning.ai_alternative_regime} />
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.55, color: "var(--text)", padding: "10px 12px", background: "var(--surface-sunk)", borderLeft: "3px solid var(--accent)", borderRadius: 4, marginBottom: 10 }}>
            {reasoning.daily_summary}
          </div>
          {reasoning.ai_agrees !== "high" && reasoning.regime_challenge_reasoning && (
            <div style={{ padding: "12px 14px", background: "rgba(231,76,60,0.08)", borderLeft: "3px solid var(--deflation)", borderRadius: 4, marginBottom: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.6, color: "var(--deflation)", marginBottom: 6 }}>
                ⚠ L'AI mette in discussione il regime
              </div>
              <div style={{ fontSize: 12, lineHeight: 1.5, color: "var(--text)" }}>
                Classifier dice <strong>{reasoning.regime_classifier_says}</strong>
                {reasoning.ai_alternative_regime && (
                  <> — AI suggerisce <strong style={{ color: "var(--deflation)" }}>{reasoning.ai_alternative_regime}</strong></>
                )}
              </div>
              <div style={{ fontSize: 12, lineHeight: 1.5, color: "var(--muted)", marginTop: 6 }}>
                {reasoning.regime_challenge_reasoning}
              </div>
            </div>
          )}
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Provider: {reasoning.provider}</div>
        </div>
      )}

      {/* Positions table */}
      <div className="card" style={{ marginTop: 12 }}>
        <h2 style={{ margin: 0, marginBottom: 8 }}>
          Posizioni attive · <span style={{ color: meta.color }}>{meta.label}</span>
        </h2>
        {positions.length === 0 ? (
          <div style={{ fontSize: 13, color: "var(--muted)", fontStyle: "italic" }}>
            Nessuna posizione aperta — l'AI attende segnali validi.
          </div>
        ) : (
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
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr key={p.asset_class}>
                    <td>{formatAsset(p.asset_class)}</td>
                    <td className="num">{pct(p.current_weight_pct, 1)}</td>
                    <td className="num">{pct(p.target_weight_pct, 1)}</td>
                    <td className="num">{p.tranches_filled}/{p.tranches_total}</td>
                    <td className="num" style={{ color: p.estimated_pnl_pct >= 0 ? "var(--reflation)" : "var(--deflation)" }}>
                      {pct(p.estimated_pnl_pct, 1)}
                    </td>
                    <td>{p.entry_regime}</td>
                    <td className="num">{p.avg_entry_score.toFixed(1)}</td>
                    <td className="num">{p.days_held}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollShadow>
        )}
      </div>

      {/* Decisions log */}
      <div className="card" style={{ marginTop: 12 }}>
        <h2 style={{ margin: 0, marginBottom: 8 }}>
          Log decisioni · <span style={{ color: meta.color }}>{meta.label}</span>
        </h2>
        {decisions.length === 0 ? (
          <div style={{ fontSize: 13, color: "var(--muted)", fontStyle: "italic" }}>Nessuna decisione ancora.</div>
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
                  <th>Mom.</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {decisions.slice(0, 100).map((d, i) => (
                  <tr key={i}>
                    <td style={{ fontSize: 11, color: "var(--muted)" }}>{d.date}</td>
                    <td style={{ fontSize: 12 }}>{formatAsset(d.asset_class)}</td>
                    <td>
                      <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 6px", background: ACTION_COLORS[d.action] || "var(--muted)", color: "white", borderRadius: 4, textTransform: "uppercase" }}>
                        {d.action.replace(/_/g, " ")}
                      </span>
                    </td>
                    <td className="num">{d.score.toFixed(1)}</td>
                    <td className="num">{(d.confidence * 100).toFixed(0)}%</td>
                    <td>
                      <span style={{ color: MOMENTUM_COLORS[d.momentum_signal] || "var(--muted)", fontSize: 10, fontWeight: 600 }}>
                        {d.momentum_signal}
                      </span>
                    </td>
                    <td style={{ fontSize: 11, color: "var(--muted)" }}>
                      {d.reason_short}
                      {reasoning?.decisions_reasoning?.[d.asset_class] && (
                        <div style={{ marginTop: 4, fontSize: 11, color: "var(--accent)", fontStyle: "italic" }}>
                          🧠 {reasoning.decisions_reasoning[d.asset_class]}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollShadow>
        )}
      </div>

      {learnings.length > 0 && (
        <div className="card" style={{ marginTop: 12 }}>
          <h2 style={{ margin: 0, marginBottom: 8 }}>
            📚 Learnings post-mortem · <span style={{ color: meta.color }}>{meta.label}</span>
          </h2>
          <ScrollShadow innerClassName="table-wrap">
            <table className="table table-responsive">
              <thead>
                <tr>
                  <th>Pattern</th>
                  <th className="num">W/L</th>
                  <th className="num">Win rate</th>
                  <th className="num">Avg PnL</th>
                  <th className="num">Threshold shift</th>
                </tr>
              </thead>
              <tbody>
                {learnings.map((L) => (
                  <tr key={L.pattern_key}>
                    <td style={{ fontSize: 11 }}>{L.pattern_key}</td>
                    <td className="num">
                      <span style={{ color: "var(--reflation)" }}>{L.n_wins}</span> / <span style={{ color: "var(--deflation)" }}>{L.n_losses}</span>
                    </td>
                    <td className="num">{(L.win_rate * 100).toFixed(0)}%</td>
                    <td className="num" style={{ color: L.avg_pnl_pct >= 0 ? "var(--reflation)" : "var(--deflation)" }}>
                      {(L.avg_pnl_pct * 100).toFixed(1)}%
                    </td>
                    <td className="num" style={{ color: L.entry_threshold_shift > 0 ? "var(--deflation)" : L.entry_threshold_shift < 0 ? "var(--reflation)" : "var(--muted)", fontWeight: 600 }}>
                      {L.entry_threshold_shift > 0 ? "+" : ""}{L.entry_threshold_shift.toFixed(1)}
                    </td>
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

function LeaderCard({
  strategy,
  current,
  isWinner,
}: {
  strategy: StrategyType;
  current: any;
  isWinner: boolean;
}) {
  const meta = STRATEGY_META[strategy];
  return (
    <div
      style={{
        flex: 1,
        padding: "12px 14px",
        background: isWinner ? "rgba(46,204,113,0.10)" : "var(--surface-sunk)",
        border: `1px solid ${isWinner ? "var(--reflation)" : "var(--stroke)"}`,
        borderRadius: 8,
        position: "relative",
      }}
    >
      {isWinner && (
        <span style={{ position: "absolute", top: 8, right: 10, fontSize: 16 }}>🏆</span>
      )}
      <div style={{ fontSize: 11, fontWeight: 700, color: meta.color, textTransform: "uppercase", letterSpacing: 0.6 }}>
        {meta.emoji} {meta.label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, marginTop: 4 }}>{money(current.nav)}</div>
      <div style={{ fontSize: 12, color: current.cumulative_return_pct >= 0 ? "var(--reflation)" : "var(--deflation)" }}>
        {pct(current.cumulative_return_pct)} cum
      </div>
      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
        {current.n_positions} posizioni · DD {pct(current.drawdown_pct, 1)}
      </div>
    </div>
  );
}

function RegimeChallengeBadge({ agree, alt }: { agree: string; alt: string | null }) {
  const colors: Record<string, { bg: string; fg: string; label: string }> = {
    high: { bg: "rgba(46,204,113,0.15)", fg: "var(--reflation)", label: "AI concorda" },
    medium: { bg: "rgba(241,196,15,0.15)", fg: "var(--goldilocks)", label: "AI perplessa" },
    low: { bg: "rgba(231,76,60,0.15)", fg: "var(--deflation)", label: "AI dissente" },
  };
  const c = colors[agree] || colors.high;
  return (
    <span
      style={{
        fontSize: 10,
        fontWeight: 700,
        padding: "3px 8px",
        background: c.bg,
        color: c.fg,
        borderRadius: 4,
        textTransform: "uppercase",
        letterSpacing: 0.5,
      }}
      title={alt ? `Regime alternativo proposto: ${alt}` : undefined}
    >
      {c.label}
      {agree !== "high" && alt && ` → ${alt}`}
    </span>
  );
}

function EquityChart4({
  model,
  data,
}: {
  model: AiPortfolioPerformanceResponse["history"];
  data: AiPortfolioPerformanceResponse["history"];
}) {
  const W = 720;
  const H = 220;
  const PAD = 30;

  // Use the longer history for x axis
  const len = Math.max(model.length, data.length);
  const allVals = [
    ...model.flatMap((h) => [h.cumulative_return_pct, h.benchmark_60_40_return_pct ?? 0, h.benchmark_sp500_return_pct ?? 0]),
    ...data.flatMap((h) => [h.cumulative_return_pct]),
  ];
  const minV = Math.min(0, ...allVals);
  const maxV = Math.max(0.001, ...allVals);
  const range = maxV - minV || 0.001;

  const xScale = (i: number) => PAD + ((W - 2 * PAD) * i) / Math.max(len - 1, 1);
  const yScale = (v: number) => H - PAD - ((H - 2 * PAD) * (v - minV)) / range;

  const linePath = (
    series: AiPortfolioPerformanceResponse["history"],
    key: "cumulative_return_pct" | "benchmark_60_40_return_pct" | "benchmark_sp500_return_pct",
  ) =>
    series
      .map((h, i) => {
        const v = h[key];
        if (v == null) return null;
        return `${i === 0 ? "M" : "L"}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} style={{ display: "block" }}>
      <line x1={PAD} y1={yScale(0)} x2={W - PAD} y2={yScale(0)} stroke="var(--stroke)" strokeDasharray="2 4" />
      <text x={W - PAD + 4} y={yScale(0) + 4} fontSize={10} fill="var(--muted)">0%</text>

      <path d={linePath(model, "benchmark_sp500_return_pct")} fill="none" stroke="var(--stagflation)" strokeWidth={1.5} strokeDasharray="4 3" />
      <path d={linePath(model, "benchmark_60_40_return_pct")} fill="none" stroke="var(--muted)" strokeWidth={1.5} strokeDasharray="2 3" />
      <path d={linePath(model, "cumulative_return_pct")} fill="none" stroke="var(--reflation)" strokeWidth={2.5} />
      <path d={linePath(data, "cumulative_return_pct")} fill="none" stroke="var(--goldilocks)" strokeWidth={2.5} />

      <g transform={`translate(${PAD}, 10)`} fontSize={11}>
        <rect width={12} height={3} y={0} fill="var(--reflation)" />
        <text x={18} y={6} fill="var(--text)">🧠 Model</text>
        <rect width={12} height={3} y={14} fill="var(--goldilocks)" />
        <text x={18} y={20} fill="var(--text)">📊 Data</text>
        <rect width={12} height={3} y={28} fill="var(--muted)" />
        <text x={18} y={34} fill="var(--text)">60/40</text>
        <rect width={12} height={3} y={42} fill="var(--stagflation)" />
        <text x={18} y={48} fill="var(--text)">S&amp;P 500</text>
      </g>
    </svg>
  );
}
