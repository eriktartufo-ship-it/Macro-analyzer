import { useEffect, useState } from "react";
import { api } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";
import type { FOMCAnalysis, FOMCReport } from "../types";

const REGIME_COLOR: Record<string, string> = {
  reflation: "#10b981",
  stagflation: "#d97706",
  deflation: "#ef4444",
  goldilocks: "#06b6d4",
};

function scoreColor(score: number): string {
  // hawkish = rosso (alza tassi, restrittivo); dovish = verde (taglia tassi, espansivo)
  if (score > 0.30) return "var(--deflation, #ef4444)";
  if (score > 0.10) return "#f97316";
  if (score < -0.30) return "var(--reflation, #10b981)";
  if (score < -0.10) return "#84cc16";
  return "var(--muted)";
}

function scoreLabel(score: number): string {
  if (score > 0.50) return "VERY HAWKISH";
  if (score > 0.20) return "HAWKISH";
  if (score > 0.05) return "leaning hawkish";
  if (score < -0.50) return "VERY DOVISH";
  if (score < -0.20) return "DOVISH";
  if (score < -0.05) return "leaning dovish";
  return "neutral";
}

function trendBadge(trend: string): { label: string; color: string } {
  switch (trend) {
    case "hawkening":
      return { label: "↑ HAWKENING", color: "var(--deflation, #ef4444)" };
    case "dovening":
      return { label: "↓ DOVENING", color: "var(--reflation, #10b981)" };
    case "stable":
      return { label: "= STABLE", color: "var(--muted)" };
    default:
      return { label: "n/a", color: "var(--muted)" };
  }
}

function ScoreBar({ score }: { score: number }) {
  // -1..+1 mappato a 0..100% di larghezza, centro a 50%
  const pct = (score + 1) * 50;
  return (
    <div style={{
      position: "relative", height: 8, width: 200,
      background: "var(--bg)", borderRadius: 4, border: "1px solid var(--border)",
    }}>
      {/* Linea di centro */}
      <div style={{
        position: "absolute", left: "50%", top: -2, bottom: -2,
        width: 1, background: "var(--text)", opacity: 0.4,
      }} />
      {/* Pallino score */}
      <div style={{
        position: "absolute", left: `${pct}%`, top: "50%",
        width: 12, height: 12, marginLeft: -6, marginTop: -6,
        borderRadius: "50%", background: scoreColor(score),
        border: "2px solid var(--bg)",
      }} />
    </div>
  );
}

function SentimentTimeline({ analyses }: { analyses: FOMCAnalysis[] }) {
  if (analyses.length === 0) return null;
  // Ordina ascendente per data per il chart
  const sorted = [...analyses].sort((a, b) => a.published_date.localeCompare(b.published_date));
  const W = 720;
  const H = 180;
  const PAD = { top: 16, right: 100, bottom: 30, left: 50 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const xScale = (i: number) =>
    PAD.left + (sorted.length === 1 ? innerW / 2 : (i / (sorted.length - 1)) * innerW);
  const yScale = (v: number) => PAD.top + (1 - (v + 1) / 2) * innerH;
  const yZero = yScale(0);

  return (
    <svg width={W} height={H} style={{ background: "var(--bg)", borderRadius: 6 }}>
      {/* Y grid -1, -0.5, 0, +0.5, +1 */}
      {[-1, -0.5, 0, 0.5, 1].map((v) => {
        const y = yScale(v);
        return (
          <g key={v}>
            <line x1={PAD.left} y1={y} x2={PAD.left + innerW} y2={y}
              stroke="var(--border)" strokeDasharray="2 2" opacity={v === 0 ? 0.6 : 0.3} />
            <text x={PAD.left - 6} y={y + 3} fontSize="10" fill="var(--muted)" textAnchor="end">
              {v > 0 ? `+${v}` : v}
            </text>
          </g>
        );
      })}
      {/* Dovish/hawkish labels */}
      <text x={PAD.left - 6} y={PAD.top + 2} fontSize="9" fill="var(--deflation, #ef4444)" textAnchor="end">
        hawk
      </text>
      <text x={PAD.left - 6} y={PAD.top + innerH + 8} fontSize="9" fill="var(--reflation, #10b981)" textAnchor="end">
        dove
      </text>

      {/* Linea zero piu' marcata */}
      <line x1={PAD.left} y1={yZero} x2={PAD.left + innerW} y2={yZero}
        stroke="var(--text)" strokeWidth={0.6} opacity={0.5} />

      {/* Linea principale */}
      <path
        d={sorted.map((a, i) => `${i === 0 ? "M" : "L"}${xScale(i)},${yScale(a.hawkish_dovish_score)}`).join(" ")}
        fill="none" stroke="var(--accent)" strokeWidth={1.8}
      />

      {/* Punti */}
      {sorted.map((a, i) => (
        <circle key={a.url} cx={xScale(i)} cy={yScale(a.hawkish_dovish_score)} r={5}
          fill={scoreColor(a.hawkish_dovish_score)} stroke="var(--bg)" strokeWidth={1.5}>
          <title>{a.published_date} {a.doc_type}: {a.hawkish_dovish_score.toFixed(2)}</title>
        </circle>
      ))}

      {/* X labels: solo prima/ultima/mediana */}
      {sorted.map((a, i) => {
        const showLabel = i === 0 || i === sorted.length - 1 ||
          (sorted.length >= 5 && i === Math.floor(sorted.length / 2));
        if (!showLabel) return null;
        return (
          <text key={`lbl-${i}`} x={xScale(i)} y={H - 10} fontSize="9" fill="var(--muted)" textAnchor="middle">
            {a.published_date.slice(0, 7)}
          </text>
        );
      })}
    </svg>
  );
}

export function FOMCSentimentPanel() {
  const [data, setData] = useState<FOMCReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  const load = async (forceRefresh = false) => {
    if (forceRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const r = await api.fomcReport(6, forceRefresh);
      setData(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Errore FOMC");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    load(false);
  }, []);

  if (error)
    return (
      <div className="card">
        <h2>FOMC sentiment (LLM-driven hawkish/dovish)</h2>
        <div className="error" style={{ fontSize: 13, marginBottom: 8 }}>{error}</div>
        <button onClick={() => load(false)} style={btnStyle}>Riprova</button>
      </div>
    );

  if (!data && loading)
    return (
      <div className="card">
        <h2>FOMC sentiment (LLM-driven hawkish/dovish)</h2>
        <div className="loading">Fetching FOMC documents…</div>
      </div>
    );

  if (!data) return null;

  const trend = trendBadge(data.trend);

  return (
    <div className="card">
      <h2>FOMC sentiment (LLM-driven hawkish/dovish)</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Statements e minutes Federal Reserve analizzati da LLM (Claude prio, Groq fallback)
        per estrarre tono <strong>hawkish</strong> (alza tassi → +) vs <strong>dovish</strong>
        (taglia tassi → −), key topics e regime implication. Cache aggressiva: stessi
        documenti non riprocessati.
      </div>

      <div style={{
        display: "flex", gap: 18, marginBottom: 14, padding: 10,
        background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 6, flexWrap: "wrap",
      }}>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Latest score</div>
          <div style={{ fontSize: 18, fontWeight: 700,
            color: data.latest_score !== null ? scoreColor(data.latest_score) : "var(--muted)" }}>
            {data.latest_score !== null
              ? `${data.latest_score >= 0 ? "+" : ""}${data.latest_score.toFixed(2)}`
              : "—"}
          </div>
          <div style={{ fontSize: 11, color: "var(--muted)" }}>
            {data.latest_score !== null ? scoreLabel(data.latest_score) : "no data"}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Avg ultimi 3</div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>
            {data.avg_score_3last !== null
              ? `${data.avg_score_3last >= 0 ? "+" : ""}${data.avg_score_3last.toFixed(2)}`
              : "—"}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Trend</div>
          <div style={{ fontSize: 14, fontWeight: 700, color: trend.color }}>{trend.label}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: "var(--muted)" }}>Documenti</div>
          <div style={{ fontSize: 14, fontWeight: 600 }}>{data.n_documents}</div>
        </div>
        <div style={{ marginLeft: "auto", alignSelf: "center" }}>
          <button onClick={() => load(true)} disabled={refreshing} style={btnStyle}>
            {refreshing ? "Re-analyzing…" : "Force re-analyze"}
          </button>
        </div>
      </div>

      {data.analyses.length > 0 && (
        <div style={{ overflowX: "auto", marginBottom: 14 }}>
          <SentimentTimeline analyses={data.analyses} />
        </div>
      )}

      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
        Documenti analizzati (clic per espandere)
      </div>
      <ScrollShadow innerClassName="table-wrap">
        <table className="table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>Date</th>
              <th>Type</th>
              <th>Score</th>
              <th>Bar</th>
              <th>Confidence</th>
              <th>Topics</th>
            </tr>
          </thead>
          <tbody>
            {data.analyses.map((a) => (
              <>
                <tr key={a.url}
                  style={{ cursor: "pointer", opacity: a.confidence < 0.2 ? 0.5 : 1 }}
                  onClick={() => setExpanded(expanded === a.url ? null : a.url)}>
                  <td>{a.published_date}</td>
                  <td>
                    <span style={{
                      padding: "2px 6px", borderRadius: 4,
                      background: a.doc_type === "statement" ? "rgba(56,189,248,0.15)" : "rgba(217,119,6,0.15)",
                      fontSize: 10, fontWeight: 600,
                    }}>
                      {a.doc_type}
                    </span>
                  </td>
                  <td className="num" style={{
                    color: scoreColor(a.hawkish_dovish_score), fontWeight: 700,
                  }}>
                    {a.hawkish_dovish_score >= 0 ? "+" : ""}{a.hawkish_dovish_score.toFixed(2)}
                    <div style={{ fontSize: 9, color: "var(--muted)" }}>
                      {scoreLabel(a.hawkish_dovish_score)}
                    </div>
                  </td>
                  <td><ScoreBar score={a.hawkish_dovish_score} /></td>
                  <td className="num">{(a.confidence * 100).toFixed(0)}%</td>
                  <td style={{ fontSize: 10, color: "var(--muted)" }}>
                    {a.key_topics.slice(0, 3).join(" · ")}
                    {a.key_topics.length > 3 && ` +${a.key_topics.length - 3}`}
                  </td>
                </tr>
                {expanded === a.url && (
                  <tr key={`${a.url}-exp`}>
                    <td colSpan={6} style={{ background: "var(--bg)", padding: 12 }}>
                      <div style={{ marginBottom: 8 }}>
                        <strong>Forward guidance:</strong>{" "}
                        <span style={{ color: "var(--muted)" }}>{a.forward_guidance}</span>
                      </div>
                      <div style={{ marginBottom: 8 }}>
                        <strong>Summary:</strong>{" "}
                        <span style={{ color: "var(--text)" }}>{a.summary}</span>
                      </div>
                      <div style={{ marginBottom: 8 }}>
                        <strong>Topics:</strong>{" "}
                        {a.key_topics.map((t) => (
                          <span key={t} style={{
                            display: "inline-block", marginRight: 6, padding: "2px 6px",
                            background: "var(--bg)", border: "1px solid var(--border)",
                            borderRadius: 10, fontSize: 10,
                          }}>{t}</span>
                        ))}
                      </div>
                      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 6 }}>
                        <strong style={{ alignSelf: "center" }}>Regime nudge:</strong>
                        {Object.entries(a.regime_implication).map(([r, v]) => (
                          <div key={r} style={{
                            padding: "3px 8px", borderRadius: 4,
                            background: REGIME_COLOR[r] + "22",
                            color: REGIME_COLOR[r],
                            fontSize: 11, fontWeight: 600,
                          }}>
                            {r}: {v >= 0 ? "+" : ""}{v.toFixed(2)}
                          </div>
                        ))}
                      </div>
                      <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 8 }}>
                        Provider: {a.provider} · analyzed {a.analyzed_at.slice(0, 16)} ·{" "}
                        <a href={a.url} target="_blank" rel="noopener noreferrer"
                          style={{ color: "var(--accent)" }}>
                          Source ↗
                        </a>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </ScrollShadow>

      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 12 }}>
        <strong>Lettura.</strong> Score &gt; +0.5 = molto hawkish (Fed segnala restrizione,
        bond long sotto pressione, USD forte). Score &lt; -0.5 = molto dovish (taglio tassi
        prossimo, equity supportate, gold tipicamente bene). Trend{" "}
        <em>hawkening</em> = irrigidimento progressivo. <em>Confidence</em> bassa = testo
        ambiguo o misto, da pesare meno. Il <em>regime nudge</em> indica come il tono FOMC
        spinge le probabilita' regime: hawkish + sticky inflation → +stagflation/-goldilocks.
      </div>
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: "6px 14px",
  background: "var(--accent)",
  color: "white",
  border: "none",
  borderRadius: 6,
  cursor: "pointer",
};
