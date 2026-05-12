import { useMemo, useState } from "react";
import type {
  CrisisHistory,
  CrisisHistoryEpisode,
  CrisisHistoryPoint,
  CrisisType,
} from "../types";

interface Props {
  data: CrisisHistory;
}

const CRISIS_COLORS: Record<CrisisType, string> = {
  no_crisis: "var(--muted)",
  deflation_crash: "var(--deflation)",
  stagflation_debasement: "var(--stagflation)",
  bubble_goldilocks: "var(--goldilocks)",
};

const CRISIS_LABELS: Record<CrisisType, string> = {
  no_crisis: "No crisis",
  deflation_crash: "Deflation crash",
  stagflation_debasement: "Stagflation / debasement",
  bubble_goldilocks: "Bubble / goldilocks",
};

// SVG coordinate system (responsive via width:100% + preserveAspectRatio)
const VB_W = 720;
const VB_H = 240;
const PAD_L = 40;
const PAD_R = 14;
const PAD_T = 16;
const PAD_B = 32;
const PLOT_W = VB_W - PAD_L - PAD_R;
const PLOT_H = VB_H - PAD_T - PAD_B;

function diffDays(a: string, b: string): number {
  const ms = new Date(b).getTime() - new Date(a).getTime();
  return Math.round(ms / 86400000);
}

function yearLabel(iso: string): string {
  return iso.slice(0, 4);
}

export function CrisisHistoryTimeline({ data }: Props) {
  const [hovered, setHovered] = useState<{ x: number; y: number; pt: CrisisHistoryPoint } | null>(
    null,
  );

  const chart = useMemo(() => {
    if (data.n_points < 2 || data.points.length < 2) return null;
    const sorted = [...data.points].sort((a, b) => (a.date < b.date ? -1 : 1));

    const firstDate = sorted[0].date;
    const lastDate = sorted[sorted.length - 1].date;
    const totalDays = Math.max(1, diffDays(firstDate, lastDate));

    const xFor = (date: string) =>
      PAD_L + (diffDays(firstDate, date) / totalDays) * PLOT_W;
    // Risk score normalizzato [0, 1] → y bottom→top
    const yFor = (score: number) =>
      PAD_T + (1 - Math.max(0, Math.min(1, score))) * PLOT_H;

    // Points come segmenti colorati per crisis_type
    const points = sorted.map((pt) => ({
      pt,
      x: xFor(pt.date),
      y: yFor(pt.risk_score),
      color: CRISIS_COLORS[pt.crisis_type],
    }));

    // Polyline lineare grigia tenue (continuità visiva)
    const linePath = points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

    // X-axis labels per anno
    const labels: { x: number; label: string }[] = [];
    const startYear = parseInt(yearLabel(firstDate), 10);
    const endYear = parseInt(yearLabel(lastDate), 10);
    const yearStride = Math.max(1, Math.ceil((endYear - startYear) / 8));
    for (let y = startYear; y <= endYear; y += yearStride) {
      const iso = `${y}-01-01`;
      if (iso < firstDate || iso > lastDate) continue;
      labels.push({ x: xFor(iso), label: String(y) });
    }

    // Episodes come fasce verticali sullo sfondo
    const epBands = data.notable_episodes.map((ep) => {
      const x1 = xFor(ep.start);
      const x2 = xFor(ep.end);
      return {
        x: Math.min(x1, x2),
        width: Math.max(2, Math.abs(x2 - x1)),
        color: CRISIS_COLORS[ep.crisis_type],
        ep,
      };
    });

    return { sorted, points, linePath, labels, epBands, totalDays };
  }, [data]);

  if (!chart) {
    return (
      <div className="card" style={{ marginTop: 16 }}>
        <h2>Crisis Risk Timeline</h2>
        <div style={{ fontSize: 13, color: "var(--muted)" }}>
          Dati insufficienti per il timeline ({data.n_points} punti).
        </div>
      </div>
    );
  }

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const xRatio = (e.clientX - rect.left) / rect.width;
    const yRatio = (e.clientY - rect.top) / rect.height;
    // Trova punto più vicino
    let best: typeof chart.points[number] | null = null;
    let bestDist = Infinity;
    for (const p of chart.points) {
      const xN = p.x / VB_W;
      const dist = Math.abs(xN - xRatio);
      if (dist < bestDist) {
        bestDist = dist;
        best = p;
      }
    }
    if (best && bestDist < 0.02) {
      setHovered({ x: e.clientX - rect.left, y: e.clientY - rect.top, pt: best.pt });
    } else {
      setHovered(null);
    }
    void yRatio;
  };

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          flexWrap: "wrap",
          gap: 12,
          marginBottom: 8,
        }}
      >
        <h2 style={{ margin: 0 }}>Crisis Risk Timeline (Backtest)</h2>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>
          {data.date_from} → {data.date_to} · {data.n_points} punti ·{" "}
          {data.notable_episodes.length} episodi
        </span>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          marginBottom: 10,
          fontSize: 12,
        }}
      >
        {(Object.keys(CRISIS_LABELS) as CrisisType[]).map((t) => (
          <span key={t} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                background: CRISIS_COLORS[t],
              }}
            />
            <span style={{ color: "var(--muted)" }}>{CRISIS_LABELS[t]}</span>
          </span>
        ))}
      </div>

      <div style={{ position: "relative" }}>
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          preserveAspectRatio="none"
          style={{ width: "100%", height: 240, display: "block" }}
          onMouseMove={onMove}
          onMouseLeave={() => setHovered(null)}
        >
          {/* Horizontal gridlines: 25/50/75/100% risk score */}
          {[0, 0.25, 0.5, 0.75, 1].map((p) => {
            const y = PAD_T + (1 - p) * PLOT_H;
            return (
              <g key={p}>
                <line
                  x1={PAD_L}
                  x2={VB_W - PAD_R}
                  y1={y}
                  y2={y}
                  stroke="var(--stroke)"
                  strokeWidth={1}
                  strokeDasharray={p === 0 || p === 1 ? "none" : "2 3"}
                />
                <text
                  x={PAD_L - 6}
                  y={y + 3}
                  fontSize={9}
                  fill="var(--muted)"
                  textAnchor="end"
                >
                  {Math.round(p * 100)}
                </text>
              </g>
            );
          })}

          {/* Threshold linee verticali: low/elevated/high/extreme */}
          {/* Risk score 0.3 = elevated, 0.6 = extreme (orientative) */}
          {[0.3, 0.6].map((thr) => {
            const y = PAD_T + (1 - thr) * PLOT_H;
            return (
              <line
                key={thr}
                x1={PAD_L}
                x2={VB_W - PAD_R}
                y1={y}
                y2={y}
                stroke={thr >= 0.6 ? "var(--deflation)" : "var(--goldilocks)"}
                strokeWidth={0.8}
                strokeDasharray="4 2"
                opacity={0.4}
              />
            );
          })}

          {/* Episode bands sullo sfondo */}
          {chart.epBands.map((b, i) => (
            <rect
              key={i}
              x={b.x}
              y={PAD_T}
              width={b.width}
              height={PLOT_H}
              fill={b.color}
              opacity={0.10}
            >
              <title>
                {b.ep.start} → {b.ep.end} · {CRISIS_LABELS[b.ep.crisis_type]} · peak{" "}
                {b.ep.peak_risk_score.toFixed(2)}
              </title>
            </rect>
          ))}

          {/* Polyline grigia tenue */}
          <polyline
            points={chart.linePath}
            fill="none"
            stroke="var(--muted)"
            strokeWidth={0.6}
            opacity={0.35}
          />

          {/* Punti colorati per crisis_type */}
          {chart.points.map((p, i) => (
            <circle
              key={i}
              cx={p.x}
              cy={p.y}
              r={p.pt.risk_level === "extreme" ? 3 : p.pt.risk_level === "high" ? 2.5 : 1.8}
              fill={p.color}
              opacity={
                p.pt.crisis_type === "no_crisis"
                  ? 0.35
                  : p.pt.risk_level === "low"
                  ? 0.45
                  : 0.85
              }
            />
          ))}

          {/* X-axis year labels */}
          {chart.labels.map((l, i) => (
            <text
              key={i}
              x={l.x}
              y={VB_H - PAD_B + 16}
              fontSize={10}
              fill="var(--muted)"
              textAnchor="middle"
            >
              {l.label}
            </text>
          ))}

          {/* Y-axis caption (risk score %) */}
          <text
            x={6}
            y={PAD_T + PLOT_H / 2}
            fontSize={9}
            fill="var(--muted)"
            textAnchor="middle"
            transform={`rotate(-90, 6, ${PAD_T + PLOT_H / 2})`}
          >
            risk score
          </text>
        </svg>

        {hovered && (
          <div
            style={{
              position: "absolute",
              left: Math.min(hovered.x + 10, 500),
              top: hovered.y - 50,
              background: "var(--surface-sunk)",
              border: "1px solid var(--stroke)",
              borderRadius: 6,
              padding: "6px 10px",
              fontSize: 11,
              pointerEvents: "none",
              boxShadow: "0 4px 12px rgba(0,0,0,0.18)",
              minWidth: 180,
              zIndex: 5,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 2 }}>{hovered.pt.date}</div>
            <div style={{ color: CRISIS_COLORS[hovered.pt.crisis_type], fontWeight: 600 }}>
              {CRISIS_LABELS[hovered.pt.crisis_type]}
            </div>
            <div style={{ color: "var(--muted)" }}>
              risk_level: <strong>{hovered.pt.risk_level}</strong>
            </div>
            <div style={{ color: "var(--muted)" }}>
              score: <strong>{(hovered.pt.risk_score * 100).toFixed(0)}%</strong>
            </div>
            <div style={{ color: "var(--muted)" }}>
              triggers: {hovered.pt.triggers_fired}/{hovered.pt.triggers_total}
            </div>
          </div>
        )}
      </div>

      {/* Notable episodes table */}
      {data.notable_episodes.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              color: "var(--muted)",
              textTransform: "uppercase",
              letterSpacing: 0.5,
              marginBottom: 6,
            }}
          >
            Episodi notabili ({data.notable_episodes.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {data.notable_episodes.map((ep, i) => (
              <EpisodeRow key={i} ep={ep} />
            ))}
          </div>
        </div>
      )}

      {/* Distribution stats */}
      <div
        style={{
          marginTop: 14,
          fontSize: 11,
          color: "var(--muted)",
          display: "flex",
          flexWrap: "wrap",
          gap: 14,
        }}
      >
        <span>
          <strong>Levels:</strong>{" "}
          {Object.entries(data.level_distribution)
            .map(([k, v]) => `${k} ${(v.pct * 100).toFixed(1)}%`)
            .join(" · ")}
        </span>
      </div>
    </div>
  );
}

function EpisodeRow({ ep }: { ep: CrisisHistoryEpisode }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "auto 1fr auto auto",
        alignItems: "center",
        gap: 10,
        padding: "6px 10px",
        background: "var(--surface-sunk)",
        borderRadius: 4,
        fontSize: 12,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: 2,
          background: CRISIS_COLORS[ep.crisis_type],
        }}
      />
      <span style={{ color: "var(--muted)" }}>
        {ep.start} → {ep.end}
      </span>
      <span style={{ fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>
        {CRISIS_LABELS[ep.crisis_type]}
      </span>
      <span style={{ fontSize: 11, color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
        peak {(ep.peak_risk_score * 100).toFixed(0)}% · {ep.duration_points} pt
      </span>
    </div>
  );
}
