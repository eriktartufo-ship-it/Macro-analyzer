import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { SmoothableIndicator, SmoothedIndicator } from "../types";

function pct(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}

function SmoothChart({ data }: { data: SmoothedIndicator }) {
  const W = 720;
  const H = 260;
  const PAD = { top: 16, right: 90, bottom: 28, left: 50 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const pts = data.points;
  if (pts.length === 0) return null;

  const allValues = pts.flatMap((p) => [p.raw, p.filtered, p.smoothed]);
  const minY = Math.min(...allValues);
  const maxY = Math.max(...allValues);
  const yRange = maxY - minY || 1;
  const startTs = new Date(pts[0].date).getTime();
  const endTs = new Date(pts[pts.length - 1].date).getTime();
  const xRange = endTs - startTs || 1;

  const xScale = (date: string) =>
    PAD.left + ((new Date(date).getTime() - startTs) / xRange) * innerW;
  const yScale = (v: number) => PAD.top + (1 - (v - minY) / yRange) * innerH;

  const buildPath = (key: "raw" | "filtered" | "smoothed") =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"}${xScale(p.date)},${yScale(p[key])}`).join(" ");

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
              {v.toFixed(1)}
            </text>
          </g>
        );
      })}
      {[0, 0.25, 0.5, 0.75, 1].map((p) => {
        const t = startTs + p * xRange;
        const x = PAD.left + p * innerW;
        return (
          <text key={p} x={x} y={H - 8} fontSize="10" fill="var(--muted)" textAnchor="middle">
            {new Date(t).toLocaleDateString("en-US", { year: "numeric", month: "short" })}
          </text>
        );
      })}

      {/* Raw: thin grey */}
      <path d={buildPath("raw")} fill="none" stroke="var(--muted)" strokeWidth={1} opacity={0.6} />
      {/* Filtered: solid accent */}
      <path d={buildPath("filtered")} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
      {/* Smoothed: dashed reflation green */}
      <path d={buildPath("smoothed")} fill="none" stroke="var(--reflation, #10b981)"
        strokeWidth={1.6} strokeDasharray="4 2" />

      {/* Legend */}
      <g transform={`translate(${PAD.left + innerW + 10}, ${PAD.top})`}>
        <line x1={0} y1={3} x2={14} y2={3} stroke="var(--muted)" strokeWidth={1} opacity={0.6} />
        <text x={18} y={6} fontSize="10" fill="var(--text)">raw</text>
        <line x1={0} y1={20} x2={14} y2={20} stroke="var(--accent)" strokeWidth={1.6} />
        <text x={18} y={23} fontSize="10" fill="var(--text)">filtered</text>
        <line x1={0} y1={37} x2={14} y2={37} stroke="var(--reflation, #10b981)"
          strokeWidth={1.6} strokeDasharray="4 2" />
        <text x={18} y={40} fontSize="10" fill="var(--text)">smoothed</text>
      </g>
    </svg>
  );
}

export function SmoothedIndicatorsPanel() {
  const [indicators, setIndicators] = useState<SmoothableIndicator[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [data, setData] = useState<SmoothedIndicator | null>(null);
  const [lambda, setLambda] = useState(10.0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .smoothedIndicatorsList()
      .then((list) => {
        setIndicators(list);
        if (list.length > 0 && selected === null) setSelected(list[0].key);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Errore lista"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    api
      .smoothedIndicator(selected, lambda)
      .then((d) => setData(d))
      .catch((e) => setError(e instanceof Error ? e.message : "Errore Kalman"))
      .finally(() => setLoading(false));
  }, [selected, lambda]);

  return (
    <div className="card">
      <h2>Indicatori smoothed (Kalman 1D filter)</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        State-space filter <code>x_t = x_(t-1) + w_t</code>, <code>y_t = x_t + v_t</code> con
        <code> lambda = R/Q</code> tunabile. Riduce rumore senza perdere segnale (es. il bug
        <code> lei_roc=30%</code> tipico). <strong>Filtered</strong> = real-time causale,
        <strong> smoothed</strong> = retrospettivo (RTS).
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
        <label style={{ fontSize: 12 }}>
          Indicatore:{" "}
          <select
            value={selected ?? ""}
            onChange={(e) => setSelected(e.target.value)}
            style={{ padding: "4px 8px", fontSize: 12 }}
          >
            {indicators?.map((ind) => (
              <option key={ind.key} value={ind.key} title={ind.description}>
                {ind.key}
              </option>
            ))}
          </select>
        </label>
        <label style={{ fontSize: 12 }}>
          Lambda (R/Q):{" "}
          <input
            type="number" min={0.5} max={200} step={0.5} value={lambda}
            onChange={(e) => setLambda(Number(e.target.value))}
            style={{ width: 60 }}
          />
        </label>
        <span style={{ fontSize: 11, color: "var(--muted)", alignSelf: "center" }}>
          alto = smoothing aggressivo · basso = segue raw
        </span>
      </div>

      {error && <div className="error" style={{ fontSize: 13 }}>{error}</div>}

      {data && !loading && (
        <>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
            <strong>{data.series_name}</strong>
            {data.description && (
              <span> — {data.description}</span>
            )}
            <div>
              {data.n_points} punti · variance reduction:{" "}
              <strong style={{ color: "var(--reflation, #10b981)" }}>
                {pct(data.variance_reduction, 1)}
              </strong>{" "}
              · lambda = {data.lambda_used}
            </div>
          </div>
          <div style={{ overflowX: "auto" }}>
            <SmoothChart data={data} />
          </div>
        </>
      )}
      {loading && <div className="loading">Computing Kalman…</div>}
    </div>
  );
}
