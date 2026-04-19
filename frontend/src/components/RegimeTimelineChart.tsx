import { useMemo } from "react";
import type { Regime, RegimeHistoryItem } from "../types";

interface Props {
  history: RegimeHistoryItem[];
  projected?: Record<string, number> | null;
}

function regimeValue(item: RegimeHistoryItem, regime: Regime): number {
  const fit = item.fit_scores?.[regime];
  if (typeof fit === "number") return fit;
  return item.probabilities[regime];
}

const REGIMES: Regime[] = ["reflation", "goldilocks", "deflation", "stagflation"];

const PROJECTION_DAYS = 90;

// SVG coordinate system (responsive via width:100% + preserveAspectRatio)
const VB_W = 640;
const VB_H = 220;
const PAD_L = 36;
const PAD_R = 14;
const PAD_T = 14;
const PAD_B = 28;
const PLOT_W = VB_W - PAD_L - PAD_R;
const PLOT_H = VB_H - PAD_T - PAD_B;

function diffDays(a: string, b: string): number {
  const ms = new Date(b).getTime() - new Date(a).getTime();
  return Math.round(ms / 86400000);
}

function monthLabel(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("it-IT", { month: "short" });
}

export function RegimeTimelineChart({ history, projected }: Props) {
  const chart = useMemo(() => {
    const sorted = [...history].sort((a, b) => (a.date < b.date ? -1 : 1));
    if (sorted.length < 2) return null;

    const firstDate = sorted[0].date;
    const lastDate = sorted[sorted.length - 1].date;
    const historyDays = Math.max(1, diffDays(firstDate, lastDate));
    const totalDays = historyDays + (projected ? PROJECTION_DAYS : 0);

    const xFor = (dayOffset: number) =>
      PAD_L + (dayOffset / totalDays) * PLOT_W;
    const yFor = (prob: number) =>
      PAD_T + (1 - prob) * PLOT_H;

    const lastPoint = sorted[sorted.length - 1];

    const series = REGIMES.map((regime) => {
      const historyPath = sorted
        .map((p) => {
          const offset = diffDays(firstDate, p.date);
          return `${xFor(offset).toFixed(1)},${yFor(regimeValue(p, regime)).toFixed(1)}`;
        })
        .join(" ");

      const projectionPath =
        projected && projected[regime] !== undefined
          ? `${xFor(historyDays).toFixed(1)},${yFor(regimeValue(lastPoint, regime)).toFixed(1)} ${xFor(totalDays).toFixed(1)},${yFor(projected[regime]).toFixed(1)}`
          : null;

      return { regime, historyPath, projectionPath };
    });

    // X-axis labels: start, monthly markers, today, end
    const xLabels: Array<{ x: number; label: string; emphasize?: boolean }> = [];
    // Monthly history labels
    const startD = new Date(firstDate);
    const endD = new Date(lastDate);
    const cursor = new Date(startD.getFullYear(), startD.getMonth() + 1, 1);
    while (cursor <= endD) {
      const offset = diffDays(firstDate, cursor.toISOString().slice(0, 10));
      if (offset >= 10 && offset <= historyDays - 10) {
        xLabels.push({ x: xFor(offset), label: monthLabel(cursor.toISOString()) });
      }
      cursor.setMonth(cursor.getMonth() + 1);
    }
    xLabels.push({ x: xFor(historyDays), label: "oggi", emphasize: true });
    if (projected) {
      xLabels.push({ x: xFor(totalDays), label: "+3m" });
    }

    const todayX = xFor(historyDays);

    return { series, xLabels, todayX, hasProjection: !!projected };
  }, [history, projected]);

  if (!chart) return null;

  return (
    <div className="card regime-chart-card">
      <h2>
        Regime Timeline
        <span className="regime-chart-sub">— 6 mesi storici + proiezione 3 mesi</span>
      </h2>

      <div className="regime-chart-legend">
        {REGIMES.map((r) => (
          <span key={r} className="regime-chart-legend-item">
            <span className={`regime-chart-swatch bg-${r}`} />
            <span className={`regime-${r}`} style={{ textTransform: "capitalize" }}>
              {r}
            </span>
          </span>
        ))}
      </div>

      <svg
        className="regime-chart"
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="Regime probability timeline"
      >
        {/* Horizontal gridlines at 25 / 50 / 75 / 100% */}
        {[0, 0.25, 0.5, 0.75, 1].map((p) => {
          const y = PAD_T + (1 - p) * PLOT_H;
          return (
            <g key={p}>
              <line
                x1={PAD_L}
                x2={VB_W - PAD_R}
                y1={y}
                y2={y}
                className="regime-chart-grid"
              />
              <text
                x={PAD_L - 6}
                y={y + 3}
                className="regime-chart-axis-label"
                textAnchor="end"
              >
                {Math.round(p * 100)}%
              </text>
            </g>
          );
        })}

        {/* "Today" vertical separator */}
        {chart.hasProjection && (
          <line
            x1={chart.todayX}
            x2={chart.todayX}
            y1={PAD_T}
            y2={VB_H - PAD_B}
            className="regime-chart-today"
          />
        )}

        {/* X-axis labels */}
        {chart.xLabels.map((l, i) => (
          <text
            key={i}
            x={l.x}
            y={VB_H - PAD_B + 16}
            className={`regime-chart-axis-label ${l.emphasize ? "regime-chart-axis-emph" : ""}`}
            textAnchor="middle"
          >
            {l.label}
          </text>
        ))}

        {/* Lines per regime */}
        {chart.series.map(({ regime, historyPath, projectionPath }) => (
          <g key={regime} className={`regime-chart-line regime-chart-line-${regime}`}>
            <polyline points={historyPath} className="regime-chart-history" />
            {projectionPath && (
              <polyline points={projectionPath} className="regime-chart-projection" />
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}
