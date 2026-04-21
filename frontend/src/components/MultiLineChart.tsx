import { useMemo } from "react";

export interface ChartSeries {
  key: string;
  label: string;
  color: string;
}

export interface ChartPoint {
  date: string;
  values: Record<string, number | null | undefined>;
}

interface Props {
  title: string;
  subtitle?: string;
  points: ChartPoint[];
  series: ChartSeries[];
  yDomain?: [number, number];
  yFormat?: (v: number) => string;
  height?: number;
  projection?: Record<string, number>;
  projectionDays?: number;
  projectionLabel?: string;
  compact?: boolean;
  showLegend?: boolean;
}

const VB_W = 640;
const PAD_L = 44;
const PAD_R = 14;
const PAD_T = 14;
const PAD_B = 28;

function diffDays(a: string, b: string): number {
  const ms = new Date(b).getTime() - new Date(a).getTime();
  return Math.round(ms / 86400000);
}

function monthLabel(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("it-IT", { month: "short" });
}

export function MultiLineChart({
  title,
  subtitle,
  points,
  series,
  yDomain,
  yFormat,
  height = 220,
  projection,
  projectionDays = 90,
  projectionLabel = "+3m",
  compact = false,
  showLegend = true,
}: Props) {
  const hasProjection = !!projection && Object.keys(projection).length > 0;

  const chart = useMemo(() => {
    const sorted = [...points].sort((a, b) => (a.date < b.date ? -1 : 1));
    if (sorted.length < 2) return null;

    const firstDate = sorted[0].date;
    const lastDate = sorted[sorted.length - 1].date;
    const historyDays = Math.max(1, diffDays(firstDate, lastDate));
    const totalDays = historyDays + (hasProjection ? projectionDays : 0);

    // Auto-domain se non fornito — considera anche la proiezione
    let yMin = yDomain?.[0] ?? Infinity;
    let yMax = yDomain?.[1] ?? -Infinity;
    if (!yDomain) {
      for (const p of sorted) {
        for (const s of series) {
          const v = p.values[s.key];
          if (typeof v === "number" && isFinite(v)) {
            if (v < yMin) yMin = v;
            if (v > yMax) yMax = v;
          }
        }
      }
      if (hasProjection) {
        for (const s of series) {
          const v = projection![s.key];
          if (typeof v === "number" && isFinite(v)) {
            if (v < yMin) yMin = v;
            if (v > yMax) yMax = v;
          }
        }
      }
      if (!isFinite(yMin) || !isFinite(yMax)) {
        yMin = 0;
        yMax = 1;
      }
      if (yMin === yMax) {
        yMin -= 0.5;
        yMax += 0.5;
      }
      const pad = (yMax - yMin) * 0.08;
      yMin -= pad;
      yMax += pad;
    }

    const plotW = VB_W - PAD_L - PAD_R;
    const plotH = height - PAD_T - PAD_B;

    const xFor = (dayOffset: number) => PAD_L + (dayOffset / totalDays) * plotW;
    const yFor = (v: number) => PAD_T + (1 - (v - yMin) / (yMax - yMin)) * plotH;

    const paths = series.map((s) => {
      const coords: string[] = [];
      let lastCoord: { x: number; y: number } | null = null;
      for (const p of sorted) {
        const v = p.values[s.key];
        if (typeof v === "number" && isFinite(v)) {
          const offset = diffDays(firstDate, p.date);
          const x = xFor(offset);
          const y = yFor(v);
          coords.push(`${x.toFixed(1)},${y.toFixed(1)}`);
          lastCoord = { x, y };
        }
      }
      const history = coords.join(" ");
      let projectionPath: string | null = null;
      if (hasProjection && lastCoord) {
        const projV = projection![s.key];
        if (typeof projV === "number" && isFinite(projV)) {
          projectionPath = `${lastCoord.x.toFixed(1)},${lastCoord.y.toFixed(1)} ${xFor(totalDays).toFixed(1)},${yFor(projV).toFixed(1)}`;
        }
      }
      return { series: s, history, projectionPath };
    });

    const xLabels: Array<{ x: number; label: string; emphasize?: boolean }> = [];
    const startD = new Date(firstDate);
    const endD = new Date(lastDate);
    const monthStride = historyDays > 200 ? 2 : 1;
    const cursor = new Date(startD.getFullYear(), startD.getMonth() + monthStride, 1);
    while (cursor <= endD) {
      const offset = diffDays(firstDate, cursor.toISOString().slice(0, 10));
      if (offset >= 10 && offset <= historyDays - 20) {
        xLabels.push({ x: xFor(offset), label: monthLabel(cursor.toISOString()) });
      }
      cursor.setMonth(cursor.getMonth() + monthStride);
    }
    xLabels.push({ x: xFor(historyDays), label: "oggi", emphasize: true });
    if (hasProjection) {
      xLabels.push({ x: xFor(totalDays), label: projectionLabel });
    }

    const todayX = xFor(historyDays);
    const gridValues = [yMin, yMin + (yMax - yMin) * 0.25, yMin + (yMax - yMin) * 0.5, yMin + (yMax - yMin) * 0.75, yMax];

    return { paths, xLabels, gridValues, yMin, yMax, plotW, plotH, todayX };
  }, [points, series, yDomain, height, projection, hasProjection, projectionDays, projectionLabel]);

  if (!chart) {
    return (
      <div className={compact ? "" : "card"}>
        {!compact && (
          <h2>{title}{subtitle && <span className="regime-chart-sub">— {subtitle}</span>}</h2>
        )}
        <div style={{ padding: compact ? "8px 0" : "20px 0", color: "var(--muted)", fontSize: 12 }}>
          Storico insufficiente per visualizzare il grafico.
        </div>
      </div>
    );
  }

  const fmt = yFormat ?? ((v: number) => v.toFixed(2));

  const content = (
    <>
      {showLegend && series.length > 1 && (
        <div className="regime-chart-legend">
          {series.map((s) => (
            <span key={s.key} className="regime-chart-legend-item">
              <span className="regime-chart-swatch" style={{ background: s.color }} />
              <span>{s.label}</span>
            </span>
          ))}
        </div>
      )}

      <svg
        className="regime-chart"
        viewBox={`0 0 ${VB_W} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={title}
      >
        {chart.gridValues.map((v, i) => {
          const y = PAD_T + (1 - (v - chart.yMin) / (chart.yMax - chart.yMin)) * chart.plotH;
          return (
            <g key={i}>
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
                {fmt(v)}
              </text>
            </g>
          );
        })}

        {hasProjection && (
          <line
            x1={chart.todayX}
            x2={chart.todayX}
            y1={PAD_T}
            y2={height - PAD_B}
            className="regime-chart-today"
          />
        )}

        {chart.xLabels.map((l, i) => (
          <text
            key={i}
            x={l.x}
            y={height - PAD_B + 16}
            className={`regime-chart-axis-label ${l.emphasize ? "regime-chart-axis-emph" : ""}`}
            textAnchor="middle"
          >
            {l.label}
          </text>
        ))}

        {chart.paths.map(({ series: s, history, projectionPath }) => (
          <g key={s.key} style={{ color: s.color }}>
            <polyline points={history} className="regime-chart-history" style={{ stroke: s.color }} />
            {projectionPath && (
              <polyline points={projectionPath} className="regime-chart-projection" style={{ stroke: s.color }} />
            )}
          </g>
        ))}
      </svg>
    </>
  );

  if (compact) {
    return (
      <div style={{ padding: "8px 0 4px 0" }}>
        {subtitle && (
          <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4 }}>{subtitle}</div>
        )}
        {content}
      </div>
    );
  }

  return (
    <div className="card regime-chart-card">
      <h2>
        {title}
        {subtitle && <span className="regime-chart-sub">— {subtitle}</span>}
      </h2>
      {content}
    </div>
  );
}
