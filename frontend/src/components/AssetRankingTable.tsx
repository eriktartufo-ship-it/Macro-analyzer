import React, { useState, useEffect } from "react";
import { ScrollShadow } from "./ScrollShadow";
import { MultiLineChart, type ChartPoint, type ChartSeries } from "./MultiLineChart";
import { api } from "../api/client";
import type { SignalsHistoryItem } from "../types";

interface Props {
  scores: Record<string, number>;
  projected?: Record<string, number> | null;
}

// Palette distinta per le linee (max 8 asset)
const LINE_COLORS = [
  "var(--reflation)",
  "var(--goldilocks)",
  "var(--stagflation)",
  "var(--deflation)",
  "var(--accent)",
  "#9b59b6",
  "#16a085",
  "#e67e22",
];

function formatAsset(name: string): string {
  return name.replace(/_/g, " ");
}

function deltaBarColor(delta: number | null): string {
  if (delta === null) return "var(--accent)";
  return delta >= 0 ? "var(--reflation)" : "var(--deflation)";
}

export function AssetRankingTable({ scores, projected }: Props) {
  const [isMobile, setIsMobile] = useState(false);
  const [history, setHistory] = useState<SignalsHistoryItem[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth <= 640);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  useEffect(() => {
    let alive = true;
    api
      .signalsHistory(365)
      .then((rows) => {
        if (alive) setHistory(rows);
      })
      .catch(() => {
        if (alive) setHistory([]);
      });
    return () => {
      alive = false;
    };
  }, []);

  const topAssets = Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([name]) => name);

  const chartSeries: ChartSeries[] = topAssets.map((name, i) => ({
    key: name,
    label: name.replace(/_/g, " "),
    color: LINE_COLORS[i % LINE_COLORS.length],
  }));

  const chartPoints: ChartPoint[] = history.map((h) => ({
    date: h.date,
    values: Object.fromEntries(topAssets.map((a) => [a, h.scores[a] ?? null])),
  }));

  const hasProj = !!projected && Object.keys(projected).length > 0;

  const rows = Object.entries(scores)
    .map(([asset_class, final_score]) => ({
      asset_class,
      final_score,
      projected_score: projected?.[asset_class] ?? null,
    }))
    .sort((a, b) => b.final_score - a.final_score);

  const allScores = rows.flatMap((r) =>
    r.projected_score !== null ? [r.final_score, r.projected_score] : [r.final_score],
  );
  const maxScore = Math.max(...allScores, 1);

  const renderAssetChart = (asset: string) => {
    const points: ChartPoint[] = history.map((h) => ({
      date: h.date,
      values: { [asset]: h.scores[asset] ?? null },
    }));
    const projVal = projected?.[asset];
    return (
      <MultiLineChart
        title={`${formatAsset(asset)} — ultimi 12 mesi`}
        points={points}
        series={[{ key: asset, label: formatAsset(asset), color: "var(--accent)" }]}
        yFormat={(v) => v.toFixed(1)}
        height={180}
        projection={typeof projVal === "number" ? { [asset]: projVal } : undefined}
        compact
        showLegend={false}
      />
    );
  };

  const toggleExpand = (asset: string) =>
    setExpanded((cur) => (cur === asset ? null : asset));

  return (
    <div>
      {chartPoints.length > 1 && (
        <div style={{ marginBottom: 18 }}>
          <MultiLineChart
            title="Asset Scores Timeline"
            subtitle={`12 mesi storici — top ${topAssets.length} asset`}
            points={chartPoints}
            series={chartSeries}
            yFormat={(v) => v.toFixed(1)}
          />
        </div>
      )}

    <div className="card">
      <h2>Asset Ranking {hasProj && <span style={{ textTransform: "none", fontWeight: 500, color: "var(--muted)" }}>— corrente vs proiezione</span>}</h2>
      {hasProj && (
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
          Barra verde = proiezione in rialzo, rossa = in ribasso. La zona tratteggiata mostra l'ampiezza del cambio atteso.
        </div>
      )}
      
      {isMobile ? (
        <div className="asset-mobile-list">
          {rows.map((r, i) => {
            const pct = (r.final_score / maxScore) * 100;
            const delta = r.projected_score !== null ? r.projected_score - r.final_score : 0;
            const deltaAbs = Math.abs(delta);
            const isUp = delta >= 0;
            const deltaClass = isUp ? "delta-up" : "delta-down";
            const barColor = deltaBarColor(r.projected_score !== null ? delta : null);
            const deltaStart = r.projected_score !== null ? (Math.min(r.final_score, r.projected_score) / maxScore) * 100 : 0;
            const deltaWidth = r.projected_score !== null ? (deltaAbs / maxScore) * 100 : 0;

            const isExpanded = expanded === r.asset_class;
            return (
              <div
                key={r.asset_class}
                className="asset-mobile-card"
                onClick={() => toggleExpand(r.asset_class)}
                style={{ cursor: "pointer" }}
                role="button"
                aria-expanded={isExpanded}
              >
                <div className="asset-mobile-header">
                  <span className="rank-badge">{i + 1}</span>
                  <span className="asset-name">
                    {formatAsset(r.asset_class)}
                    <span style={{ marginLeft: 6, fontSize: 11, color: "var(--muted)" }}>
                      {isExpanded ? "▼" : "▶"}
                    </span>
                  </span>
                  <span className="score-num" style={{ fontWeight: 700 }}>{r.final_score.toFixed(1)}</span>
                </div>
                <div style={{ padding: "12px 0 0 36px" }}>
                  <span className="score-bar" aria-hidden="true" style={{ width: "100%", display: "block" }}>
                    <span className="score-bar-fill" style={{ width: `${pct}%`, background: barColor }} />
                    {r.projected_score !== null && deltaWidth > 0.01 && (
                      <span className={`score-bar-delta ${isClassOnly(isUp)}`} style={{ left: `${deltaStart}%`, width: `${deltaWidth}%` }} />
                    )}
                  </span>
                </div>
                {hasProj && (
                  <div className="asset-mobile-proj" style={{ marginLeft: "36px" }}>
                    <span className="proj-label">Proiezione:</span>
                    <span className={`score-num ${deltaClass}`}>{r.projected_score !== null ? r.projected_score.toFixed(1) : "—"}</span>
                    <span className={`score-num ${deltaClass}`} style={{ marginLeft: "auto" }}>
                      {r.projected_score !== null ? (delta > 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1)) : "—"}
                    </span>
                  </div>
                )}
                {isExpanded && history.length > 1 && (
                  <div onClick={(e) => e.stopPropagation()} style={{ padding: "0 12px 8px 12px" }}>
                    {renderAssetChart(r.asset_class)}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <>
          <div className="scroll-label">← Scorri per vedere tutte le colonne →</div>
          <ScrollShadow innerClassName="table-wrap">
            <table className="table table-responsive">
              <thead>
                <tr>
                  <th className="num" style={{ width: 44 }}>#</th>
                  <th>Asset class</th>
                  <th className="num">Score</th>
                  {hasProj && <th className="num">Proiezione</th>}
                  {hasProj && <th className="num">Δ</th>}
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const pct = (r.final_score / maxScore) * 100;
                  const delta = r.projected_score !== null ? r.projected_score - r.final_score : 0;
                  const deltaAbs = Math.abs(delta);
                  const isUp = delta >= 0;
                  const deltaClass = isUp ? "delta-up" : "delta-down";
                  const barColor = deltaBarColor(r.projected_score !== null ? delta : null);
                  const deltaStart = r.projected_score !== null ? (Math.min(r.final_score, r.projected_score) / maxScore) * 100 : 0;
                  const deltaWidth = r.projected_score !== null ? (deltaAbs / maxScore) * 100 : 0;
                  const isExpanded = expanded === r.asset_class;
                  const colSpan = 3 + (hasProj ? 2 : 0);

                  return (
                    <React.Fragment key={r.asset_class}>
                    <tr
                      onClick={() => toggleExpand(r.asset_class)}
                      style={{ cursor: "pointer" }}
                      aria-expanded={isExpanded}
                    >
                      <td className="rank" data-label="Rank">{i + 1}</td>
                      <td className="asset-name" data-label="Asset Class">
                        <span style={{ marginRight: 6, fontSize: 11, color: "var(--muted)" }}>
                          {isExpanded ? "▼" : "▶"}
                        </span>
                        {formatAsset(r.asset_class)}
                      </td>
                      <td className="num" data-label="Score">
                        <div className="score-stack">
                          <span className="score-num">{r.final_score.toFixed(1)}</span>
                          <span className="score-bar" aria-hidden="true">
                            <span className="score-bar-fill" style={{ width: `${pct}%`, background: barColor }} />
                            {r.projected_score !== null && deltaWidth > 0.01 && (
                              <span className={`score-bar-delta ${isClassOnly(isUp)}`} style={{ left: `${deltaStart}%`, width: `${deltaWidth}%` }} />
                            )}
                          </span>
                        </div>
                      </td>
                      {hasProj && (
                        <td className="num" data-label="Proiezione">
                          <span className={`score-num ${deltaClass}`} style={{ fontWeight: 600 }}>
                            {r.projected_score !== null ? r.projected_score.toFixed(1) : "—"}
                          </span>
                        </td>
                      )}
                      {hasProj && (
                        <td className="num" data-label="Δ">
                          {r.projected_score !== null ? (
                            <span className={`score-num ${deltaClass}`} data-nowrap>
                              {delta > 0 ? "+" : ""}
                              {delta.toFixed(1)}
                            </span>
                          ) : (
                            <span style={{ color: "var(--muted)" }}>—</span>
                          )}
                        </td>
                      )}
                    </tr>
                    {isExpanded && history.length > 1 && (
                      <tr>
                        <td colSpan={colSpan} style={{ padding: "4px 12px 12px 12px", background: "var(--bg)" }}>
                          {renderAssetChart(r.asset_class)}
                        </td>
                      </tr>
                    )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </ScrollShadow>
        </>
      )}
    </div>
    </div>
  );
}

function isClassOnly(isUp: boolean) {
  return isUp ? "delta-up" : "delta-down";
}
