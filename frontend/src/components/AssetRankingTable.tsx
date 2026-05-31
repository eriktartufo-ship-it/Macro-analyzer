import React, { useState, useEffect } from "react";
import { ScrollShadow } from "./ScrollShadow";
import { MultiLineChart, type ChartPoint, type ChartSeries } from "./MultiLineChart";
import { api } from "../api/client";
import type { SignalsHistoryItem } from "../types";

interface Props {
  scores: Record<string, number>;
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

export function AssetRankingTable({ scores }: Props) {
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

  const rows = Object.entries(scores)
    .map(([asset_class, final_score]) => ({ asset_class, final_score }))
    .sort((a, b) => b.final_score - a.final_score);

  const maxScore = Math.max(...rows.map((r) => r.final_score), 1);

  const renderAssetChart = (asset: string) => {
    const points: ChartPoint[] = history.map((h) => ({
      date: h.date,
      values: { [asset]: h.scores[asset] ?? null },
    }));
    return (
      <MultiLineChart
        title={`${formatAsset(asset)} — ultimi 12 mesi`}
        points={points}
        series={[{ key: asset, label: formatAsset(asset), color: "var(--accent)" }]}
        yFormat={(v) => v.toFixed(1)}
        height={180}
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
            subtitle={`12 mesi storici — top ${topAssets.length} asset (score corrente)`}
            points={chartPoints}
            series={chartSeries}
            yFormat={(v) => v.toFixed(1)}
          />
        </div>
      )}

      <div className="card">
        <h2>
          Asset Ranking
          <span style={{ textTransform: "none", fontWeight: 500, color: "var(--muted)", marginLeft: 8 }}>
            — score corrente del regime attivo
          </span>
        </h2>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
          Per le proiezioni della transizione di regime vedi il pannello "Asset beneficiari del regime proiettato" nella home.
        </div>

        {isMobile ? (
          <div className="asset-mobile-list">
            {rows.map((r, i) => {
              const pct = (r.final_score / maxScore) * 100;
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
                      <span className="score-bar-fill" style={{ width: `${pct}%`, background: "var(--accent)" }} />
                    </span>
                  </div>
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
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => {
                    const pct = (r.final_score / maxScore) * 100;
                    const isExpanded = expanded === r.asset_class;
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
                                <span className="score-bar-fill" style={{ width: `${pct}%`, background: "var(--accent)" }} />
                              </span>
                            </div>
                          </td>
                        </tr>
                        {isExpanded && history.length > 1 && (
                          <tr>
                            <td colSpan={3} style={{ padding: "4px 12px 12px 12px", background: "var(--bg)" }}>
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
