import { useState, useEffect } from "react";
import { ScrollShadow } from "./ScrollShadow";

interface Props {
  scores: Record<string, number>;
  projected?: Record<string, number> | null;
}

function formatAsset(name: string): string {
  return name.replace(/_/g, " ");
}

function deltaBarColor(delta: number | null): string {
  if (delta === null) return "var(--accent)";
  return delta >= 0 ? "var(--reflation)" : "var(--deflation)";
}

export function AssetRankingTable({ scores, projected }: Props) {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth <= 640);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

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

  return (
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

            return (
              <div key={r.asset_class} className="asset-mobile-card">
                <div className="asset-mobile-header">
                  <span className="rank-badge">{i + 1}</span>
                  <span className="asset-name">{formatAsset(r.asset_class)}</span>
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

                  return (
                    <tr key={r.asset_class}>
                      <td className="rank" data-label="Rank">{i + 1}</td>
                      <td className="asset-name" data-label="Asset Class">{formatAsset(r.asset_class)}</td>
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
                  );
                })}
              </tbody>
            </table>
          </ScrollShadow>
        </>
      )}
    </div>
  );
}

function isClassOnly(isUp: boolean) {
  return isUp ? "delta-up" : "delta-down";
}
