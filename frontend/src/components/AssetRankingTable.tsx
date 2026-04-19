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
      <div className="scroll-label">← Scorri per vedere tutte le colonne →</div>
      <ScrollShadow innerClassName="table-wrap">
        <table className="table">
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
              const delta =
                r.projected_score !== null ? r.projected_score - r.final_score : 0;
              const deltaAbs = Math.abs(delta);
              const isUp = delta >= 0;
              const deltaClass = isUp ? "delta-up" : "delta-down";
              const barColor = deltaBarColor(r.projected_score !== null ? delta : null);
              const deltaStart =
                r.projected_score !== null
                  ? (Math.min(r.final_score, r.projected_score) / maxScore) * 100
                  : 0;
              const deltaWidth =
                r.projected_score !== null ? (deltaAbs / maxScore) * 100 : 0;

              return (
                <tr key={r.asset_class}>
                  <td className="rank">{i + 1}</td>
                  <td className="asset-name">{formatAsset(r.asset_class)}</td>
                  <td className="num">
                    <div className="score-stack">
                      <span className="score-num">{r.final_score.toFixed(1)}</span>
                      <span className="score-bar" aria-hidden="true">
                        <span
                          className="score-bar-fill"
                          style={{ width: `${pct}%`, background: barColor }}
                        />
                        {r.projected_score !== null && deltaWidth > 0.01 && (
                          <span
                            className={`score-bar-delta ${isUp ? "delta-up" : "delta-down"}`}
                            style={{ left: `${deltaStart}%`, width: `${deltaWidth}%` }}
                          />
                        )}
                      </span>
                    </div>
                  </td>
                  {hasProj && (
                    <td className="num">
                      <span
                        className={`score-num ${deltaClass}`}
                        style={{ fontWeight: 600 }}
                      >
                        {r.projected_score !== null ? r.projected_score.toFixed(1) : "—"}
                      </span>
                    </td>
                  )}
                  {hasProj && (
                    <td className="num">
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
    </div>
  );
}
