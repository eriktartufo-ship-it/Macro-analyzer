interface Props {
  scores: Record<string, number>;
}

function formatAsset(name: string): string {
  return name.replace(/_/g, " ");
}

export function AssetRankingTable({ scores }: Props) {
  const sorted = Object.entries(scores)
    .map(([asset_class, final_score]) => ({ asset_class, final_score }))
    .sort((a, b) => b.final_score - a.final_score);
  const maxScore = Math.max(...sorted.map((s) => s.final_score), 1);

  return (
    <div className="card">
      <h2>Asset Ranking</h2>
      <table className="table">
        <thead>
          <tr>
            <th style={{ width: 32 }}>#</th>
            <th>Asset class</th>
            <th className="num">Final score</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((s, i) => {
            const pct = (s.final_score / maxScore) * 100;
            return (
              <tr key={s.asset_class}>
                <td style={{ color: "var(--muted)" }}>{i + 1}</td>
                <td className="asset-name">{formatAsset(s.asset_class)}</td>
                <td className="num">
                  <span style={{ fontWeight: 600 }}>{s.final_score.toFixed(1)}</span>
                  <span className="score-bar">
                    <span className="score-bar-fill" style={{ width: `${pct}%` }} />
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
