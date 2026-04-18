import type { Trajectory } from "../types";

interface Props {
  trajectory: Trajectory;
  currentScores: Record<string, number>;
  topN?: number;
}

function formatAsset(name: string): string {
  return name.replace(/_/g, " ");
}

export function ProjectedAssetsPanel({ trajectory, currentScores, topN = 5 }: Props) {
  const projected = trajectory.projected_scores || {};
  if (Object.keys(projected).length === 0) return null;

  const rows = Object.entries(projected).map(([asset, projScore]) => {
    const current = currentScores[asset] ?? 0;
    return {
      asset,
      current,
      projected: projScore,
      delta: projScore - current,
    };
  });

  const topBenefit = [...rows]
    .sort((a, b) => b.delta - a.delta)
    .slice(0, topN);
  const topLoss = [...rows]
    .sort((a, b) => a.delta - b.delta)
    .slice(0, topN);

  const regimeChanging = trajectory.current_regime !== trajectory.projected_regime;
  const maxAbsDelta = Math.max(
    ...rows.map((r) => Math.abs(r.delta)),
    0.01
  );

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
        <h2 style={{ margin: 0 }}>Asset beneficiari del regime proiettato</h2>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>
          <span className={`regime-${trajectory.current_regime}`} style={{ textTransform: "capitalize" }}>
            {trajectory.current_regime}
          </span>
          {" → "}
          <span className={`regime-${trajectory.projected_regime}`} style={{ textTransform: "capitalize", fontWeight: 600 }}>
            {trajectory.projected_regime}
          </span>
        </span>
      </div>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 14 }}>
        {regimeChanging
          ? "Transizione di regime in corso: ecco chi guadagna e chi perde se la proiezione si materializza."
          : "Regime stabile ma probabilità in movimento: variazioni di score se la drift si concretizza."}
      </div>

      <div className="grid grid-2" style={{ gap: 16, marginBottom: 0 }}>
        <AssetList
          title="Top beneficiari"
          accent="var(--reflation)"
          rows={topBenefit}
          maxAbsDelta={maxAbsDelta}
        />
        <AssetList
          title="Peggiori performer"
          accent="var(--deflation)"
          rows={topLoss}
          maxAbsDelta={maxAbsDelta}
        />
      </div>
    </div>
  );
}

interface ListProps {
  title: string;
  accent: string;
  rows: { asset: string; current: number; projected: number; delta: number }[];
  maxAbsDelta: number;
}

function AssetList({ title, accent, rows, maxAbsDelta }: ListProps) {
  return (
    <div>
      <div style={{
        fontSize: 11, fontWeight: 700, color: accent,
        textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 8,
      }}>
        {title}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {rows.map((r) => {
          const isUp = r.delta >= 0;
          const barPct = (Math.abs(r.delta) / maxAbsDelta) * 100;
          return (
            <div
              key={r.asset}
              style={{
                display: "grid",
                gridTemplateColumns: "1fr auto auto",
                alignItems: "center",
                gap: 10,
                padding: "6px 10px",
                background: "var(--bg)",
                borderRadius: 4,
              }}
            >
              <span style={{ fontSize: 13, textTransform: "capitalize" }}>
                {formatAsset(r.asset)}
              </span>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 11, color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
                  {r.current.toFixed(1)}
                </span>
                <span style={{ fontSize: 11, color: "var(--muted)" }}>→</span>
                <span style={{ fontSize: 12, fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
                  {r.projected.toFixed(1)}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 90, justifyContent: "flex-end" }}>
                <div style={{
                  width: 48, height: 4, background: "var(--surface-2)", borderRadius: 2, overflow: "hidden",
                }}>
                  <div style={{
                    width: `${barPct}%`, height: "100%",
                    background: isUp ? "var(--reflation)" : "var(--deflation)",
                  }} />
                </div>
                <span style={{
                  fontSize: 12, fontWeight: 700, fontVariantNumeric: "tabular-nums",
                  color: isUp ? "var(--reflation)" : "var(--deflation)",
                  minWidth: 42, textAlign: "right",
                }}>
                  {isUp ? "+" : ""}{r.delta.toFixed(1)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
