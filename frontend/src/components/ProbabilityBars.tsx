import type { RegimeProbabilities, Regime } from "../types";

interface Props {
  probabilities: RegimeProbabilities;
  projected?: Record<string, number> | null;
}

const REGIMES: Regime[] = [
  "reflation",
  "goldilocks",
  "deflation",
  "stagflation",
];

export function ProbabilityBars({ probabilities, projected }: Props) {
  const rows = REGIMES.map((r) => ({
    regime: r,
    value: probabilities[r],
    proj: projected?.[r] ?? null,
  })).sort((a, b) => b.value - a.value);

  const allValues = rows.flatMap((r) => [r.value, r.proj ?? 0]);
  const max = Math.max(...allValues, 0.001);
  const hasProjection = projected && Object.keys(projected).length > 0;

  return (
    <div className="card probabilities-card">
      <h2>Regime Probabilities</h2>
      <div className="probabilities-list">
      {rows.map(({ regime, value, proj }) => {
        const width = (value / max) * 100;
        const delta = proj !== null ? proj - value : 0;
        const deltaPct = (delta * 100).toFixed(1);
        const isFlat = Math.abs(delta) < 0.005;
        const isUp = delta > 0;
        const deltaClass = isFlat ? "delta-flat" : isUp ? "delta-up" : "delta-down";
        // Delta zone overlay: from min(value,proj) to max(value,proj)
        const deltaStart = proj !== null ? (Math.min(value, proj) / max) * 100 : 0;
        const deltaWidth =
          proj !== null ? (Math.abs(delta) / max) * 100 : 0;
        return (
          <div key={regime} className="prob-row">
            <div className={`prob-label regime-${regime}`}>{regime}</div>
            <div className="prob-bar">
              <div
                className={`prob-fill bg-${regime}`}
                style={{ width: `${width}%` }}
              />
              {proj !== null && !isFlat && (
                <div
                  className={`prob-fill-delta ${isUp ? "delta-up" : "delta-down"}`}
                  style={{ left: `${deltaStart}%`, width: `${deltaWidth}%` }}
                  title={`Proiezione: ${(proj * 100).toFixed(1)}% (${isUp ? "+" : ""}${deltaPct}pp)`}
                />
              )}
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
              <div className="prob-value">{(value * 100).toFixed(1)}%</div>
              {proj !== null && hasProjection && (
                <div className={`prob-projected-note ${deltaClass}`} data-nowrap>
                  {delta > 0 ? "+" : ""}{deltaPct}pp
                </div>
              )}
            </div>
          </div>
        );
      })}
      </div>
      {hasProjection && (
        <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 10 }}>
          La zona tratteggiata mostra il cambio atteso: verde se il regime guadagna forza, rossa se la perde.
        </div>
      )}
    </div>
  );
}
