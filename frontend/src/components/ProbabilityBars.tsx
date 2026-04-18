import type { RegimeProbabilities, Regime } from "../types";

interface Props {
  probabilities: RegimeProbabilities;
}

const REGIMES: Regime[] = [
  "reflation",
  "goldilocks",
  "deflation",
  "stagflation",
];

export function ProbabilityBars({ probabilities }: Props) {
  const sorted = REGIMES.map((r) => ({ regime: r, value: probabilities[r] })).sort(
    (a, b) => b.value - a.value
  );
  const max = Math.max(...sorted.map((s) => s.value), 0.001);

  return (
    <div className="card">
      <h2>Regime Probabilities</h2>
      {sorted.map(({ regime, value }) => {
        const width = (value / max) * 100;
        return (
          <div key={regime} className="prob-row">
            <div className={`prob-label regime-${regime}`}>{regime}</div>
            <div className="prob-bar">
              <div className={`prob-fill bg-${regime}`} style={{ width: `${width}%` }} />
            </div>
            <div className="prob-value">{(value * 100).toFixed(1)}%</div>
          </div>
        );
      })}
    </div>
  );
}
