import type { CurrentRegime, Trajectory } from "../types";

interface Props {
  data: CurrentRegime;
  trajectory?: Trajectory | null;
}

function riskColor(risk: number): string {
  if (risk > 0.7) return "var(--deflation)";
  if (risk > 0.4) return "var(--goldilocks)";
  return "var(--reflation)";
}

export function RegimeCard({ data, trajectory }: Props) {
  const pct = (data.confidence * 100).toFixed(0);
  const hasTrajectory = !!trajectory;
  const isTransition =
    hasTrajectory && trajectory.current_regime !== trajectory.projected_regime;

  return (
    <div className="card regime-card">
      <h2>Current Regime</h2>
      <div className="regime-card-top">
        <div className="regime-headline">
          <div className={`regime-name regime-${data.regime}`}>{data.regime}</div>
          {hasTrajectory && (
            <>
              <div className="regime-arrow">{"\u2192"}</div>
              <div className={`regime-name regime-projected regime-${trajectory.projected_regime}`}>
                {trajectory.projected_regime}
              </div>
              {isTransition && (
                <span className="chip chip-danger" data-nowrap>
                  TRANSITION
                </span>
              )}
            </>
          )}
        </div>
        <div className="regime-meta" style={{ marginBottom: 0 }}>
          As of {data.date}
        </div>
      </div>

      <div className="regime-card-bottom">
        <div className="regime-meta" style={{ marginBottom: 6 }}>
          Confidence
        </div>
        <div className="confidence">
          <div className="confidence-bar">
            <div className="confidence-fill" style={{ width: `${pct}%` }} />
          </div>
          <div style={{ fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>
            {pct}%
          </div>
        </div>
      </div>

      {hasTrajectory && (
        <div className="regime-trajectory">
          <div className="regime-trajectory-meta">
            <span
              data-nowrap
              style={{ color: riskColor(trajectory.transition_risk), fontWeight: 600 }}
            >
              Transition risk: {(trajectory.transition_risk * 100).toFixed(0)}%
            </span>
            <span style={{ color: "var(--muted)" }}>{trajectory.summary}</span>
          </div>
        </div>
      )}
    </div>
  );
}
