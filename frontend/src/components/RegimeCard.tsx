import type { CurrentRegime } from "../types";

interface Props {
  data: CurrentRegime;
}

export function RegimeCard({ data }: Props) {
  const pct = (data.confidence * 100).toFixed(0);
  return (
    <div className="card regime-card">
      <h2>Current Regime</h2>
      <div className="regime-card-top">
        <div className={`regime-name regime-${data.regime}`}>{data.regime}</div>
        <div className="regime-meta" style={{ marginBottom: 0 }}>As of {data.date}</div>
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
    </div>
  );
}
