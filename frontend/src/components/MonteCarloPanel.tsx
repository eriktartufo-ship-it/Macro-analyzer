import { useState } from "react";
import { api } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";
import type { MonteCarloForecast, RegimeBand, ScenarioPreset, ScenarioResult } from "../types";

const REGIME_LABEL: Record<string, string> = {
  reflation: "Reflation",
  stagflation: "Stagflation",
  deflation: "Deflation",
  goldilocks: "Goldilocks",
};

const REGIME_COLOR: Record<string, string> = {
  reflation: "#10b981",
  stagflation: "#d97706",
  deflation: "#ef4444",
  goldilocks: "#06b6d4",
};

function ConeChart({ bands, horizonDays }: { bands: RegimeBand[]; horizonDays: number }) {
  if (bands.length === 0) return null;
  const W = 720;
  const H = 280;
  const PAD = { top: 16, right: 90, bottom: 30, left: 50 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const nPoints = bands[0].median.length;

  const xScale = (i: number) => PAD.left + (i / (nPoints - 1)) * innerW;
  const yScale = (v: number) => PAD.top + (1 - v) * innerH;

  return (
    <svg width={W} height={H} style={{ background: "var(--bg)", borderRadius: 6 }}>
      {/* y-axis grid 0/25/50/75/100% */}
      {[0, 0.25, 0.5, 0.75, 1].map((p) => {
        const y = yScale(p);
        return (
          <g key={p}>
            <line x1={PAD.left} y1={y} x2={PAD.left + innerW} y2={y}
              stroke="var(--border)" strokeDasharray="2 2" opacity={0.3} />
            <text x={PAD.left - 6} y={y + 3} fontSize="10" fill="var(--muted)" textAnchor="end">
              {(p * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}

      {/* x-axis labels: ogni 3 step */}
      {Array.from({ length: nPoints }).map((_, i) => {
        if (i % 3 !== 0) return null;
        const x = xScale(i);
        return (
          <text key={i} x={x} y={H - 10} fontSize="10" fill="var(--muted)" textAnchor="middle">
            {i === 0 ? "now" : `+${i * horizonDays}d`}
          </text>
        );
      })}

      {/* Bands per regime */}
      {bands.map((b) => {
        // p10-p90 (light) + p25-p75 (medium) + median line
        const upperOuter = b.p90.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");
        const lowerOuter = b.p10.map((v, i) => `${xScale(i)},${yScale(v)}`).reverse().join(" ");
        const upperInner = b.p75.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");
        const lowerInner = b.p25.map((v, i) => `${xScale(i)},${yScale(v)}`).reverse().join(" ");
        const medianPath = b.median.map((v, i) => `${i === 0 ? "M" : "L"}${xScale(i)},${yScale(v)}`).join(" ");
        const color = REGIME_COLOR[b.regime] ?? "var(--text)";

        return (
          <g key={b.regime}>
            <polygon points={`${upperOuter} ${lowerOuter}`} fill={color} opacity={0.10} />
            <polygon points={`${upperInner} ${lowerInner}`} fill={color} opacity={0.20} />
            <path d={medianPath} fill="none" stroke={color} strokeWidth={2} />
          </g>
        );
      })}

      {/* Legend */}
      {bands.map((b, i) => (
        <g key={b.regime} transform={`translate(${PAD.left + innerW + 10}, ${PAD.top + i * 22})`}>
          <rect x={0} y={0} width={14} height={6} fill={REGIME_COLOR[b.regime]} opacity={0.20} />
          <line x1={0} y1={3} x2={14} y2={3} stroke={REGIME_COLOR[b.regime]} strokeWidth={2} />
          <text x={18} y={7} fontSize="10" fill="var(--text)">
            {REGIME_LABEL[b.regime]}
          </text>
        </g>
      ))}
      {/* Legenda bande */}
      <g transform={`translate(${PAD.left + innerW + 10}, ${PAD.top + bands.length * 22 + 10})`}>
        <text x={0} y={0} fontSize="9" fill="var(--muted)">band: p10-p90</text>
        <text x={0} y={12} fontSize="9" fill="var(--muted)">inner: p25-p75</text>
        <text x={0} y={24} fontSize="9" fill="var(--muted)">line: median</text>
      </g>
    </svg>
  );
}

export function MonteCarloPanel() {
  const [forecast, setForecast] = useState<MonteCarloForecast | null>(null);
  const [presets, setPresets] = useState<ScenarioPreset[] | null>(null);
  const [scenarioResult, setScenarioResult] = useState<ScenarioResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nSteps, setNSteps] = useState(12);
  const [horizonDays, setHorizonDays] = useState(30);
  const [nPaths, setNPaths] = useState(500);

  const runForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      const f = await api.monteCarloForecast({ nPaths, nSteps, horizonDays });
      setForecast(f);
      if (!presets) {
        const p = await api.scenariosList();
        setPresets(p);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Errore Monte Carlo");
    } finally {
      setLoading(false);
    }
  };

  const runScenario = async (key: string) => {
    setScenarioLoading(true);
    try {
      const r = await api.scenarioRun(key);
      setScenarioResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Errore scenario");
    } finally {
      setScenarioLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Monte Carlo forecast + shock scenarios</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Cone forecast a N step della distribuzione regime via N traiettorie campionate
        dalla transition matrix empirica. Bande p10/p25/p75/p90 + median.
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
        <label style={{ fontSize: 12 }}>
          Steps:{" "}
          <input type="number" min={1} max={36} value={nSteps}
            onChange={(e) => setNSteps(Number(e.target.value))} style={{ width: 50 }} />
        </label>
        <label style={{ fontSize: 12 }}>
          Horizon (gg/step):{" "}
          <input type="number" min={7} max={90} value={horizonDays}
            onChange={(e) => setHorizonDays(Number(e.target.value))} style={{ width: 60 }} />
        </label>
        <label style={{ fontSize: 12 }}>
          N paths:{" "}
          <input type="number" min={100} max={5000} step={100} value={nPaths}
            onChange={(e) => setNPaths(Number(e.target.value))} style={{ width: 70 }} />
        </label>
        <button onClick={runForecast} disabled={loading}
          style={{ padding: "6px 14px", background: "var(--accent)", color: "white",
            border: "none", borderRadius: 6, cursor: loading ? "wait" : "pointer" }}>
          {loading ? "Simulating…" : forecast ? "Riesegui MC" : "Esegui Monte Carlo"}
        </button>
      </div>

      {error && <div className="error" style={{ fontSize: 13 }}>{error}</div>}

      {forecast && (
        <>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
            {forecast.n_paths} paths × {forecast.n_steps} steps · transition matrix da{" "}
            {forecast.transition_matrix_observations} obs · horizon totale{" "}
            {forecast.n_steps * forecast.horizon_days} giorni
            {forecast.notes.length > 0 && (
              <div style={{ color: "var(--deflation)" }}>{forecast.notes.join(" · ")}</div>
            )}
          </div>

          <div style={{ overflowX: "auto", marginBottom: 16 }}>
            <ConeChart bands={forecast.regime_bands} horizonDays={forecast.horizon_days} />
          </div>

          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
            Riepilogo numerico (median ± p10/p90)
          </div>
          <ScrollShadow innerClassName="table-wrap">
            <table className="table" style={{ fontSize: 12 }}>
              <thead>
                <tr>
                  <th>Regime</th>
                  <th className="num">Now</th>
                  <th className="num">+3 step</th>
                  <th className="num">+6 step</th>
                  <th className="num">+12 step</th>
                </tr>
              </thead>
              <tbody>
                {forecast.regime_bands.map((b) => (
                  <tr key={b.regime}>
                    <td>
                      <span style={{ display: "inline-block", width: 8, height: 8,
                        background: REGIME_COLOR[b.regime], borderRadius: 4, marginRight: 6 }} />
                      <strong>{REGIME_LABEL[b.regime]}</strong>
                    </td>
                    {[0, 3, 6, Math.min(12, forecast.n_steps)].map((t) => (
                      <td key={t} className="num">
                        <strong>{(b.median[t] * 100).toFixed(1)}%</strong>
                        <div style={{ fontSize: 10, color: "var(--muted)" }}>
                          [{(b.p10[t] * 100).toFixed(0)}–{(b.p90[t] * 100).toFixed(0)}]
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollShadow>
        </>
      )}

      {/* Shock scenarios */}
      <div style={{ marginTop: 18 }}>
        <h3 style={{ marginBottom: 6 }}>Shock scenarios — what-if analysis</h3>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
          Modifica indicatori correnti e ri-applica il classifier. Vedi come cambiano regime e asset score.
        </div>
        {presets && (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
            {presets.map((p) => (
              <button key={p.key}
                onClick={() => runScenario(p.key)}
                disabled={scenarioLoading}
                title={p.description}
                style={{
                  padding: "5px 10px", fontSize: 11,
                  background: scenarioResult?.scenario_key === p.key ? "var(--accent)" : "var(--bg)",
                  color: scenarioResult?.scenario_key === p.key ? "white" : "var(--text)",
                  border: "1px solid var(--border)", borderRadius: 6,
                  cursor: scenarioLoading ? "wait" : "pointer",
                }}>
                {p.label}
              </button>
            ))}
          </div>
        )}
        {scenarioResult && (
          <div style={{ padding: 12, background: "var(--bg)", borderRadius: 8,
            border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{scenarioResult.label}</div>
            <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 10 }}>
              {scenarioResult.description}
            </div>

            <div style={{ display: "flex", gap: 14, marginBottom: 12, flexWrap: "wrap" }}>
              <div style={{ flex: 1, minWidth: 200 }}>
                <div style={{ fontSize: 11, color: "var(--muted)" }}>Baseline</div>
                <div style={{ fontSize: 18, fontWeight: 700,
                  color: REGIME_COLOR[scenarioResult.baseline_regime] }}>
                  {REGIME_LABEL[scenarioResult.baseline_regime]}
                </div>
                <div style={{ fontSize: 11, color: "var(--muted)" }}>
                  {Object.entries(scenarioResult.baseline_probabilities)
                    .map(([r, p]) => `${REGIME_LABEL[r]?.[0]}: ${(p * 100).toFixed(0)}%`)
                    .join(" · ")}
                </div>
              </div>
              <div style={{ fontSize: 18, alignSelf: "center" }}>→</div>
              <div style={{ flex: 1, minWidth: 200 }}>
                <div style={{ fontSize: 11, color: "var(--muted)" }}>Shocked</div>
                <div style={{ fontSize: 18, fontWeight: 700,
                  color: REGIME_COLOR[scenarioResult.shocked_regime] }}>
                  {REGIME_LABEL[scenarioResult.shocked_regime]}
                </div>
                <div style={{ fontSize: 11, color: "var(--muted)" }}>
                  {Object.entries(scenarioResult.shocked_probabilities)
                    .map(([r, p]) => `${REGIME_LABEL[r]?.[0]}: ${(p * 100).toFixed(0)}%`)
                    .join(" · ")}
                </div>
              </div>
            </div>

            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
              Asset score deltas (top movers)
            </div>
            <ScrollShadow innerClassName="table-wrap">
              <table className="table" style={{ fontSize: 12 }}>
                <thead>
                  <tr>
                    <th>Asset</th>
                    <th className="num">Baseline</th>
                    <th className="num">Shocked</th>
                    <th className="num">Δ score</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(scenarioResult.asset_score_deltas)
                    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                    .slice(0, 8)
                    .map(([asset, delta]) => (
                      <tr key={asset}>
                        <td>{asset.replace(/_/g, " ")}</td>
                        <td className="num">{scenarioResult.baseline_scores[asset]?.toFixed(1)}</td>
                        <td className="num"><strong>{scenarioResult.shocked_scores[asset]?.toFixed(1)}</strong></td>
                        <td className="num" style={{
                          color: delta > 0 ? "var(--reflation)" : "var(--deflation)",
                          fontWeight: 600,
                        }}>
                          {delta >= 0 ? "+" : ""}{delta.toFixed(1)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </ScrollShadow>
          </div>
        )}
      </div>
    </div>
  );
}
