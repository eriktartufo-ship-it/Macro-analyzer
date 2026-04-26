import { useEffect, useState } from "react";
import { api } from "../api/client";
import { AssetCalibrationPanel } from "./AssetCalibrationPanel";
import { BacktestPanel } from "./BacktestPanel";
import { RegimeDynamicsPanel } from "./RegimeDynamicsPanel";
import { RegimeEnsemblePanel } from "./RegimeEnsemblePanel";
import { ScrollShadow } from "./ScrollShadow";
import type { DataSnapshot, Regime } from "../types";

const REGIMES: Regime[] = ["reflation", "stagflation", "deflation", "goldilocks"];

const REGIME_LABELS: Record<Regime, string> = {
  reflation: "Reflation",
  stagflation: "Stagflation",
  deflation: "Deflation",
  goldilocks: "Goldilocks",
};

function formatName(name: string): string {
  return name.replace(/_/g, " ");
}

function formatValue(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

function formatPercent(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function sensitivityColor(v: number): string {
  if (v > 0) return "var(--reflation)";
  if (v < 0) return "var(--deflation)";
  return "var(--muted)";
}

export function DataPage() {
  const [data, setData] = useState<DataSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .dataSnapshot()
      .then((d) => {
        if (alive) setData(d);
      })
      .catch((e) => {
        if (alive) setError(e instanceof Error ? e.message : "Errore caricamento dati");
      });
    return () => {
      alive = false;
    };
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!data) return <div className="loading">Caricamento dati…</div>;

  const macroEntries = Object.entries(data.current_indicators).sort(([a], [b]) =>
    a.localeCompare(b),
  );
  const dedollarEntries = Object.entries(data.current_dedollar_indicators).sort(([a], [b]) =>
    a.localeCompare(b),
  );
  const assetEntries = Object.entries(data.current_asset_scores).sort((a, b) => b[1] - a[1]);
  const sensEntries = Object.entries(data.asset_dedollar_sensitivity).sort(
    (a, b) => b[1] - a[1],
  );
  const assetsInMatrix = Object.keys(data.asset_regime_data);
  const combined = data.current_dedollar_combined ?? 0;

  return (
    <div>
      <div className="card">
        <h2>Data — Formule & Valori Raw</h2>
        <div style={{ fontSize: 13, color: "var(--muted)" }}>
          Tutti i valori grezzi e le formule che determinano regime e asset ranking.
          Aggiornato al {data.date ?? "—"}.
        </div>
      </div>

      <div className="card">
        <h2>Formule principali</h2>
        <div style={{ display: "grid", gap: 14 }}>
          <FormulaBlock
            title="Regime fit score (per regime r)"
            body={
              <>
                <code>
                  fit(r) = Σ<sub>i</sub> weight(c<sub>i</sub>) · eval(c<sub>i</sub>, x<sub>i</sub>) − penalty(r, x)
                </code>
                <p>
                  Dove <em>eval</em> usa una funzione sigmoid o a campana (bell) in base alla condizione.
                  La <em>penalty</em> cross-regime riduce fit se indicatori chiave contraddicono il regime
                  (es. CPI &gt; 3.5% penalizza deflation e goldilocks; GDP &lt; 0 penalizza reflation).
                </p>
              </>
            }
          />
          <FormulaBlock
            title="Regime probabilities"
            body={
              <>
                <code>
                  prob(r) = max(0, fit(r)) / Σ<sub>r</sub> max(0, fit(r))
                </code>
                <p>
                  Normalizzazione con floor a zero: se tutti i fit sono negativi, viene
                  usato il fallback uniforme. Σ prob = 1.0.
                </p>
              </>
            }
          />
          <FormulaBlock
            title="Asset final score"
            body={
              <>
                <code>
                  score(a) = Σ<sub>r</sub> prob(r) · [0.25·hit(a,r) + 0.50·real_ret_norm(a,r) + 0.25·sharpe_norm(a,r)] · 100
                  + secular_bonus(a) + news(a) − penalty(a)
                </code>
                <p>
                  <em>real_ret_norm = clamp((real_ret + 0.30) / 0.60, 0, 1)</em> (range −30% / +30% reale,
                  inflation-adjusted 12m). <em>sharpe_norm = clamp((sharpe + 1) / 3, 0, 1)</em>.
                  Il peso 50% sul real return evita il bug del cash (sharpe falsamente alto in regimi avversi).
                </p>
              </>
            }
          />
          <FormulaBlock
            title="Dedollarization combined score"
            body={
              <>
                <code>
                  combined = 0.35 · cyclical + 0.25 · structural + 0.20 · decade
                  + 0.10 · twenty_year + 0.10 · geopolitical
                </code>
                <p>
                  Ogni componente è 0–1. L'<em>acceleration</em> è la derivata a 30 giorni
                  del combined. Il <em>secular_bonus</em> per asset è
                  <code> sensitivity(a) · combined</code>.
                </p>
              </>
            }
          />
        </div>
      </div>

      <div className="card">
        <h2>Condizioni di regime (pesi)</h2>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
          Somma dei pesi ≠ 1: ogni condizione contribuisce al fit score in base al suo weight.
        </div>
        <div className="grid grid-2">
          {REGIMES.map((r) => {
            const conds = data.regime_conditions[r] ?? [];
            const totalWeight = conds.reduce((acc, c) => acc + c.weight, 0);
            return (
              <div key={r} className="regime-conditions">
                <h3 style={{ marginTop: 0, textTransform: "capitalize" }}>
                  {REGIME_LABELS[r]}
                  <span
                    style={{
                      marginLeft: 8,
                      fontSize: 12,
                      color: "var(--muted)",
                      fontWeight: 400,
                    }}
                  >
                    Σw = {totalWeight.toFixed(2)} — fit={formatValue(data.current_fit_scores[r])} · prob=
                    {formatPercent(data.current_probabilities[r] ?? 0)}
                  </span>
                </h3>
                <ScrollShadow innerClassName="table-wrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Condizione</th>
                        <th className="num">Peso</th>
                        <th>Descrizione</th>
                      </tr>
                    </thead>
                    <tbody>
                      {conds.map((c) => (
                        <tr key={c.name}>
                          <td>{formatName(c.name)}</td>
                          <td className="num">{c.weight.toFixed(2)}</td>
                          <td style={{ fontSize: 12, color: "var(--muted)" }}>
                            {c.description}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </ScrollShadow>
              </div>
            );
          })}
        </div>
      </div>

      <div className="card">
        <h2>Asset × Regime — matrice di performance</h2>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
          Hit rate, rendimento medio 12m, volatilità, Sharpe — dati storici hardcoded
          usati come prior di scoring.
        </div>
        <ScrollShadow innerClassName="table-wrap">
          <table className="table table-responsive">
            <thead>
              <tr>
                <th>Asset</th>
                {REGIMES.map((r) => (
                  <th key={r} colSpan={4} className="num" style={{ textAlign: "center" }}>
                    {REGIME_LABELS[r]}
                  </th>
                ))}
              </tr>
              <tr>
                <th></th>
                {REGIMES.flatMap((r) => [
                  <th key={`${r}-hr`} className="num" style={{ fontSize: 11 }}>
                    Hit%
                  </th>,
                  <th key={`${r}-ret`} className="num" style={{ fontSize: 11 }}>
                    Ret%
                  </th>,
                  <th key={`${r}-vol`} className="num" style={{ fontSize: 11 }}>
                    Vol%
                  </th>,
                  <th key={`${r}-sh`} className="num" style={{ fontSize: 11 }}>
                    Sharpe
                  </th>,
                ])}
              </tr>
            </thead>
            <tbody>
              {assetsInMatrix.map((asset) => {
                const row = data.asset_regime_data[asset];
                return (
                  <tr key={asset}>
                    <td data-label="Asset">{formatName(asset)}</td>
                    {REGIMES.flatMap((r) => {
                      const s = row[r];
                      return [
                        <td key={`${r}-hr`} className="num" data-label={`${r} hit`}>
                          {s ? formatPercent(s.hit_rate) : "—"}
                        </td>,
                        <td key={`${r}-ret`} className="num" data-label={`${r} ret`}>
                          {s ? formatValue(s.avg_return * 100, 1) : "—"}
                        </td>,
                        <td key={`${r}-vol`} className="num" data-label={`${r} vol`}>
                          {s ? formatValue(s.vol * 100, 1) : "—"}
                        </td>,
                        <td key={`${r}-sh`} className="num" data-label={`${r} sharpe`}>
                          {s ? formatValue(s.sharpe, 2) : "—"}
                        </td>,
                      ];
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </ScrollShadow>
      </div>

      <div className="card">
        <h2>Sensibilità alla dedollarizzazione</h2>
        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
          Valori −5 (penalizzato) … +10 (fortemente favorito). Moltiplicato per il combined
          score genera il <em>secular_bonus</em> applicato a ciascun asset.
        </div>
        <ScrollShadow innerClassName="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Asset</th>
                <th className="num">Sensitivity</th>
                <th className="num">Bonus corrente</th>
              </tr>
            </thead>
            <tbody>
              {sensEntries.map(([asset, s]) => (
                <tr key={asset}>
                  <td>{formatName(asset)}</td>
                  <td className="num" style={{ color: sensitivityColor(s), fontWeight: 600 }}>
                    {s > 0 ? `+${s}` : s}
                  </td>
                  <td className="num" style={{ color: sensitivityColor(s) }}>
                    {(s * combined).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollShadow>
      </div>

      <div className="card">
        <h2>Stato corrente</h2>
        <div className="grid grid-2">
          <div>
            <h3 style={{ marginTop: 0 }}>Regime attivo</h3>
            <div style={{ fontSize: 22, fontWeight: 700, textTransform: "capitalize" }}>
              {data.regime ? REGIME_LABELS[data.regime] : "—"}
            </div>
            <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 4 }}>
              Data snapshot: {data.date ?? "—"}
            </div>
            <div style={{ marginTop: 12 }}>
              {REGIMES.map((r) => (
                <div
                  key={r}
                  style={{ display: "flex", justifyContent: "space-between", padding: "4px 0" }}
                >
                  <span style={{ textTransform: "capitalize" }}>{REGIME_LABELS[r]}</span>
                  <span className="num">
                    fit {formatValue(data.current_fit_scores[r])} · prob{" "}
                    {formatPercent(data.current_probabilities[r] ?? 0)}
                  </span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <h3 style={{ marginTop: 0 }}>Dedollarizzazione</h3>
            <div style={{ fontSize: 22, fontWeight: 700 }}>
              {formatValue(data.current_dedollar_combined, 4)}
            </div>
            <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 4 }}>
              Combined score (0–1)
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Asset scores correnti</h2>
        <ScrollShadow innerClassName="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th className="num" style={{ width: 44 }}>
                  #
                </th>
                <th>Asset</th>
                <th className="num">Score</th>
              </tr>
            </thead>
            <tbody>
              {assetEntries.map(([asset, score], i) => (
                <tr key={asset}>
                  <td className="num">{i + 1}</td>
                  <td>{formatName(asset)}</td>
                  <td className="num">{formatValue(score, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollShadow>
      </div>

      <RegimeEnsemblePanel />

      <RegimeDynamicsPanel />

      <AssetCalibrationPanel />

      <BacktestPanel />

      <div className="card">
        <h2>Macro indicators ({macroEntries.length})</h2>
        <ScrollShadow innerClassName="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Indicatore</th>
                <th className="num">Valore</th>
              </tr>
            </thead>
            <tbody>
              {macroEntries.map(([k, v]) => (
                <tr key={k}>
                  <td>{formatName(k)}</td>
                  <td className="num">{formatValue(v, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollShadow>
      </div>

      <div className="card">
        <h2>Dedollarization indicators ({dedollarEntries.length})</h2>
        <ScrollShadow innerClassName="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Indicatore</th>
                <th className="num">Valore</th>
              </tr>
            </thead>
            <tbody>
              {dedollarEntries.map(([k, v]) => (
                <tr key={k}>
                  <td>{formatName(k)}</td>
                  <td className="num">{formatValue(v, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollShadow>
      </div>
    </div>
  );
}

function FormulaBlock({ title, body }: { title: string; body: React.ReactNode }) {
  return (
    <div
      style={{
        padding: 12,
        border: "1px solid var(--border)",
        borderRadius: 8,
        background: "var(--bg)",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{title}</div>
      <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.5 }}>{body}</div>
    </div>
  );
}
