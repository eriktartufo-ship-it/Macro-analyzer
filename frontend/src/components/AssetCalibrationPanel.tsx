import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";
import type { CalibrationDiagnostic, CalibrationPayload } from "../types";

const REGIME_LABEL: Record<string, string> = {
  reflation: "Reflation",
  stagflation: "Stagflation",
  deflation: "Deflation",
  goldilocks: "Goldilocks",
};

function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(digits)}%`;
}

function num(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function deltaColor(delta: number): string {
  if (Math.abs(delta) < 0.02) return "var(--muted)";
  return delta > 0 ? "var(--reflation)" : "var(--deflation)";
}

function formatAsset(name: string): string {
  return name.replace(/_/g, " ");
}

export function AssetCalibrationPanel() {
  const [data, setData] = useState<CalibrationPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<"all" | "high_delta" | "blended" | "measured">("high_delta");

  useEffect(() => {
    let alive = true;
    api
      .assetCalibration()
      .then((d) => {
        if (alive) setData(d);
      })
      .catch((e) => {
        if (alive) setError(e instanceof ApiError ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  const onRun = async () => {
    setLoading(true);
    setError(null);
    try {
      await api.runAssetCalibration();
      const fresh = await api.assetCalibration();
      setData(fresh);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Errore rigenerazione");
    } finally {
      setLoading(false);
    }
  };

  if (error)
    return (
      <div className="card">
        <h2>Asset calibration (Phase 2)</h2>
        <div style={{ color: "var(--deflation)", fontSize: 13, marginBottom: 8 }}>{error}</div>
        <button onClick={onRun} disabled={loading} style={btnStyle}>
          {loading ? "Generazione…" : "Genera calibrazione"}
        </button>
      </div>
    );

  if (!data)
    return (
      <div className="card">
        <h2>Asset calibration (Phase 2)</h2>
        <div className="loading">Caricamento…</div>
      </div>
    );

  const diags = filterDiags(data.diagnostics, filter);
  const sorted = [...diags].sort((a, b) => {
    const da = Math.abs(a.calibrated.avg_return - a.prior.avg_return);
    const db = Math.abs(b.calibrated.avg_return - b.prior.avg_return);
    return db - da;
  });

  return (
    <div className="card">
      <h2>Asset calibration — prior vs measured vs calibrated</h2>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Confronto tra hardcoded (mio prior expert), measured (rendimento reale 12m forward
        deflazionato CPI, dai prezzi Yahoo + bond TR sintetizzato), e calibrated (shrinkage
        Bayesiano peso = min(1, n/{(data.params.n_full as number) || 40})).
        <br />
        Calibrato il <strong>{data.calibrated_on}</strong> su {data.n_classifications} regime
        observations · scoring corrente:{" "}
        <strong style={{ color: data.use_calibrated_scoring ? "var(--reflation)" : "var(--muted)" }}>
          {data.use_calibrated_scoring ? "calibrated ATTIVO" : "hardcoded (calibrazione opt-in)"}
        </strong>
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value as typeof filter)}
          style={{ padding: "4px 8px", fontSize: 12 }}
        >
          <option value="high_delta">Solo |delta| ≥ 5pt</option>
          <option value="all">Tutti</option>
          <option value="blended">Blended (peso 0&lt;w&lt;1)</option>
          <option value="measured">Measured (peso=1)</option>
        </select>
        <button onClick={onRun} disabled={loading} style={btnStyle}>
          {loading ? "Rigenerazione…" : "Rigenera calibrazione"}
        </button>
        <span style={{ fontSize: 12, color: "var(--muted)", alignSelf: "center" }}>
          {sorted.length} righe (di {data.diagnostics.length} totali)
        </span>
      </div>

      <ScrollShadow innerClassName="table-wrap">
        <table className="table" style={{ fontSize: 12 }}>
          <thead>
            <tr>
              <th>Asset</th>
              <th>Regime</th>
              <th className="num">n</th>
              <th className="num">w</th>
              <th className="num" colSpan={2}>real_return</th>
              <th className="num" colSpan={2}>sharpe</th>
              <th className="num" colSpan={2}>hit_rate</th>
            </tr>
            <tr style={{ fontSize: 10, color: "var(--muted)" }}>
              <th></th>
              <th></th>
              <th></th>
              <th></th>
              <th className="num">prior</th>
              <th className="num">calib (Δ)</th>
              <th className="num">prior</th>
              <th className="num">calib (Δ)</th>
              <th className="num">prior</th>
              <th className="num">calib (Δ)</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((d, i) => {
              const dRet = d.calibrated.avg_return - d.prior.avg_return;
              const dSh = d.calibrated.sharpe - d.prior.sharpe;
              const dHit = d.calibrated.hit_rate - d.prior.hit_rate;
              return (
                <tr key={`${d.asset}-${d.regime}-${i}`}>
                  <td>{formatAsset(d.asset)}</td>
                  <td>{REGIME_LABEL[d.regime] ?? d.regime}</td>
                  <td className="num">{d.n_observations}</td>
                  <td className="num">{d.weight_measured.toFixed(2)}</td>
                  <td className="num" style={{ color: "var(--muted)" }}>
                    {pct(d.prior.avg_return)}
                  </td>
                  <td className="num" style={{ color: deltaColor(dRet), fontWeight: 600 }}>
                    {pct(d.calibrated.avg_return)}
                    <div style={{ fontSize: 9, color: deltaColor(dRet) }}>{pct(dRet, 1)}</div>
                  </td>
                  <td className="num" style={{ color: "var(--muted)" }}>{num(d.prior.sharpe)}</td>
                  <td className="num" style={{ color: deltaColor(dSh / 2), fontWeight: 600 }}>
                    {num(d.calibrated.sharpe)}
                    <div style={{ fontSize: 9, color: deltaColor(dSh / 2) }}>
                      {dSh >= 0 ? "+" : ""}
                      {dSh.toFixed(2)}
                    </div>
                  </td>
                  <td className="num" style={{ color: "var(--muted)" }}>{pct(d.prior.hit_rate, 0)}</td>
                  <td className="num" style={{ color: deltaColor(dHit / 2), fontWeight: 600 }}>
                    {pct(d.calibrated.hit_rate, 0)}
                    <div style={{ fontSize: 9, color: deltaColor(dHit / 2) }}>
                      {dHit >= 0 ? "+" : ""}
                      {(dHit * 100).toFixed(0)}pt
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </ScrollShadow>

      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 12 }}>
        <strong>Come leggerla.</strong> n = osservazioni mensili dove prob(regime) ≥{" "}
        {data.params.regime_threshold as number}. w = peso shrinkage del measured (1 = misurato
        intero). Delta colorato verde = calibrazione amplifica l'asset in quel regime, rosso =
        riduce. <br />
        <strong>Limiti noti:</strong> il classifier moderno (post-2020) chiama "stagflation"
        periodi misti che hanno avuto comportamento atipico (es. Nasdaq +30% nel 2023). Il
        sample bias verso anni recenti puo' portare a calibrazioni divergenti dal prior expert.
        Per attivare il calibrato nello scoring impostare env <code>USE_CALIBRATED_SCORING=1</code>{" "}
        e riavviare il backend.
      </div>
    </div>
  );
}

function filterDiags(
  diags: CalibrationDiagnostic[],
  filter: "all" | "high_delta" | "blended" | "measured",
): CalibrationDiagnostic[] {
  switch (filter) {
    case "high_delta":
      return diags.filter(
        (d) => Math.abs(d.calibrated.avg_return - d.prior.avg_return) >= 0.05,
      );
    case "blended":
      return diags.filter((d) => d.weight_measured > 0 && d.weight_measured < 1);
    case "measured":
      return diags.filter((d) => d.weight_measured >= 1);
    default:
      return diags;
  }
}

const btnStyle: React.CSSProperties = {
  padding: "6px 14px",
  background: "var(--accent)",
  color: "white",
  border: "none",
  borderRadius: 6,
  cursor: "pointer",
};
