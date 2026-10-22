/**
 * I 4 regimi come quadrante crescita × inflazione (S54): stessa lettura dei
 * framework a 4 caselle (crescita e inflazione, su o giù). Unisce le barre di
 * probabilità e il cambio atteso in un solo oggetto.
 */
import type { Regime, RegimeProbabilities } from "../types";
import { fmtSigned, REGIME_LABEL } from "../lib/assets";

// Righe: inflazione su (alto) / giù (basso). Colonne: crescita giù (sx) / su (dx).
const LAYOUT: Regime[] = ["stagflation", "reflation", "deflation", "goldilocks"];

export function RegimeQuadrant({
  current,
  probabilities,
  projected,
  compact = false,
}: {
  current: Regime;
  probabilities: RegimeProbabilities;
  projected?: Record<string, number> | null;
  compact?: boolean;
}) {
  return (
    <div className={`quad ${compact ? "quad-compact" : ""}`}>
      <div className="quad-ax quad-ax-top">Inflazione in salita</div>
      <div className="quad-ax quad-ax-left">Crescita giù</div>
      <div className="quad-grid">
        {LAYOUT.map((r) => {
          const p = probabilities[r] ?? 0;
          const delta = projected?.[r] != null ? (projected[r] - p) * 100 : null;
          const on = r === current;
          return (
            <div key={r} className={`quad-cell ${on ? "on" : ""}`} style={{ ["--rc" as string]: `var(--${r})` }}>
              <span className="quad-name">{REGIME_LABEL[r]}</span>
              <span className="quad-p">{Math.round(p * 100)}%</span>
              {delta != null && Math.abs(delta) >= 0.05 && (
                <span className={`quad-d ${delta > 0 ? "up" : "down"}`}>{fmtSigned(delta)} attesi</span>
              )}
            </div>
          );
        })}
      </div>
      <div className="quad-ax quad-ax-right">Crescita su</div>
      <div className="quad-ax quad-ax-h" aria-hidden="true">
        <span>← crescita giù</span>
        <span>crescita su →</span>
      </div>
      <div className="quad-ax quad-ax-bottom">Inflazione in discesa</div>
    </div>
  );
}
