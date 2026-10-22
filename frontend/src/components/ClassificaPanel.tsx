/**
 * Mercati › Classifica (S54): i 24 asset in una lista a rango (convenzione
 * «classifiche = rank-item»), con la variazione attesa del regime proiettato e il
 * segno dove il modello Crisi dice il contrario. Ogni riga apre la scheda asset.
 * Sostituisce AssetRankingTable + «Asset beneficiari del regime proiettato».
 */
import type { CrisisRiskAssessment, RegimeExplain, Scoreboard } from "../types";
import { assetFamily, assetName, crisisConflict, fmt, fmtSigned, SCORE_FAVORED, SCORE_UNFAVORED, scoreTone } from "../lib/assets";
import { toAsset, type Route } from "../lib/nav";

interface Props {
  scoreboard: Scoreboard;
  explain: RegimeExplain | null;
  crisis: CrisisRiskAssessment | null;
  navigate: (r: Route) => void;
}

export function ClassificaPanel({ scoreboard, explain, crisis, navigate }: Props) {
  const ranked = Object.entries(scoreboard.scores).sort((a, b) => b[1] - a[1]);
  const proj = explain?.trajectory?.projected_scores ?? {};
  const max = Math.max(...ranked.map(([, s]) => s), 1);
  const nConflict = ranked.filter(([a, s]) => crisisConflict(s, crisis?.positioning?.[a])).length;

  return (
    <section className="card">
      <div className="card-head">
        <h2 className="h2">Classifica dei {ranked.length} asset nel regime attuale</h2>
        <span className="muted-s">score 0-100, 50 = neutro</span>
      </div>
      <p className="legend-line">
        <span className="tone-up">da {SCORE_FAVORED} in su: favorito</span> ·{" "}
        <span className="tone-down">sotto {SCORE_UNFAVORED}: sfavorito</span> · «atteso» = dove porta il regime
        proiettato a 3 mesi
        {nConflict > 0 && (
          <>
            {" "}
            · <span className="conflict-dot" aria-hidden="true" /> il modello Crisi dice il contrario ({nConflict})
          </>
        )}
      </p>
      <div className="rank-list">
        {ranked.map(([a, s], i) => {
          const p = proj[a];
          const d = p != null ? p - s : null;
          const conflict = crisisConflict(s, crisis?.positioning?.[a]);
          return (
            <button key={a} className="rk rk-full" onClick={() => navigate(toAsset(a))}>
              <span className="rk-n">{i + 1}</span>
              <span className="rk-txt">
                <span className="rk-name">
                  {assetName(a)}
                  {conflict && <span className="conflict-dot" title="Il modello Crisi dice il contrario" />}
                </span>
                <span className="rk-sub">
                  {assetFamily(a)}
                  {d != null && Math.abs(d) >= 0.5 && (
                    <span className={d > 0 ? "tone-up" : "tone-down"}> · atteso {fmtSigned(d)}</span>
                  )}
                </span>
                <span className="rk-bar" aria-hidden="true">
                  <span style={{ width: `${(s / max) * 100}%` }} className={`tone-bg-${scoreTone(s)}`} />
                </span>
              </span>
              <span className={`rk-v tone-${scoreTone(s)}`}>{fmt(s)}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
