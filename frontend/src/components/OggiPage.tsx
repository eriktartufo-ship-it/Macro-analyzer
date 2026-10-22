/**
 * Home «Oggi» (S54, bozza A approvata da Erik il 2026-10-22): risponde in ordine
 * a dove siamo, cosa favorisce, perché, cosa tenere d'occhio, quanto ci si può
 * fidare. Tutto il resto è a un tocco, nelle altre sezioni.
 */
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type {
  CrisisRiskAssessment,
  CurrentRegime,
  DataQuality,
  EuropeEnergy,
  MacroLlmAnalysis,
  RegimeExplain,
  RegimeHistoryItem,
  RiskOnOff,
  Scoreboard,
} from "../types";
import {
  assetFamily,
  assetName,
  crisisConflict,
  fmt,
  fmtPct,
  fmtPctSigned,
  fmtSigned,
  REGIME_DESCRIPTION,
  REGIME_LABEL,
  scoreTone,
} from "../lib/assets";
import { to, toAsset, type Route } from "../lib/nav";
import { loadTrackRecord, sealDate, type TrackRecord } from "../lib/trackRecord";
import { RegimeBand, regimeSince } from "./RegimeBand";
import { RegimeQuadrant } from "./RegimeQuadrant";
import { Icon } from "./Icon";

type Leaderboard = Awaited<ReturnType<typeof api.aiPortfolioLeaderboard>>;

async function safe<T>(p: Promise<T>): Promise<T | null> {
  try {
    return await p;
  } catch {
    return null;
  }
}

const CRISIS_LEVEL: Record<string, string> = {
  low: "basso",
  elevated: "elevato",
  moderate: "moderato",
  high: "alto",
  extreme: "estremo",
};

const MONTHS = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"];

function dayMonth(iso: string | null): string {
  if (!iso) return "—";
  const [, m, d] = iso.split("-");
  return `${Number(d)} ${MONTHS[Number(m) - 1] ?? ""}`;
}

function timeAgo(iso: string | undefined): string {
  if (!iso) return "";
  const ms = Date.now() - Date.parse(iso);
  if (!Number.isFinite(ms) || ms < 0) return "";
  const min = Math.round(ms / 60000);
  if (min < 60) return `${min} min fa`;
  const h = Math.round(min / 60);
  if (h < 48) return `${h} ore fa`;
  return `${Math.round(h / 24)} giorni fa`;
}

function energyValue(e: EuropeEnergy | null, key: string): number | null {
  for (const g of e?.groups ?? []) {
    const it = g.items.find((i) => i.key === key);
    if (it) return it.value;
  }
  return null;
}

interface Props {
  regime: CurrentRegime;
  explain: RegimeExplain | null;
  scoreboard: Scoreboard;
  crisis: CrisisRiskAssessment | null;
  history: RegimeHistoryItem[];
  navigate: (r: Route) => void;
}

export function OggiPage({ regime, explain, scoreboard, crisis, history, navigate }: Props) {
  const [llm, setLlm] = useState<MacroLlmAnalysis | null>(null);
  const [llmDone, setLlmDone] = useState(false);
  const [energy, setEnergy] = useState<EuropeEnergy | null>(null);
  const [risk, setRisk] = useState<RiskOnOff | null>(null);
  const [track, setTrack] = useState<TrackRecord | null>(null);
  const [board, setBoard] = useState<Leaderboard | null>(null);
  const [dq, setDq] = useState<DataQuality | null>(null);
  const [side, setSide] = useState<"up" | "down">("up");

  useEffect(() => {
    let alive = true;
    safe(api.macroLlmAnalysis()).then((v) => {
      if (!alive) return;
      setLlm(v);
      setLlmDone(true);
    });
    safe(api.europeEnergy()).then((v) => alive && setEnergy(v));
    safe(api.flowsRiskOnOff(13)).then((v) => alive && setRisk(v));
    safe(loadTrackRecord("3m")).then((v) => alive && setTrack(v));
    safe(api.aiPortfolioLeaderboard()).then((v) => alive && setBoard(v));
    safe(api.dataQuality()).then((v) => alive && setDq(v));
    return () => {
      alive = false;
    };
  }, []);

  const r = regime.regime;
  const p = regime.probabilities[r] ?? 0;
  const traj = explain?.trajectory ?? null;
  const projected = traj?.projected_probabilities ?? null;
  const delta = projected?.[r] != null ? (projected[r] - p) * 100 : null;
  const since = history.length ? regimeSince(history, r) : null;

  let outlook = "e il modello non si aspetta grandi cambiamenti.";
  if (traj && traj.projected_regime && traj.projected_regime !== r) {
    outlook = `ma il modello vede arrivare la ${REGIME_LABEL[traj.projected_regime as keyof typeof REGIME_LABEL]?.toLowerCase() ?? traj.projected_regime}.`;
  } else if (delta != null && delta >= 1) {
    outlook = `e il modello si aspetta che si rafforzi di altri ${fmt(delta)} punti.`;
  } else if (delta != null && delta <= -1) {
    outlook = `ma il modello si aspetta che perda ${fmt(-delta)} punti.`;
  }

  const ranked = Object.entries(scoreboard.scores).sort((a, b) => b[1] - a[1]);
  const top = ranked.slice(0, 5);
  const bottom = ranked.slice(-5).reverse();
  const projScores = traj?.projected_scores ?? {};
  const conflicts = ranked.filter(([a, s]) => crisisConflict(s, crisis?.positioning?.[a])).map(([a]) => a);

  const dqTotal = dq ? dq.n_ok + dq.n_fail + dq.n_na : 0;
  const md = (board?.by_strategy?.model_driven ?? null) as
    | { cumulative_return_pct?: number; benchmark_sp500_return_pct?: number }
    | null;

  const brent = energyValue(energy, "brent_spot");
  const brentYoy = energyValue(energy, "brent_yoy");
  const storage = energyValue(energy, "ea_gas_storage_pct");
  const storageVs = energyValue(energy, "ea_gas_storage_vs_1y");

  const rankRow = (a: string, s: number, i: number) => {
    const proj = projScores[a];
    const d = proj != null ? proj - s : null;
    const tone = scoreTone(s);
    return (
      <button key={a} className="rk" onClick={() => navigate(toAsset(a))}>
        <span className="rk-n">{i}</span>
        <span className="rk-txt">
          <span className="rk-name">{assetName(a)}</span>
          <span className="rk-sub">
            {assetFamily(a)}
            {d != null && Math.abs(d) >= 0.5 && (
              <span className={d > 0 ? "tone-up" : "tone-down"}> · atteso {fmtSigned(d)}</span>
            )}
          </span>
        </span>
        <span className={`rk-v tone-${tone}`}>{fmt(s)}</span>
      </button>
    );
  };

  return (
    <div className="oggi">
      {/* la data dei dati sta già nella sidebar e nella topbar */}
      <div className="oggi-top">
        {dq?.available && dqTotal > 0 && (
          <button
            className={`chip-status ${dq.n_fail > 0 ? "warn" : "ok"}`}
            onClick={() => navigate(to("metodo", "track"))}
          >
            <span className="dot" />
            Dati verificati: {dq.n_ok} su {dqTotal}
          </button>
        )}
      </div>

      <section className="card oggi-hero">
        <div className="oggi-hero-main">
          <span className="lbl">Regime macro</span>
          <h1 className="oggi-regime" style={{ color: `var(--${r})` }}>
            {REGIME_LABEL[r]}
          </h1>
          <p className="oggi-lead">
            {REGIME_DESCRIPTION[r]} È il regime più probabile, al <b>{fmtPct(p)}</b>, {outlook}
          </p>
          <div className="kpi-row">
            <div className="kpi">
              <span>Fiducia del modello</span>
              <b>{fmtPct(regime.confidence)}</b>
            </div>
            <div className="kpi">
              <span>Rischio di cambio</span>
              <b>{traj ? fmtPct(traj.transition_risk) : "—"}</b>
            </div>
            <div className="kpi">
              <span>In vigore dal</span>
              <b>{dayMonth(since)}</b>
            </div>
          </div>
          {history.length > 1 && (
            <div className="oggi-band">
              <div className="oggi-band-head">
                <span>Ultimi 12 mesi</span>
                <button className="link" onClick={() => navigate(to("scenari", "regime"))}>
                  Storia completa
                </button>
              </div>
              <RegimeBand history={history} />
            </div>
          )}
        </div>
        <RegimeQuadrant current={r} probabilities={regime.probabilities} projected={projected} />
      </section>

      <div className="oggi-grid">
        <section className="card oggi-fav">
          <div className="card-head">
            <h2 className="h2">Cosa favorisce questo regime</h2>
            <button className="link" onClick={() => navigate(to("mercati", "classifica"))}>
              Classifica completa ({ranked.length})
            </button>
          </div>
          <div className="segmented fav-switch" role="tablist">
            <button role="tab" aria-selected={side === "up"} className={side === "up" ? "active" : ""} onClick={() => setSide("up")}>
              Più favoriti
            </button>
            <button role="tab" aria-selected={side === "down"} className={side === "down" ? "active" : ""} onClick={() => setSide("down")}>
              Meno favoriti
            </button>
          </div>
          <div className="fav-cols" data-side={side}>
            <div className="fav-col up">
              <div className="fav-h tone-up">Più favoriti</div>
              {top.map(([a, s], i) => rankRow(a, s, i + 1))}
            </div>
            <div className="fav-col down">
              <div className="fav-h tone-down">Meno favoriti</div>
              {bottom.map(([a, s], i) => rankRow(a, s, ranked.length - i))}
            </div>
          </div>
          {conflicts.length > 0 && (
            <button className="note-warn" onClick={() => navigate(to("mercati", "classifica"))}>
              <Icon name="alert" size={18} />
              <span>
                Il modello Crisi la pensa all'opposto su {conflicts.length} asset (
                {conflicts.map((a) => assetName(a)).join(", ")}). Ogni scheda mostra i due pareri.
              </span>
            </button>
          )}
        </section>

        <section className="card oggi-why">
          <div className="card-head">
            <h2 className="h2">Perché</h2>
            {llm && <span className="muted-s">analisi AI · {timeAgo(llm.cached_at)}</span>}
          </div>
          {llm ? (
            <>
              <p className="why-title">{llm.headline}</p>
              <ul className="why-list">
                {llm.key_drivers.slice(0, 3).map((d) => (
                  <li key={d}>{d}</li>
                ))}
              </ul>
              {llm.risks[0] && (
                <div className="why-box risk">
                  <b>Il rischio da guardare:</b> {llm.risks[0]}
                </div>
              )}
              {llm.opportunities[0] && (
                <div className="why-box opp">
                  <b>L'occasione:</b> {llm.opportunities[0]}
                </div>
              )}
            </>
          ) : (
            <p className="muted-s">{llmDone ? "Analisi non disponibile in questo momento." : "Caricamento dell'analisi…"}</p>
          )}
          <button className="link why-more" onClick={() => navigate(to("scenari", "regime"))}>
            Leggi l'analisi completa
          </button>
        </section>
      </div>

      <h2 className="oggi-h">Da tenere d'occhio</h2>
      <div className="watch-grid">
        <button className="card watch" onClick={() => navigate(to("scenari", "crisi"))}>
          <span className="lbl">Rischio crisi</span>
          <span className="watch-v">
            {crisis ? fmtPct(crisis.risk_score) : "—"}
            {crisis && (
              <span className={`watch-tag ${crisis.risk_level === "low" ? "tone-up" : "tone-down"}`}>
                {CRISIS_LEVEL[crisis.risk_level] ?? crisis.risk_level}
              </span>
            )}
          </span>
          <span className="watch-s">
            {crisis
              ? `${crisis.triggers_fired.length} segnali accesi su ${crisis.triggers_all.length}.${crisis.historical_analog ? ` Somiglia a: ${crisis.historical_analog}.` : ""}`
              : "Dato non disponibile."}
          </span>
        </button>
        <button className="card watch" onClick={() => navigate(to("scenari", "europa"))}>
          <span className="lbl">Energia</span>
          <span className="watch-v">
            {brent != null ? `${fmt(brent)} $` : "—"}
            {brentYoy != null && <span className="watch-tag tone-flat">{fmtSigned(brentYoy, 0)}% in un anno</span>}
          </span>
          <span className="watch-s">
            {storage != null
              ? `Brent. Stoccaggi gas UE al ${fmt(storage)}%${storageVs != null ? ` (${fmtSigned(storageVs)} punti in un anno)` : ""}. Contesto: non cambia il regime.`
              : "Dato non disponibile."}
          </span>
        </button>
        <button className="card watch" onClick={() => navigate(to("mercati", "rotazione"))}>
          <span className="lbl">Propensione al rischio · 13 settimane</span>
          <span className="watch-v">
            {risk ? (risk.regime === "risk_on" ? "Risk-on" : risk.regime === "risk_off" ? "Risk-off" : "Neutrale") : "—"}
            {risk && <span className="watch-tag tone-flat">{fmtSigned(risk.gauge)} punti</span>}
          </span>
          <span className="watch-s">
            {risk
              ? `${risk.regime === "risk_on" ? "Il mercato compra rischio." : risk.regime === "risk_off" ? "Il mercato cerca riparo." : "Nessuna direzione netta."}${risk.best_refuge_label ? ` Rifugio migliore: ${risk.best_refuge_label}.` : ""}`
              : "Dato non disponibile."}
          </span>
        </button>
      </div>

      <section className="card oggi-proof">
        <span className="lbl">Il modello finora</span>
        {track && track.active.summary.mean_alpha != null && (
          <span>
            <b className={track.active.summary.mean_alpha >= 0 ? "tone-up" : "tone-down"}>{fmtPctSigned(track.active.summary.mean_alpha, 2)}</b>{" "}
            medio sul 60/40 a 3 mesi · {track.active.summary.n_evaluated} previsioni verificate
          </span>
        )}
        {/* Nuovo sigillo senza ancora verifiche: si dice che è ripartito e si mostra lo storico, etichettato */}
        {track && track.active.summary.n_evaluated === 0 && track.previous && (
          <span>
            Track record ripartito col sigillo del {sealDate(track.activeSnapshot)}
            {track.previous.summary.mean_alpha != null && (
              <>
                {" "}· sigillo precedente{" "}
                <b className={track.previous.summary.mean_alpha >= 0 ? "tone-up" : "tone-down"}>{fmtPctSigned(track.previous.summary.mean_alpha, 2)}</b>{" "}
                sul 60/40 su {track.previous.summary.n_evaluated} previsioni
              </>
            )}
          </span>
        )}
        {md?.cumulative_return_pct != null && (
          <span>
            Bot Modello <b className={md.cumulative_return_pct >= 0 ? "tone-up" : "tone-down"}>{fmtPctSigned(md.cumulative_return_pct, 2)}</b>
            {md.benchmark_sp500_return_pct != null && <> contro S&amp;P 500 {fmtPctSigned(md.benchmark_sp500_return_pct, 2)}</>}
          </span>
        )}
        <span className="proof-links">
          <button className="link" onClick={() => navigate(to("metodo", "track"))}>
            Metodo
          </button>
          <button className="link" onClick={() => navigate(to("portafogli", "bot"))}>
            Portafogli
          </button>
        </span>
      </section>
    </div>
  );
}
