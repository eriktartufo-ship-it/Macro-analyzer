/**
 * Scheda di un asset (S54, dalla bozza C approvata da Erik il 2026-10-22).
 * Unisce in una pagina pezzi che prima stavano in 7 posti: score e storia
 * (Asset Ranking), proiezione (Dashboard), comportamento per regime e
 * sensibilità al dollaro (Data), flussi (Flussi), modello Crisi (Dashboard),
 * notizie (Sentiment), bot (AI Portfolio). Quando due modelli non sono d'accordo,
 * li mette uno accanto all'altro.
 */
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type {
  AiPortfolioPosition,
  CrisisRiskAssessment,
  DataSnapshot,
  DebasementRegime,
  DedollarComparison,
  FlowTimeline,
  NewsItem,
  Regime,
  RegimeExplain,
  RegimeHistoryItem,
  RiskOnOff,
  Scoreboard,
  SignalsHistoryItem,
} from "../types";
import {
  assetFamily,
  assetName,
  crisisConflict,
  fmt,
  fmtPct,
  fmtSigned,
  REGIME_LABEL,
  scoreTone,
} from "../lib/assets";
import { to, toAsset, type Route } from "../lib/nav";
import { MultiLineChart } from "./MultiLineChart";
import { RegimeBand } from "./RegimeBand";
import { Icon } from "./Icon";

const STRATS = [
  { id: "model_driven", label: "Modello" },
  { id: "llm_driven", label: "LLM" },
  { id: "risk_managed_sp", label: "S&P a rischio gestito" },
] as const;

const RRG_LABEL: Record<string, string> = {
  leading: "In testa",
  weakening: "In indebolimento",
  lagging: "In coda",
  improving: "In miglioramento",
};

const REGIME_ORDER: Regime[] = ["stagflation", "reflation", "deflation", "goldilocks"];

const CRISIS_TYPE: Record<string, string> = {
  deflation_crash: "crollo deflattivo",
  stagflation_debasement: "stagflazione con perdita di valore della moneta",
  bubble_goldilocks: "bolla in fase Goldilocks",
  no_crisis: "nessun tipo dominante",
};

async function safe<T>(p: Promise<T>): Promise<T | null> {
  try {
    return await p;
  } catch {
    return null;
  }
}

type Tone = "up" | "down" | "mixed" | "flat";

interface Opinion {
  key: string;
  label: string;
  tone: Tone;
  verdict: string;
  detail: string;
}

interface Props {
  asset: string;
  scoreboard: Scoreboard;
  explain: RegimeExplain | null;
  crisis: CrisisRiskAssessment | null;
  news: NewsItem[];
  history: RegimeHistoryItem[];
  navigate: (r: Route) => void;
}

export function AssetDetailPage({ asset, scoreboard, explain, crisis, news, history, navigate }: Props) {
  const [signals, setSignals] = useState<SignalsHistoryItem[]>([]);
  const [snapshot, setSnapshot] = useState<DataSnapshot | null>(null);
  const [ddc, setDdc] = useState<DedollarComparison | null>(null);
  const [flows, setFlows] = useState<FlowTimeline | null>(null);
  const [risk, setRisk] = useState<RiskOnOff | null>(null);
  const [debase, setDebase] = useState<DebasementRegime | null>(null);
  const [positions, setPositions] = useState<Record<string, AiPortfolioPosition[] | null>>({});

  const precious = asset === "gold" || asset === "silver";

  useEffect(() => {
    let alive = true;
    safe(api.signalsHistory(365)).then((v) => alive && setSignals(v ?? []));
    safe(api.dataSnapshot()).then((v) => alive && setSnapshot(v));
    safe(api.dedollarComparison()).then((v) => alive && setDdc(v));
    safe(api.flowsTimeline(52, 8)).then((v) => alive && setFlows(v));
    safe(api.flowsRiskOnOff(13)).then((v) => alive && setRisk(v));
    if (precious) safe(api.cycleDebasement()).then((v) => alive && setDebase(v));
    STRATS.forEach((s) =>
      safe(api.aiPortfolioPositions(s.id)).then((v) => alive && setPositions((p) => ({ ...p, [s.id]: v }))),
    );
    return () => {
      alive = false;
    };
  }, [asset, precious]);

  const ranked = Object.entries(scoreboard.scores).sort((a, b) => b[1] - a[1]);
  const idx = ranked.findIndex(([a]) => a === asset);
  const score = scoreboard.scores[asset];

  if (score == null) {
    return (
      <div className="card">
        <p>Asset non trovato: «{asset}».</p>
        <button className="link" onClick={() => navigate(to("mercati", "classifica"))}>
          Torna alla classifica
        </button>
      </div>
    );
  }

  const proj = explain?.trajectory?.projected_scores?.[asset];
  const projDelta = proj != null ? proj - score : null;
  const regimeStats = snapshot?.asset_regime_data?.[asset] ?? null;
  const current = scoreboard.regime;
  const maxRet = regimeStats ? Math.max(...REGIME_ORDER.map((r) => Math.abs(regimeStats[r]?.avg_return ?? 0)), 0.01) : 1;

  // ---- i pareri degli altri modelli ----
  const opinions: Opinion[] = [];

  const stance = crisis?.positioning?.[asset];
  if (stance) {
    const conflict = crisisConflict(score, stance);
    opinions.push({
      key: "crisi",
      label: "Modello crisi",
      tone: stance === "overweight" ? "up" : stance === "underweight" ? "down" : "flat",
      verdict: stance === "overweight" ? "Sovrappesare" : stance === "underweight" ? "Sottopesare" : "Neutrale",
      detail: conflict
        ? `Lo score del regime lo mette fra i ${stance === "overweight" ? "sfavoriti" : "favoriti"} (${fmt(score)}), il modello Crisi lo ${stance === "overweight" ? "vuole in più" : "ridurrebbe"}: ragiona su come difendersi se arrivasse una crisi da ${CRISIS_TYPE[crisis?.crisis_type ?? ""] ?? "—"}, oggi al ${fmtPct(crisis?.risk_score)}.`
        : `Posizionamento pensato per una crisi da ${CRISIS_TYPE[crisis?.crisis_type ?? ""] ?? "—"}; rischio oggi ${fmtPct(crisis?.risk_score)}.`,
    });
  }

  const lastFrame = flows?.frames?.[flows.frames.length - 1];
  const rrg = lastFrame?.rrg.find((x) => x.a === asset);
  const extra = lastFrame?.flows.find((x) => x.a === asset)?.extra;
  if (rrg || extra != null) {
    const isRefuge = risk?.best_refuge === asset;
    const tone: Tone =
      extra != null && rrg ? ((extra >= 0) === (rrg.q === "leading" || rrg.q === "improving") ? (extra >= 0 ? "up" : "down") : "mixed") : extra != null ? (extra >= 0 ? "up" : "down") : "flat";
    opinions.push({
      key: "flussi",
      label: "Flussi",
      tone,
      verdict: rrg ? RRG_LABEL[rrg.q] ?? rrg.q : extra != null && extra >= 0 ? "Sopra la media" : "Sotto la media",
      detail: [
        rrg ? `Forza relativa ${fmt(rrg.ratio)} (100 = media), slancio ${fmt(rrg.mom)}.` : "",
        extra != null ? `In ${flows?.flow_lookback ?? 8} settimane ha reso ${fmtSigned(extra)} punti rispetto alla media degli asset.` : "",
        isRefuge && risk ? `È il rifugio migliore delle ultime ${risk.lookback_weeks} settimane.` : "",
      ]
        .filter(Boolean)
        .join(" "),
    });
  }

  const dd = ddc?.items.find((x) => x.asset === asset);
  if (dd) {
    const sorted = [...(ddc?.items ?? [])].sort((a, b) => b.sensitivity - a.sensitivity);
    const pos = sorted.findIndex((x) => x.asset === asset) + 1;
    opinions.push({
      key: "dollaro",
      label: "Dollaro",
      tone: dd.sensitivity > 0 ? "up" : dd.sensitivity < 0 ? "down" : "flat",
      verdict:
        dd.sensitivity > 0
          ? pos === 1
            ? "La sensibilità più alta"
            : "Guadagna se il dollaro perde peso"
          : dd.sensitivity < 0
            ? "Perde se il dollaro perde peso"
            : "Indifferente",
      detail: `Sensibilità ${fmtSigned(dd.sensitivity, 0)}. Con il bonus dollaro lo score sarebbe ${fmt(dd.adjusted_score)} invece di ${fmt(dd.pure_score)}; oggi il bonus è ${ddc?.use_dedollar_bonus_active ? "acceso" : "spento"}.`,
    });
  }

  if (precious && debase?.available && debase.divergence?.gap_pct != null && debase.meta) {
    opinions.push({
      key: "debasement",
      label: "Debasement",
      tone: debase.meta.active === "debasement" ? "up" : "flat",
      verdict: `Oro ${fmtSigned(debase.divergence.gap_pct)}% rispetto ai tassi reali`,
      detail: `Il prezzo dell'oro rispetto al livello che i tassi reali (${fmt(debase.divergence.real_yield, 2)}%) giustificherebbero. Debasement ${debase.meta.active === "debasement" ? "attivo" : "non attivo"}, intensità ${Math.round(debase.meta.intensity * 100)} su 100.`,
    });
  }

  const mentions = news.filter((n) => n.affected_assets && asset in n.affected_assets);
  opinions.push({
    key: "notizie",
    label: "Notizie",
    tone: mentions.length === 0 ? "flat" : mentions.reduce((s, n) => s + n.affected_assets[asset], 0) >= 0 ? "up" : "down",
    verdict: mentions.length === 0 ? "Nessuna menzione" : `${mentions.length} notizie lo citano`,
    detail:
      mentions.length === 0
        ? `Nessuna delle ${news.length} notizie analizzate oggi lo riguarda.`
        : mentions
            .slice(0, 3)
            .map((n) => `«${n.title}» (${fmtSigned(n.affected_assets[asset] * 100, 0)}%)`)
            .join(" · "),
  });

  const holders = STRATS.map((s) => ({ ...s, pos: positions[s.id]?.find((p) => p.asset_class === asset) ?? null }));
  const loadedBots = STRATS.every((s) => positions[s.id] !== undefined);
  if (loadedBots) {
    const held = holders.filter((h) => h.pos);
    opinions.push({
      key: "bot",
      label: "Bot",
      tone: held.length ? "up" : "flat",
      verdict: held.length ? `Nel portafoglio di ${held.map((h) => h.label).join(", ")}` : "Nessun bot lo tiene",
      detail: held.length
        ? held.map((h) => `${h.label}: ${fmtPct(h.pos!.current_weight_pct)} del portafoglio, da ${h.pos!.days_held} giorni`).join(" · ")
        : "Nessuna delle tre strategie lo ha in portafoglio oggi.",
    });
  }

  const count = (t: Tone) => opinions.filter((o) => o.tone === t).length;

  const chartPoints = signals.map((h) => ({ date: h.date, values: { [asset]: h.scores[asset] ?? null } }));

  return (
    <div className="asset-page">
      <nav className="crumbs" aria-label="Percorso">
        <button className="back-pill" onClick={() => navigate(to("mercati", "classifica"))}>
          <Icon name="chevron-left" size={16} strokeWidth={2} />
          Classifica
        </button>
        <span className="muted-s">{assetFamily(asset)}</span>
      </nav>

      <header className="asset-head">
        <div className="asset-head-main">
          <h1>{assetName(asset)}</h1>
          <div className="chips">
            <span className="chip">{idx + 1}° su {ranked.length}</span>
            <span className="chip">
              regime: {REGIME_LABEL[current]}
            </span>
            {opinions.length > 0 && (
              <span className="chip">
                {count("up")} a favore · {count("down")} contro · {count("mixed")} {count("mixed") === 1 ? "misto" : "misti"} ·{" "}
                {count("flat")} {count("flat") === 1 ? "neutro" : "neutri"}
              </span>
            )}
          </div>
        </div>
        <div className="asset-head-nums">
          <div>
            <span className="lbl">Score oggi</span>
            <b className={`tone-${scoreTone(score)}`}>{fmt(score)}</b>
          </div>
          {proj != null && (
            <div>
              <span className="lbl">Atteso a 3 mesi</span>
              <b className="asset-proj">
                {fmt(proj)}{" "}
                {projDelta != null && <small className={projDelta >= 0 ? "tone-up" : "tone-down"}>{fmtSigned(projDelta)}</small>}
              </b>
            </div>
          )}
        </div>
      </header>

      <div className="asset-grid">
        <section className="card">
          <div className="card-head">
            <h2 className="h2">Score negli ultimi 12 mesi</h2>
            <span className="muted-s">sotto: il regime di ogni giorno</span>
          </div>
          {chartPoints.length > 1 ? (
            <MultiLineChart
              title={`Score di ${assetName(asset)} negli ultimi 12 mesi`}
              points={chartPoints}
              series={[{ key: asset, label: assetName(asset), color: "var(--accent)" }]}
              yFormat={(v) => v.toFixed(0)}
              height={180}
              compact
              showLegend={false}
            />
          ) : (
            <p className="muted-s">Caricamento della storia…</p>
          )}
          {/* allineata all'area del grafico (margini di MultiLineChart: 44 e 14 su 640) */}
          {history.length > 1 && (
            <div style={{ margin: "0 2.19% 0 6.88%" }}>
              <RegimeBand history={history} height={8} showTicks={false} />
            </div>
          )}
        </section>

        <section className="card">
          <h2 className="h2">Come va in ogni regime</h2>
          <p className="muted-s" style={{ margin: "4px 0 10px" }}>
            Rendimento reale medio annuo (al netto dell'inflazione) e volte in cui ha battuto l'inflazione, dallo storico usato dal modello.
          </p>
          {regimeStats ? (
            <div className="reg-rows">
              <div className="reg-row reg-head" aria-hidden="true">
                <span />
                <span />
                <span>rend. reale</span>
                <span>batte infl.</span>
              </div>
              {REGIME_ORDER.map((r) => {
                const st = regimeStats[r];
                if (!st) return null;
                return (
                  <div key={r} className={`reg-row ${r === current ? "on" : ""}`} style={{ ["--rc" as string]: `var(--${r})` }}>
                    <span className="reg-name">{REGIME_LABEL[r]}</span>
                    <span className="reg-bar" aria-hidden="true">
                      <span style={{ width: `${(Math.max(st.avg_return, 0) / maxRet) * 100}%` }} />
                    </span>
                    <b className={st.avg_return >= 0 ? "" : "tone-down"}>{fmtSigned(st.avg_return * 100)}%</b>
                    <span className="reg-hit">{fmtPct(st.hit_rate)}</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="muted-s">Caricamento…</p>
          )}
        </section>
      </div>

      <h2 className="oggi-h">Cosa dicono gli altri modelli</h2>
      <div className="op-grid">
        {opinions.map((o) => (
          <section key={o.key} className="card op">
            <span className="op-top">
              <span className={`op-dot ${o.tone}`} aria-hidden="true" />
              {o.label}
            </span>
            <b>{o.verdict}</b>
            <p>{o.detail}</p>
          </section>
        ))}
      </div>

      <div className="asset-next">
        {idx > 0 && (
          <button className="link" onClick={() => navigate(toAsset(ranked[idx - 1][0]))}>
            ← {assetName(ranked[idx - 1][0])}
          </button>
        )}
        {idx < ranked.length - 1 && (
          <button className="link" onClick={() => navigate(toAsset(ranked[idx + 1][0]))}>
            {assetName(ranked[idx + 1][0])} →
          </button>
        )}
      </div>
    </div>
  );
}
