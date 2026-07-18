import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type {
  CycleClock, CycleFrame, CyclePhaseMeta, DebtThermometer,
  DebasementRegime, DebasementIntensityPoint,
} from "../types";

const EASE = "cubic-bezier(.23,1,.32,1)";
const POS = "#10b981";
const NEG = "#ef4444";
const AMBER = "#f59e0b";
const BLUE = "#3f82ff";
const GOLD = "#e0a82e";

// colore per fase (quadrante). Coerente con la palette app: crescita=verde/ambra, stress=rosso, bond=blu.
const PHASE_COLOR: Record<string, string> = {
  recovery: POS, overheat: AMBER, stagflation: NEG, reflation: BLUE,
};
// frasi in chiaro per fase (dove siamo nel ciclo)
const PHASE_SENTENCE: Record<string, string> = {
  recovery: "Fase di ripresa: la crescita accelera con inflazione contenuta. Inizio/early ciclo.",
  overheat: "Surriscaldamento: crescita forte ma inflazione in salita. Ci si avvicina al tardo ciclo.",
  stagflation: "Rallentamento con inflazione alta: tipicamente tardo ciclo / pre-contrazione.",
  reflation: "Contrazione/recessione: crescita negativa e inflazione in calo. Fase di reflation dei bond.",
};

// ---------- Orologio del ciclo (Investment Clock) ----------
function ClockChart({ frames, wi, phaseMeta }: { frames: CycleFrame[]; wi: number; phaseMeta: CyclePhaseMeta[] }) {
  const W = 520, H = 520, PAD = 70;
  const cx = W / 2, cy = H / 2;
  const halfW = (W - PAD * 2) / 2, halfH = (H - PAD * 2) / 2;
  const xS = (x: number) => cx + x * halfW;   // inflazione: -1 sx, +1 dx
  const yS = (y: number) => cy - y * halfH;   // crescita: +1 alto, -1 basso
  const f = frames[wi];
  const nx = xS(f.x), ny = yS(f.y);
  const opacity = Math.max(0.35, f.confidence);

  // scia ultime 8 settimane (rotazione oraria visibile)
  const tail: string[] = [];
  for (let k = Math.max(0, wi - 7); k <= wi; k++) tail.push(`${xS(frames[k].x)},${yS(frames[k].y)}`);

  const meta = (q: string) => phaseMeta.find((p) => p.quadrant === q);
  const corner: Record<string, { lx: number; ly: number; anchor: "start" | "end" }> = {
    tl: { lx: PAD + 6, ly: PAD + 4, anchor: "start" },
    tr: { lx: W - PAD - 6, ly: PAD + 4, anchor: "end" },
    br: { lx: W - PAD - 6, ly: H - PAD - 30, anchor: "end" },
    bl: { lx: PAD + 6, ly: H - PAD - 30, anchor: "start" },
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 560, display: "block", margin: "0 auto", borderRadius: 10 }}>
      {/* zone quadranti (soft) */}
      <rect x={PAD} y={PAD} width={halfW} height={halfH} fill={POS} opacity={0.07} />
      <rect x={cx} y={PAD} width={halfW} height={halfH} fill={AMBER} opacity={0.07} />
      <rect x={cx} y={cy} width={halfW} height={halfH} fill={NEG} opacity={0.07} />
      <rect x={PAD} y={cy} width={halfW} height={halfH} fill={BLUE} opacity={0.07} />
      <rect x={PAD} y={PAD} width={halfW * 2} height={halfH * 2} fill="none" stroke="var(--border)" rx={8} />

      {/* assi */}
      <line x1={cx} y1={PAD} x2={cx} y2={H - PAD} stroke="var(--border)" strokeDasharray="3 3" />
      <line x1={PAD} y1={cy} x2={W - PAD} y2={cy} stroke="var(--border)" strokeDasharray="3 3" />
      <text x={W - PAD} y={cy - 6} fontSize="10.5" fill="var(--muted)" textAnchor="end">inflazione →</text>
      <text x={cx + 6} y={PAD + 12} fontSize="10.5" fill="var(--muted)">↑ crescita</text>
      {/* hint rotazione oraria */}
      <text x={cx} y={H - PAD + 26} fontSize="10.5" fill="var(--muted)" textAnchor="middle">il ciclo ruota in senso orario ⟳</text>

      {/* etichette quadranti: nome ML (grande) + regime (piccolo) */}
      {(["tl", "tr", "br", "bl"] as const).map((q) => {
        const m = meta(q); if (!m) return null;
        const c = corner[q]; const col = PHASE_COLOR[m.key];
        const lead = m.leaders.slice(0, 3).map((l) => l.label).join(" · ");
        return (
          <g key={q}>
            <text x={c.lx} y={c.ly} fontSize="14" fontWeight={800} fill={col} textAnchor={c.anchor}
              style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 3 }}>{m.ml_name}</text>
            <text x={c.lx} y={c.ly + 14} fontSize="9.5" fill="var(--muted)" textAnchor={c.anchor}
              style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 2.5 }}>{m.it}</text>
            <text x={c.lx} y={c.ly + 26} fontSize="8.5" fill="var(--muted)" textAnchor={c.anchor}
              style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 2.5 }}>{lead}</text>
          </g>
        );
      })}

      {/* scia */}
      {tail.length > 1 && (
        <polyline points={tail.join(" ")} fill="none" stroke="var(--muted)" strokeWidth={1.4}
          opacity={0.4} strokeLinejoin="round" strokeLinecap="round" />
      )}

      {/* lancetta (opacità = confidence). Transizione -> alone tratteggiato + fade */}
      <g style={{ transition: `opacity .3s ease` }} opacity={opacity}>
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="var(--text)" strokeWidth={2.4} strokeLinecap="round"
          style={{ transition: `x2 .5s ${EASE}, y2 .5s ${EASE}` }} />
        <g transform={`translate(${nx} ${ny})`} style={{ transition: `transform .5s ${EASE}` }}>
          {f.transition && <circle r={16} fill="none" stroke="var(--muted)" strokeWidth={1.4} strokeDasharray="3 3" opacity={0.7} />}
          <circle r={8} fill={PHASE_COLOR[f.phase]} stroke="var(--bg)" strokeWidth={2} />
        </g>
      </g>
      <circle cx={cx} cy={cy} r={3} fill="var(--muted)" />
    </svg>
  );
}

// ---------- Termometro del ciclo del debito (orizzonte lungo) ----------
// HTML/CSS (no SVG) -> il testo resta a dimensione reale a ogni larghezza (mobile-safe).
function DebtBar({ debt }: { debt: DebtThermometer }) {
  const c = debt.composite;
  const pct = c === null ? null : Math.max(0, Math.min(100, c * 100));
  const stageCol = debt.stage === "late" ? NEG : debt.stage === "mid" ? AMBER : POS;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, fontWeight: 600, marginBottom: 6 }}>
        <span style={{ color: POS }}>◄ inizio ciclo (bassa leva)</span>
        <span style={{ color: NEG }}>tardo ciclo (alta leva) ►</span>
      </div>
      <div style={{ position: "relative", height: 16, borderRadius: 999,
        background: `linear-gradient(90deg, ${POS} 0%, ${AMBER} 50%, ${NEG} 100%)`, opacity: 0.85 }}>
        {pct !== null && (
          <div style={{ position: "absolute", left: `${pct}%`, top: -3, bottom: -3, width: 3,
            transform: "translateX(-50%)", background: "var(--text)", borderRadius: 2,
            boxShadow: "0 0 0 2px var(--bg)", transition: `left .5s ${EASE}` }} />
        )}
      </div>
      {pct !== null && (
        <div style={{ textAlign: "center", marginTop: 8, fontSize: 13, fontWeight: 700, color: stageCol }}>
          {debt.stage_label}{c !== null ? ` · ${Math.round(c * 100)}/100` : ""}
        </div>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 14, marginTop: 12 }}>
        {debt.components.map((cp) => (
          <span key={cp.key} style={{ fontSize: 12, color: cp.available ? "var(--text)" : "var(--muted)" }}>
            {cp.label}: <strong>{cp.available ? `${cp.value}${cp.unit}` : "n/d"}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

// ---------- Overlay asset-leadership: teoria vs osservato ----------
function LeadershipOverlay({ frame, phaseMeta }: { frame: CycleFrame; phaseMeta: CyclePhaseMeta[] }) {
  const phase = phaseMeta.find((p) => p.key === frame.phase);
  const agr = new Set(frame.agreement.map((a) => a.asset));
  const obsAgr = new Set(frame.agreement.map((a) => a.asset));
  const ano = new Set(frame.anomalies.map((a) => a.asset));
  const nAgr = frame.agreement.length, nObs = frame.observed_leaders.length;
  const verdict = nObs === 0
    ? { txt: "RRG senza leader netti in questa settimana.", col: "var(--muted)" }
    : nAgr > 0 && nAgr >= ano.size
      ? { txt: "Il mercato CONFERMA la fase: i leader osservati coincidono con la teoria.", col: POS }
      : { txt: "Il mercato è in DISACCORDO con i fondamentali: guida chi la fase non prevede (anomalia da spiegare).", col: AMBER };
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ fontSize: 13, color: verdict.col, fontWeight: 600 }}>{verdict.txt}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>Dovrebbero guidare (teoria della fase)</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {(phase?.leaders ?? []).map((l) => {
              const hit = agr.has(l.asset);
              return <span key={l.asset} className="chip" style={{
                fontSize: 12, padding: "3px 9px", borderRadius: 999,
                background: hit ? "color-mix(in srgb, #10b981 18%, transparent)" : "var(--surface-2, rgba(127,127,127,.1))",
                border: hit ? `1px solid ${POS}` : "1px solid var(--border)",
                color: hit ? POS : "var(--text)", fontWeight: hit ? 700 : 400,
              }}>{l.label}{hit ? " ✓" : ""}</span>;
            })}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>Guidano ora (RRG osservato)</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {frame.observed_leaders.length === 0 && <span style={{ fontSize: 12, color: "var(--muted)" }}>—</span>}
            {frame.observed_leaders.map((l) => {
              const hit = obsAgr.has(l.asset);
              return <span key={l.asset} style={{
                fontSize: 12, padding: "3px 9px", borderRadius: 999,
                background: hit ? "color-mix(in srgb, #10b981 18%, transparent)" : "color-mix(in srgb, #f59e0b 16%, transparent)",
                border: `1px solid ${hit ? POS : AMBER}`, color: hit ? POS : AMBER, fontWeight: 600,
              }}>{l.label}</span>;
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------- Meta-regime Debasement (DREAM): sparkline intensità + isteresi ----------
function IntensitySpark({ points, on }: { points: DebasementIntensityPoint[]; on: number }) {
  const W = 520, H = 60, PAD = 4;
  if (points.length < 2) return null;
  const n = points.length;
  const xS = (i: number) => PAD + (i / (n - 1)) * (W - 2 * PAD);
  const yS = (v: number) => H - PAD - Math.max(0, Math.min(1, v)) * (H - 2 * PAD);
  const line = points.map((p, i) => `${xS(i)},${yS(p.intensity)}`).join(" ");
  // segmenti "acceso" (debasement confermato) ombreggiati
  const rects: { x: number; w: number }[] = [];
  let start = -1;
  points.forEach((p, i) => {
    if (p.state && start < 0) start = i;
    if (start >= 0 && (!p.state || i === n - 1)) {
      const end = p.state ? i : i - 1;
      rects.push({ x: xS(start), w: Math.max(1.5, xS(end) - xS(start)) });
      start = -1;
    }
  });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }} aria-hidden>
      {rects.map((r, i) => <rect key={i} x={r.x} y={0} width={r.w} height={H} fill={GOLD} opacity={0.13} />)}
      <line x1={PAD} y1={yS(on)} x2={W - PAD} y2={yS(on)} stroke="var(--muted)" strokeDasharray="3 3" opacity={0.55} />
      <polyline points={line} fill="none" stroke={GOLD} strokeWidth={1.8} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={xS(n - 1)} cy={yS(points[n - 1].intensity)} r={3.2} fill={GOLD} stroke="var(--bg)" strokeWidth={1.5} />
    </svg>
  );
}

function ScoreBar({ label, value, color, suffix }: { label: string; value: number; color: string; suffix: string }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div style={{ display: "grid", gap: 3 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
        <span style={{ color: "var(--text)" }}>{label}</span>
        <strong style={{ color, fontVariantNumeric: "tabular-nums" }}>{Math.round(pct)}{suffix}</strong>
      </div>
      <div style={{ height: 6, borderRadius: 999, background: "var(--surface-2, rgba(127,127,127,.14))" }}>
        <div style={{ width: `${pct}%`, height: "100%", borderRadius: 999, background: color, transition: `width .5s ${EASE}` }} />
      </div>
    </div>
  );
}

function DebasementPanel({ deb }: { deb: DebasementRegime }) {
  if (!deb.available || !deb.meta || !deb.scores || !deb.divergence) {
    return (
      <div style={{ fontSize: 12.5, color: "var(--muted)" }}>
        Segnale non disponibile: il feed oro / tassi reali è momentaneamente irraggiungibile.
      </div>
    );
  }
  const { meta, scores, divergence } = deb;
  const on = meta.active === "debasement";
  const gap = divergence.gap_pct ?? 0;
  const LO = -30, HI = 60; // range barra divergenza
  const gapPos = Math.max(0, Math.min(100, ((gap - LO) / (HI - LO)) * 100));
  const zeroPos = ((0 - LO) / (HI - LO)) * 100;
  const subColor = meta.sub_phase === "Late" ? NEG : meta.sub_phase === "Early" ? POS : GOLD;
  const subLabelIt: Record<string, string> = { Early: "iniziale", Mature: "matura", Late: "tardiva" };
  const onThr = deb.thresholds?.on ?? 0.6;

  return (
    <div style={{ display: "grid", gap: 16 }}>
      {/* Badge stato meta-regime */}
      <div style={{
        display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap",
        padding: "12px 14px", borderRadius: 12,
        background: on ? "color-mix(in srgb, #e0a82e 12%, transparent)" : "var(--surface-2, rgba(127,127,127,.08))",
        border: `1px solid ${on ? GOLD : "var(--border)"}`,
      }}>
        <span style={{ fontSize: 22 }}>{on ? "🔶" : "🕐"}</span>
        <div style={{ display: "grid", gap: 2 }}>
          <div style={{ fontSize: 16, fontWeight: 800, color: on ? GOLD : "var(--text)" }}>
            {on ? "Debasement monetario ATTIVO" : "Prevale il ciclo economico"}
            {on && meta.sub_phase && (
              <span style={{ fontSize: 13, fontWeight: 700, color: subColor }}> · fase {subLabelIt[meta.sub_phase] ?? meta.sub_phase.toLowerCase()}</span>
            )}
          </div>
          <div style={{ fontSize: 12, color: "var(--muted)" }}>
            {on
              ? `Il mercato prezza il ciclo monetario più di quello economico da ${meta.months_in} mesi (intensità ${(meta.intensity * 100).toFixed(0)}/100).`
              : `Intensità debasement ${(meta.intensity * 100).toFixed(0)}/100: sotto la soglia di conferma, il ciclo economico resta dominante.`}
          </div>
        </div>
      </div>

      {/* Gauge divergenza oro vs tassi reali */}
      <div style={{ display: "grid", gap: 8 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: 6 }}>
          <span style={{ fontSize: 12.5, color: "var(--muted)" }}>Divergenza oro / tassi reali (rolling 60 mesi)</span>
          <strong style={{ fontSize: 20, fontWeight: 800, color: gap >= 0 ? GOLD : POS, fontVariantNumeric: "tabular-nums" }}>
            {gap >= 0 ? "+" : ""}{gap.toFixed(1)}%
          </strong>
        </div>
        <div style={{ position: "relative", height: 16, borderRadius: 999,
          background: `linear-gradient(90deg, ${POS} 0%, var(--surface-2, rgba(127,127,127,.2)) ${zeroPos}%, ${GOLD} 100%)`, opacity: 0.85 }}>
          {/* tacca zero (fair value) */}
          <div style={{ position: "absolute", left: `${zeroPos}%`, top: -4, bottom: -4, width: 2, transform: "translateX(-50%)",
            background: "var(--muted)", opacity: 0.7 }} />
          {/* marker oro attuale */}
          <div style={{ position: "absolute", left: `${gapPos}%`, top: -3, bottom: -3, width: 3, transform: "translateX(-50%)",
            background: "var(--text)", borderRadius: 2, boxShadow: "0 0 0 2px var(--bg)", transition: `left .5s ${EASE}` }} />
        </div>
        <div style={{ fontSize: 12, color: "var(--text)", lineHeight: 1.45 }}>
          L'oro quota <strong>{gap >= 0 ? "sopra" : "sotto"}</strong> il livello implicito dalla sua relazione storica con i
          tassi reali (10Y reale: <strong>{divergence.real_yield?.toFixed(2)}%</strong>). Un gap alto e persistente è il
          segnale-firma del <em>debasement</em>: il mercato "sente puzza di bruciato" sul sistema monetario.
        </div>
      </div>

      {/* 5 score: 4 cicli economici + debasement (intensità) */}
      <div style={{ display: "grid", gap: 9 }}>
        <div style={{ fontSize: 12.5, color: "var(--muted)" }}>I cinque regimi — 4 <em>cicli economici</em> (probabilità) + il <em>ciclo monetario</em> (intensità, orizzonte diverso)</div>
        <ScoreBar label="Reflation · ciclo" value={scores.reflation} color={BLUE} suffix="%" />
        <ScoreBar label="Stagflation · ciclo" value={scores.stagflation} color={NEG} suffix="%" />
        <ScoreBar label="Goldilocks · ciclo" value={scores.goldilocks} color={POS} suffix="%" />
        <ScoreBar label="Deflation · ciclo" value={scores.deflation} color={AMBER} suffix="%" />
        <ScoreBar label="Debasement · meta (intensità)" value={scores.debasement} color={GOLD} suffix="/100" />
      </div>

      {/* Storia intensità + isteresi */}
      <div style={{ display: "grid", gap: 6 }}>
        <div style={{ fontSize: 12.5, color: "var(--muted)" }}>
          Intensità nel tempo — le zone <span style={{ color: GOLD, fontWeight: 700 }}>oro</span> sono i periodi di debasement
          confermato (isteresi: {deb.thresholds?.confirm_on ?? 2} mesi per accendersi, {deb.thresholds?.confirm_off ?? 6} per spegnersi).
        </div>
        <IntensitySpark points={deb.intensity_history} on={onThr} />
      </div>

      <div style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.5,
        borderTop: "1px solid var(--border)", paddingTop: 10 }}>
        Riproduce il <strong>DREAM Index</strong> di Vito Lops (segnale oro↔tassi reali). Soglie e conferme sono
        euristiche <strong>non calibrate</strong> su backtest. I tassi reali (TIPS) partono dal 2003: i picchi storici
        pre-2003 (es. 1980-81) non sono riproducibili con questi dati. Vista informativa, non un segnale operativo.
      </div>
    </div>
  );
}

function Disclaimers() {
  const items = [
    "Lettura COINCIDENTE su dati rivedibili: dove siamo ORA, non una previsione.",
    "Il 'tardo ciclo' descrive la posizione, non il timing di una recessione: nessun conto alla rovescia.",
    "Mappa di regime MACRO, non di rischio: ignora valutazioni, posizionamento, leva, liquidità.",
    "Relazioni asset↔fase storiche, non garantite (es. correlazione azioni-bond saltata nel 2022).",
    "In transizione la lancetta è incerta (sfumata): probabilità sparse = posizione ballerina.",
  ];
  return (
    <ul style={{ margin: 0, paddingLeft: 18, display: "grid", gap: 5 }}>
      {items.map((t, i) => <li key={i} style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.45 }}>{t}</li>)}
    </ul>
  );
}

export function CyclePage() {
  const [cc, setCc] = useState<CycleClock | null>(null);
  const [deb, setDeb] = useState<DebasementRegime | null>(null);
  const [wi, setWi] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;
    api.cycleClock(52)
      .then((r) => { if (alive) { setCc(r); setWi(Math.max(0, r.frames.length - 1)); } })
      .catch((e) => { if (alive) setErr(String(e)); });
    // Meta-regime Debasement (DREAM): fetch indipendente, non blocca l'orologio se fallisce
    api.cycleDebasement()
      .then((r) => { if (alive) setDeb(r); })
      .catch(() => { if (alive) setDeb(null); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!playing || !cc) return;
    timer.current = window.setInterval(() => setWi((i) => (i + 1 >= cc.frames.length ? 0 : i + 1)), 900);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing, cc]);

  const phaseMeta = useMemo(() => cc?.phase_meta ?? [], [cc]);

  if (err) return <div className="card"><h2>Ciclo</h2><p style={{ color: NEG }}>Errore: {err}</p></div>;
  if (!cc || cc.frames.length === 0)
    return <div className="card"><h2>Ciclo</h2><p style={{ color: "var(--muted)" }}>Calcolo la posizione nel ciclo…</p></div>;

  const frame = cc.frames[wi];
  const phaseCol = PHASE_COLOR[frame.phase];
  const regimeIt = phaseMeta.find((p) => p.key === frame.phase)?.it ?? "";

  return (
    <div style={{ display: "grid", gap: 16 }}>
      {/* Header + timeline globale */}
      <div className="card" style={{ position: "sticky", top: 8, zIndex: 5 }}>
        <h2 style={{ marginBottom: 2 }}>Ciclo economico — in che punto siamo</h2>
        <p style={{ color: "var(--muted)", fontSize: 13, margin: "0 0 12px", maxWidth: 700 }}>
          Orologio del ciclo (Investment Clock): i 4 regimi macro sui quadranti crescita×inflazione.
          Sposta il cursore (o Play) per vedere la <strong>rotazione</strong> nel tempo. Vista informativa, non un segnale d'acquisto.
        </p>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <button onClick={() => setPlaying((p) => !p)}
            style={{ border: "none", cursor: "pointer", borderRadius: 999, padding: "7px 16px", fontSize: 13,
              fontWeight: 700, background: "var(--accent, #3f82ff)", color: "#fff", minWidth: 96 }}>
            {playing ? "❚❚ Pausa" : "▶ Play"}
          </button>
          <input type="range" min={0} max={cc.frames.length - 1} value={wi}
            onChange={(e) => { setPlaying(false); setWi(Number(e.target.value)); }}
            style={{ flex: 1, minWidth: 200, accentColor: "var(--accent, #3f82ff)" }} />
          <div style={{ textAlign: "right", minWidth: 150 }}>
            <div style={{ fontWeight: 700, fontSize: 15, fontVariantNumeric: "tabular-nums" }}>{frame.week}</div>
            <div style={{ fontSize: 11.5, color: frame.transition ? AMBER : "var(--muted)" }}>
              {frame.transition ? "⚠ transizione (incerto)" : `confidence ${Math.round(frame.confidence * 100)}%`}
            </div>
          </div>
        </div>
      </div>

      {/* Orologio + lettura della fase */}
      <div className="card">
        <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr)", gap: 8 }}>
          <ClockChart frames={cc.frames} wi={wi} phaseMeta={phaseMeta} />
          <div style={{ borderTop: "1px solid var(--border)", paddingTop: 12, display: "grid", gap: 6 }}>
            <div style={{ fontSize: 13, color: "var(--muted)" }}>Siamo in:</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: phaseCol }}>
              {frame.phase_ml} <span style={{ fontSize: 14, fontWeight: 600, color: "var(--muted)" }}>· {regimeIt}</span>
            </div>
            <div style={{ fontSize: 13, color: "var(--text)", maxWidth: 640 }}>{PHASE_SENTENCE[frame.phase]}</div>
            {frame.transition && (
              <div style={{ fontSize: 12.5, color: AMBER, background: "color-mix(in srgb, #f59e0b 12%, transparent)",
                border: `1px solid ${AMBER}`, borderRadius: 8, padding: "6px 10px" }}>
                Regime non netto (probabilità sparse): la posizione è in transizione, leggila con cautela.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Overlay asset-leadership */}
      <div className="card">
        <h2 style={{ marginBottom: 2 }}>Classi di prodotto — teoria vs mercato</h2>
        <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: 0 }}>
          Chi <em>dovrebbe</em> guidare in questa fase (mappa storica ML) vs chi <em>guida ora</em> nel RRG. Accordo = conferma; disaccordo = anomalia da spiegare.
        </p>
        <LeadershipOverlay frame={frame} phaseMeta={phaseMeta} />
      </div>

      {/* Termometro del ciclo del debito */}
      <div className="card">
        <h2 style={{ marginBottom: 2 }}>Termometro del ciclo del debito (Dalio)</h2>
        <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: 0, maxWidth: 700 }}>
          Orizzonte <strong>diverso</strong> dall'orologio: quello gira in mesi (ciclo breve), questo in anni (ciclo lungo della leva).
          Per capire se siamo "a fine ciclo" si leggono <strong>insieme</strong>. Contesto strutturale, non timing.
        </p>
        <DebtBar debt={cc.debt} />
      </div>

      {/* Ciclo monetario / Debasement (DREAM) — gemello del termometro debito */}
      <div className="card">
        <h2 style={{ marginBottom: 2 }}>Ciclo monetario / Debasement (DREAM)</h2>
        <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: 0, maxWidth: 700 }}>
          Un <strong>terzo orizzonte</strong>: non il ciclo economico (orologio) né quello della leva (termometro),
          ma il <strong>ciclo monetario</strong>. Misura quando il mercato prezza la messa in discussione del sistema
          monetario (debasement del dollaro) più del normale ciclo. Segnale-chiave: l'oro rispetto ai tassi reali.
        </p>
        {deb ? <DebasementPanel deb={deb} />
          : <p style={{ color: "var(--muted)", fontSize: 12.5 }}>Calcolo il segnale di debasement…</p>}
      </div>

      {/* Disclaimer */}
      <div className="card">
        <h2 style={{ marginBottom: 6, fontSize: 15 }}>Come leggerlo (onestà)</h2>
        <Disclaimers />
      </div>
    </div>
  );
}
