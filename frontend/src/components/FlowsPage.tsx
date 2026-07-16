import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { CrossAssetRiskPanel } from "./CrossAssetRiskPanel";
import type {
  FlowArea, FlowAssetMeta, FlowFrame, FlowTimeline,
} from "../types";

const AREA_COLOR: Record<FlowArea, string> = {
  usa: "#3f82ff", world: "#a855f7", em: "#f59e0b", global: "#14b8a6",
};
const AREA_LABEL: Record<FlowArea, string> = {
  usa: "USA", world: "World ex-US", em: "Emergenti", global: "Globale",
};
const POS = "#10b981";
const NEG = "#ef4444";
const EASE = "cubic-bezier(.23,1,.32,1)";
// transizione fluida per i nodi (movimento tra settimane) — solo transform/opacity (GPU)
const NODE_TR = `transform .45s ${EASE}, opacity .3s ease`;

type MetaMap = Record<string, FlowAssetMeta>;

// ---------- RRG (colore per area, scia 8 settimane, nodi che scorrono) ----------
function RrgChart({ frames, wi, meta, order }: { frames: FlowFrame[]; wi: number; meta: MetaMap; order: string[] }) {
  const W = 640, H = 470, PAD = { t: 22, r: 20, b: 28, l: 34 };
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
  const dom = useMemo(() => {
    let a = 100, b = 100, c = 100, d = 100;
    for (const f of frames) for (const p of f.rrg) {
      a = Math.min(a, p.ratio); b = Math.max(b, p.ratio); c = Math.min(c, p.mom); d = Math.max(d, p.mom);
    }
    const dx = Math.max(b - 100, 100 - a, 0.6) * 1.12, dy = Math.max(d - 100, 100 - c, 0.6) * 1.12;
    return { xLo: 100 - dx, xHi: 100 + dx, yLo: 100 - dy, yHi: 100 + dy };
  }, [frames]);
  const xS = (v: number) => PAD.l + ((v - dom.xLo) / (dom.xHi - dom.xLo)) * iw;
  const yS = (v: number) => PAD.t + (1 - (v - dom.yLo) / (dom.yHi - dom.yLo)) * ih;
  const cx = xS(100), cy = yS(100);
  const cur: Record<string, { ratio: number; mom: number }> = {};
  frames[wi].rrg.forEach((p) => (cur[p.a] = p));

  // de-clutter label per lato, con posizioni del frame corrente
  const present = order.filter((a) => cur[a]);
  const lab: Record<string, { lx: number; ly: number; anchor: string }> = {};
  for (const side of [0, 1]) {
    const colu = present.filter((a) => (xS(cur[a].ratio) >= cx ? 1 : 0) === side)
      .sort((a, b) => yS(cur[a].mom) - yS(cur[b].mom));
    let last = -Infinity;
    for (const a of colu) {
      const px = xS(cur[a].ratio); let ly = yS(cur[a].mom) + 3;
      if (ly - last < 11) ly = last + 11; last = ly;
      lab[a] = { lx: side === 1 ? px + 7 : px - 7, ly, anchor: side === 1 ? "start" : "end" };
    }
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: "100%", display: "block", borderRadius: 8 }}>
      <rect x={cx} y={PAD.t} width={PAD.l + iw - cx} height={cy - PAD.t} fill={POS} opacity={0.05} />
      <rect x={PAD.l} y={cy} width={cx - PAD.l} height={PAD.t + ih - cy} fill={NEG} opacity={0.05} />
      <text x={PAD.l + iw - 6} y={PAD.t + 13} fontSize="10.5" fill="var(--muted)" textAnchor="end">Leading</text>
      <text x={PAD.l + 6} y={PAD.t + ih - 5} fontSize="10.5" fill="var(--muted)">Lagging</text>
      <line x1={cx} y1={PAD.t} x2={cx} y2={PAD.t + ih} stroke="var(--border)" strokeDasharray="3 3" />
      <line x1={PAD.l} y1={cy} x2={PAD.l + iw} y2={cy} stroke="var(--border)" strokeDasharray="3 3" />
      <text x={PAD.l + iw / 2} y={H - 6} fontSize="10" fill="var(--muted)" textAnchor="middle">RS-Ratio (forza) →</text>
      {/* scia 8 settimane (aggiornata per frame, non transizionata) */}
      {present.map((a) => {
        const col = AREA_COLOR[meta[a]?.area ?? "global"];
        const tail: string[] = [];
        for (let k = Math.max(0, wi - 7); k <= wi; k++) {
          const q = frames[k].rrg.find((x) => x.a === a);
          if (q) tail.push(`${xS(q.ratio)},${yS(q.mom)}`);
        }
        return tail.length > 1 ? <polyline key={`t-${a}`} points={tail.join(" ")} fill="none" stroke={col} strokeWidth={1.1} opacity={0.32} strokeLinejoin="round" /> : null;
      })}
      {/* nodi: sempre montati (order), scorrono con transform, fade se assenti */}
      {order.map((a) => {
        const p = cur[a];
        const col = AREA_COLOR[meta[a]?.area ?? "global"];
        const x = p ? xS(p.ratio) : cx, y = p ? yS(p.mom) : cy;
        return (
          <g key={a} transform={`translate(${x} ${y})`} opacity={p ? 1 : 0} style={{ transition: NODE_TR }}>
            <circle r={4.5} fill={col} stroke="var(--bg)" strokeWidth={1.3} />
          </g>
        );
      })}
      {/* label de-clutterate (frame corrente) */}
      {present.map((a) => {
        const l = lab[a]; if (!l) return null;
        return (
          <text key={`l-${a}`} x={l.lx} y={l.ly} fontSize="9.3" textAnchor={l.anchor as "start" | "end"} fill="var(--text)"
            style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 2.5, transition: `x .45s ${EASE}, y .45s ${EASE}` }}>
            {meta[a]?.label ?? a}
          </text>
        );
      })}
    </svg>
  );
}

// ---------- Rete di rotazione: Y FISSA per asset (per risk) -> nessun salto ----------
function NetChart({ frame, meta, order }: { frame: FlowFrame; meta: MetaMap; order: FlowAssetMeta[] }) {
  const W = 640, H = 520, top = 34, bot = 16;
  const xL = 150, xR = W - 150;
  const flow: Record<string, number> = {};
  frame.flows.forEach((f) => (flow[f.a] = f.extra));
  // Y fissa per ogni asset = risk_rank (safe in alto, risky in basso). Non cambia mai -> stabile.
  const yOf = (risk: number) => top + risk * (H - top - bot);
  const maxAbs = Math.max(1e-6, ...frame.flows.map((f) => Math.abs(f.extra)));
  const maxW = Math.max(1e-6, ...frame.edges.map((e) => e.w));
  const r = (a: string) => 4 + (Math.abs(flow[a] ?? 0) / maxAbs) * 10;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: "100%", display: "block" }}>
      <defs>
        <marker id="fp-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0,0 L7,3.5 L0,7 Z" fill="var(--muted)" /></marker>
      </defs>
      <text x={xL} y={20} fontSize="11.5" fontWeight={700} fill={NEG} textAnchor="middle">Cedono →</text>
      <text x={xR} y={20} fontSize="11.5" fontWeight={700} fill={POS} textAnchor="middle">→ Attraggono</text>
      {frame.edges.map((e) => {
        const y1 = yOf(meta[e.s]?.risk ?? 0.5), y2 = yOf(meta[e.t]?.risk ?? 0.5), mx = (xL + xR) / 2;
        const col = AREA_COLOR[meta[e.s]?.area ?? "global"];
        return <path key={`${e.s}-${e.t}`} d={`M ${xL + 12} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${xR - 14} ${y2}`}
          fill="none" stroke={col} strokeWidth={1 + (e.w / maxW) * 10} opacity={0.16 + 0.5 * (e.w / maxW)}
          markerEnd="url(#fp-arrow)" strokeLinecap="round" style={{ transition: "opacity .3s ease" }} />;
      })}
      {order.map((m) => {
        const f = flow[m.asset];
        const isLoser = f !== undefined && f < -1e-9;
        const shown = frame.edges.some((e) => e.s === m.asset || e.t === m.asset);
        const x = isLoser ? xL : xR, y = yOf(m.risk);
        const anchor = isLoser ? "end" : "start";
        const lx = isLoser ? -r(m.asset) - 7 : r(m.asset) + 7;
        return (
          <g key={m.asset} transform={`translate(${x} ${y})`} opacity={shown ? 1 : 0} style={{ transition: NODE_TR }}>
            <circle r={r(m.asset)} fill={AREA_COLOR[m.area]} stroke="var(--bg)" strokeWidth={1.4} />
            <text x={lx} y={3.5} fontSize="10" textAnchor={anchor} fill="var(--text)"
              style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 3 }}>{m.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------- Spettro safe->risky: x FISSO (risk), y=flusso scorre ----------
function SpectrumChart({ frame, assets }: { frame: FlowFrame; assets: FlowAssetMeta[] }) {
  const W = 680, H = 360, PAD = { t: 24, r: 16, b: 40, l: 40 };
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
  const fm: Record<string, { extra: number; cand: boolean }> = {};
  frame.flows.forEach((f) => (fm[f.a] = { extra: f.extra, cand: f.cand }));
  const shown = assets.filter((a) => fm[a.asset] !== undefined);
  const maxAbs = Math.max(1e-6, ...shown.map((a) => Math.abs(fm[a.asset].extra)));
  const xS = (risk: number) => PAD.l + risk * iw;
  const yS = (extra: number) => PAD.t + ih / 2 - (extra / maxAbs) * (ih / 2 - 8);
  const cy = PAD.t + ih / 2;
  // de-clutter label a corsie (frame corrente)
  const placed: { x: number; ly: number }[] = [];
  const lab: Record<string, number> = {};
  [...shown].sort((a, b) => xS(a.risk) - xS(b.risk)).forEach((a) => {
    const x = xS(a.risk), rr = 3.5 + (Math.abs(fm[a.asset].extra) / maxAbs) * 8;
    let ly = yS(fm[a.asset].extra) - rr - 4;
    for (const p of placed) if (Math.abs(p.x - x) < 52 && Math.abs(p.ly - ly) < 11) ly = p.ly - 11;
    ly = Math.max(PAD.t + 6, ly); placed.push({ x, ly }); lab[a.asset] = ly;
  });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: "100%", display: "block", borderRadius: 8 }}>
      <defs>
        <linearGradient id="spec-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#10b981" stopOpacity="0.10" />
          <stop offset="55%" stopColor="#f59e0b" stopOpacity="0.08" />
          <stop offset="100%" stopColor="#ef4444" stopOpacity="0.12" />
        </linearGradient>
      </defs>
      <rect x={PAD.l} y={PAD.t} width={iw} height={ih} fill="url(#spec-grad)" rx={6} />
      <line x1={PAD.l} y1={cy} x2={PAD.l + iw} y2={cy} stroke="var(--border)" strokeDasharray="3 3" />
      <text x={PAD.l} y={H - 10} fontSize="10.5" fontWeight={600} fill="#10b981">◄ SAFE (cash, bond)</text>
      <text x={PAD.l + iw} y={H - 10} fontSize="10.5" fontWeight={600} fill="#ef4444" textAnchor="end">RISKY (bitcoin) ►</text>
      <text x={PAD.l - 6} y={PAD.t + 10} fontSize="9.5" fill={POS} textAnchor="end">flusso +</text>
      <text x={PAD.l - 6} y={PAD.t + ih - 2} fontSize="9.5" fill={NEG} textAnchor="end">flusso −</text>
      {shown.map((a) => {
        const { extra, cand } = fm[a.asset];
        const x = xS(a.risk), y = yS(extra), rr = 3.5 + (Math.abs(extra) / maxAbs) * 8;
        const ly = lab[a.asset];
        return (
          <g key={a.asset}>
            <line x1={x} y1={cy} x2={x} y2={y} stroke={AREA_COLOR[a.area]} strokeWidth={1} opacity={0.3} style={{ transition: `y2 .45s ${EASE}` }} />
            <g transform={`translate(${x} ${y})`} style={{ transition: NODE_TR }}>
              {cand && <circle r={rr + 4} fill="none" stroke="#facc15" strokeWidth={2} opacity={0.9} />}
              <circle r={rr} fill={AREA_COLOR[a.area]} stroke="var(--bg)" strokeWidth={1.2} />
            </g>
            {ly < y - rr - 6 && <line x1={x} y1={y - rr - 2} x2={x} y2={ly + 2} stroke={AREA_COLOR[a.area]} strokeWidth={0.7} opacity={0.4} style={{ transition: `y1 .45s ${EASE}` }} />}
            <text x={x} y={ly} fontSize="8.6" fill="var(--text)" textAnchor="middle"
              style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 2.2, transition: `y .45s ${EASE}` }}>{a.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------- Geografia: flusso netto per area (USA/World/EM/Globale) ----------
function GeoChart({ frame, assets }: { frame: FlowFrame; assets: FlowAssetMeta[] }) {
  const areaOf: Record<string, FlowArea> = {};
  assets.forEach((a) => (areaOf[a.asset] = a.area));
  const net: Record<FlowArea, number> = { usa: 0, world: 0, em: 0, global: 0 };
  frame.flows.forEach((f) => { const ar = areaOf[f.a]; if (ar) net[ar] += f.extra; });
  const areas = (Object.keys(net) as FlowArea[]);
  const maxAbs = Math.max(1e-6, ...areas.map((k) => Math.abs(net[k])));
  const W = 640, rowH = 46, PAD = 90, barMax = (W - PAD * 2) / 2;
  const cx = W / 2, H = areas.length * rowH + 20;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: "100%", display: "block" }}>
      <line x1={cx} y1={10} x2={cx} y2={H - 10} stroke="var(--border)" strokeDasharray="3 3" />
      <text x={cx - 6} y={16} fontSize="9.5" fill={NEG} textAnchor="end">cede</text>
      <text x={cx + 6} y={16} fontSize="9.5" fill={POS} textAnchor="start">attrae</text>
      {areas.map((k, i) => {
        const y = 28 + i * rowH, v = net[k], w = (Math.abs(v) / maxAbs) * barMax, pos = v >= 0;
        return (
          <g key={k}>
            <text x={16} y={y + 4} fontSize="12.5" fontWeight={600} fill="var(--text)">{AREA_LABEL[k]}</text>
            <rect x={pos ? cx : cx - w} y={y - 9} width={w} height={18} rx={4} fill={AREA_COLOR[k]} opacity={0.85}
              style={{ transition: `width .45s ${EASE}, x .45s ${EASE}` }} />
            <text x={pos ? cx + w + 6 : cx - w - 6} y={y + 4} fontSize="12" textAnchor={pos ? "start" : "end"}
              fill={pos ? POS : NEG} fontWeight={600}>{v >= 0 ? "+" : ""}{v.toFixed(1)}</text>
          </g>
        );
      })}
    </svg>
  );
}

function AreaLegend() {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 12, margin: "10px 0 0" }}>
      {(Object.keys(AREA_LABEL) as FlowArea[]).map((k) => (
        <span key={k} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: "var(--muted)" }}>
          <span style={{ width: 10, height: 10, borderRadius: 3, background: AREA_COLOR[k] }} />{AREA_LABEL[k]}
        </span>
      ))}
      <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: "var(--muted)" }}>
        <span style={{ width: 12, height: 12, borderRadius: 999, border: "2px solid #facc15" }} />candidato asimmetrico
      </span>
    </div>
  );
}

export function FlowsPage() {
  const [tl, setTl] = useState<FlowTimeline | null>(null);
  const [wi, setWi] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;
    api.flowsTimeline(52, 8)
      .then((r) => { if (alive) { setTl(r); setWi(r.frames.length - 1); } })
      .catch((e) => { if (alive) setErr(String(e)); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!playing || !tl) return;
    timer.current = window.setInterval(() => setWi((i) => (i + 1 >= tl.frames.length ? 0 : i + 1)), 900);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing, tl]);

  const meta: MetaMap = useMemo(() => {
    const m: MetaMap = {}; (tl?.assets ?? []).forEach((a) => (m[a.asset] = a)); return m;
  }, [tl]);
  // ordine stabile dei nodi RRG = tutti gli asset che appaiono in almeno una frame (montati sempre)
  const rrgOrder = useMemo(() => {
    const s = new Set<string>(); (tl?.frames ?? []).forEach((f) => f.rrg.forEach((p) => s.add(p.a)));
    return (tl?.assets ?? []).map((a) => a.asset).filter((a) => s.has(a));
  }, [tl]);

  if (err) return <div className="card"><h2>Flussi</h2><p style={{ color: NEG }}>Errore: {err}</p></div>;
  if (!tl || tl.frames.length === 0) return <div className="card"><h2>Flussi</h2><p style={{ color: "var(--muted)" }}>Calcolo la timeline dei flussi…</p></div>;

  const frame = tl.frames[wi];
  const rel = frame.sp_vs_sma_pct;
  const relCol = rel === null ? "var(--muted)" : rel >= 0 ? POS : "#f59e0b";

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div className="card" style={{ position: "sticky", top: 8, zIndex: 5 }}>
        <h2 style={{ marginBottom: 2 }}>Flussi — dove si muove il capitale nel tempo</h2>
        <p style={{ color: "var(--muted)", fontSize: 13, margin: "0 0 12px", maxWidth: 680 }}>
          Sposta il cursore (o premi Play) per vedere l'evoluzione settimana per settimana. Colore = area
          geografica. Rotazione relativa, non $. In <strong style={{ color: "#facc15" }}>giallo</strong> i candidati
          asimmetrici: asset safe-ish col flusso negativo esaurito che inizia a girare su.
        </p>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <button onClick={() => setPlaying((p) => !p)}
            style={{ border: "none", cursor: "pointer", borderRadius: 999, padding: "7px 16px", fontSize: 13,
              fontWeight: 700, background: "var(--accent, #3f82ff)", color: "#fff", minWidth: 96, transition: `transform .1s ${EASE}` }}>
            {playing ? "❚❚ Pausa" : "▶ Play"}
          </button>
          <input type="range" min={0} max={tl.frames.length - 1} value={wi}
            onChange={(e) => { setPlaying(false); setWi(Number(e.target.value)); }}
            style={{ flex: 1, minWidth: 200, accentColor: "var(--accent, #3f82ff)" }} />
          <div style={{ textAlign: "right", minWidth: 150 }}>
            <div style={{ fontWeight: 700, fontSize: 15, fontVariantNumeric: "tabular-nums" }}>{frame.week}</div>
            <div style={{ fontSize: 11.5, color: relCol }}>
              {rel === null ? "" : rel >= 0 ? `✓ affidabile (S&P +${rel.toFixed(1)}%)` : `⚠ stress (S&P ${rel.toFixed(1)}%)`}
            </div>
          </div>
        </div>
        <AreaLegend />
      </div>

      <div className="card">
        <h2>Rotation Graph (RRG)</h2>
        <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: -4 }}>I 25 asset nei 4 quadranti · scia = ultime 8 settimane.</p>
        <RrgChart frames={tl.frames} wi={wi} meta={meta} order={rrgOrder} />
      </div>

      <div className="card">
        <h2>Rete di rotazione</h2>
        <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: -4 }}>
          Chi cede forza relativa → chi la attrae. Posizione verticale = rischiosità (safe in alto). Ruotato ≈{frame.rotated_pp.toFixed(1)}pp.
        </p>
        <NetChart frame={frame} meta={meta} order={tl.assets} />
      </div>

      <div className="card">
        <h2>Spettro rischio — dal safe al risky</h2>
        <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: -4 }}>
          Ogni asset alla sua rischiosità (safe → risky), altezza = flusso. Cerchiati in <strong style={{ color: "#facc15" }}>giallo</strong> i candidati asimmetrici.
        </p>
        <SpectrumChart frame={frame} assets={tl.assets} />
      </div>

      <div className="card">
        <h2>Rotazione geografica</h2>
        <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: -4 }}>Flusso netto per area: dove va il capitale tra USA, World ex-US, Emergenti e asset globali.</p>
        <GeoChart frame={frame} assets={tl.assets} />
      </div>

      <CrossAssetRiskPanel />
    </div>
  );
}
