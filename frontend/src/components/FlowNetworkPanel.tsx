import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { ReliabilityBadge } from "./ReliabilityBadge";
import type { FlowEdge, FlowNetwork, FlowNode } from "../types";

const WIN = "#10b981";
const LOSE = "#ef4444";
const LOOKBACKS = [4, 8, 13, 26];

function NetChart({ net }: { net: FlowNetwork }) {
  const byAsset = useMemo(() => {
    const m: Record<string, FlowNode> = {};
    for (const n of net.nodes) m[n.asset] = n;
    return m;
  }, [net]);

  // solo i nodi coinvolti nelle frecce mostrate -> grafo coerente
  const { losers, winners } = useMemo(() => {
    const src = new Set<string>(), tgt = new Set<string>();
    for (const e of net.edges) { src.add(e.source); tgt.add(e.target); }
    const losers = [...src].map((a) => byAsset[a]).sort((a, b) => a.extra_pp - b.extra_pp);
    const winners = [...tgt].map((a) => byAsset[a]).sort((a, b) => b.extra_pp - a.extra_pp);
    return { losers, winners };
  }, [net, byAsset]);

  const W = 640;
  const rowH = 46;
  const H = Math.max(260, Math.max(losers.length, winners.length) * rowH + 60);
  const PAD = { top: 40, bottom: 20 };
  const xL = 150, xR = W - 150;

  const maxExtra = Math.max(1e-6, ...net.nodes.map((n) => Math.abs(n.extra_pp)));
  const maxW = Math.max(1e-6, ...net.edges.map((e) => e.weight_pp));
  const rNode = (n: FlowNode) => 5 + (Math.abs(n.extra_pp) / maxExtra) * 11;
  const strokeW = (e: FlowEdge) => 1 + (e.weight_pp / maxW) * 11;

  const yFor = (list: FlowNode[], i: number) => {
    const inner = H - PAD.top - PAD.bottom;
    return PAD.top + (list.length === 1 ? inner / 2 : (i / (list.length - 1)) * inner);
  };
  const posL = new Map(losers.map((n, i) => [n.asset, yFor(losers, i)]));
  const posR = new Map(winners.map((n, i) => [n.asset, yFor(winners, i)]));

  const topShares = net.edges.slice(0, 3);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      style={{ maxWidth: "100%", display: "block" }}>
      <defs>
        <marker id="fn-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5"
          orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0,0 L7,3.5 L0,7 Z" fill={WIN} opacity={0.85} />
        </marker>
      </defs>

      <text x={xL} y={22} fontSize="12" fontWeight={700} fill={LOSE} textAnchor="middle">
        Cedono forza →
      </text>
      <text x={xR} y={22} fontSize="12" fontWeight={700} fill={WIN} textAnchor="middle">
        → Attraggono
      </text>

      {/* frecce (bezier), spessore = portata */}
      {net.edges.map((e, k) => {
        const y1 = posL.get(e.source)!, y2 = posR.get(e.target)!;
        const mx = (xL + xR) / 2;
        const d = `M ${xL + 14} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${xR - 16} ${y2}`;
        return (
          <path key={k} d={d} fill="none" stroke={WIN}
            strokeWidth={strokeW(e)} opacity={0.14 + 0.5 * (e.weight_pp / maxW)}
            markerEnd="url(#fn-arrow)" strokeLinecap="round" />
        );
      })}

      {/* etichette % sulle 3 frecce principali */}
      {topShares.map((e, k) => {
        const y1 = posL.get(e.source)!, y2 = posR.get(e.target)!;
        const mx = (xL + xR) / 2, my = (y1 + y2) / 2;
        return (
          <text key={`s${k}`} x={mx} y={my - 3} fontSize="10.5" fontWeight={700}
            fill="var(--text)" textAnchor="middle"
            style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 3 }}>
            {e.share.toFixed(0)}%
          </text>
        );
      })}

      {/* nodi perdenti (sinistra) */}
      {losers.map((n) => {
        const y = posL.get(n.asset)!;
        return (
          <g key={n.asset}>
            <circle cx={xL} cy={y} r={rNode(n)} fill={LOSE} opacity={0.9}
              stroke="var(--bg)" strokeWidth={1.5} />
            <text x={xL - rNode(n) - 8} y={y + 3.5} fontSize="11" textAnchor="end" fill="var(--text)"
              style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 3 }}>
              {n.label} <tspan fill={LOSE} fontWeight={600}>{n.extra_pp.toFixed(1)}</tspan>
            </text>
          </g>
        );
      })}

      {/* nodi vincitori (destra) */}
      {winners.map((n) => {
        const y = posR.get(n.asset)!;
        return (
          <g key={n.asset}>
            <circle cx={xR} cy={y} r={rNode(n)} fill={WIN} opacity={0.9}
              stroke="var(--bg)" strokeWidth={1.5} />
            <text x={xR + rNode(n) + 8} y={y + 3.5} fontSize="11" textAnchor="start" fill="var(--text)"
              style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 3 }}>
              <tspan fill={WIN} fontWeight={600}>+{n.extra_pp.toFixed(1)}</tspan> {n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function FlowNetworkPanel() {
  const [net, setNet] = useState<FlowNetwork | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [lookback, setLookback] = useState(8);

  useEffect(() => {
    let alive = true;
    setLoading(true); setErr(null);
    api.flowsNetwork(lookback, 14)
      .then((r) => { if (alive) { setNet(r); setLoading(false); } })
      .catch((e) => { if (alive) { setErr(String(e)); setLoading(false); } });
    return () => { alive = false; };
  }, [lookback]);

  return (
    <div className="card">
      <h2>Flussi — Rete di rotazione</h2>
      <p style={{ color: "var(--muted)", fontSize: 13, marginTop: -4, maxWidth: 660 }}>
        Dove ruota la forza relativa: le frecce vanno da chi la <strong style={{ color: LOSE }}>cede</strong> a
        chi la <strong style={{ color: WIN }}>attrae</strong>, lo spessore è la portata. I soldi sono
        sempre quelli: ciò che un asset guadagna, un altro lo perde (Σ = 0). Portata in{" "}
        <em>punti percentuali di rotazione relativa</em>, non dollari (i flussi netti in $ non sono
        misurabili dai prezzi).
      </p>

      <div style={{ display: "inline-flex", gap: 2, background: "var(--surface-2, var(--bg))",
        border: "1px solid var(--border)", borderRadius: 999, padding: 2, marginBottom: 12 }}>
        {LOOKBACKS.map((w) => (
          <button key={w} onClick={() => setLookback(w)}
            style={{
              border: "none", cursor: "pointer", borderRadius: 999, padding: "5px 14px",
              fontSize: 12.5, fontWeight: 600,
              background: lookback === w ? WIN : "transparent",
              color: lookback === w ? "#fff" : "var(--muted)",
            }}>
            {w}w
          </button>
        ))}
      </div>

      {loading && <p style={{ color: "var(--muted)" }}>Calcolo la rotazione…</p>}
      {err && <p style={{ color: LOSE }}>Errore: {err}</p>}

      {net && !loading && net.edges.length > 0 && (
        <>
          <ReliabilityBadge r={net.reliability} />
          <p style={{ color: "var(--muted)", fontSize: 12, margin: "0 0 8px" }}>
            Aggiornato al {net.as_of} · <strong style={{ color: "var(--text)" }}>≈{net.rotated_pp.toFixed(1)}pp</strong>{" "}
            di forza relativa ruotata nelle ultime {net.lookback_weeks} settimane ·
            {" "}{net.edges_shown}/{net.edges_total} frecce principali
          </p>
          <NetChart net={net} />
        </>
      )}
      {net && !loading && net.edges.length === 0 && (
        <p style={{ color: "var(--muted)" }}>Nessuna rotazione netta significativa nel periodo.</p>
      )}
    </div>
  );
}
