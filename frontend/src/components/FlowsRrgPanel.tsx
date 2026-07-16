import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { ReliabilityBadge } from "./ReliabilityBadge";
import { ScrollShadow } from "./ScrollShadow";
import type { RrgPoint, RrgQuadrant, RrgResult } from "../types";

const QUAD_COLOR: Record<RrgQuadrant, string> = {
  leading: "#10b981",
  weakening: "#f59e0b",
  lagging: "#ef4444",
  improving: "#06b6d4",
};

const QUAD_LABEL: Record<RrgQuadrant, string> = {
  leading: "Leading",
  weakening: "Weakening",
  lagging: "Lagging",
  improving: "Improving",
};

const QUAD_HINT: Record<RrgQuadrant, string> = {
  leading: "forte e accelera",
  weakening: "forte ma rallenta",
  lagging: "debole e peggiora",
  improving: "debole ma accelera",
};

const WINDOWS = [26, 52, 78];

function RrgChart({ points, hovered, setHovered }: {
  points: RrgPoint[];
  hovered: string | null;
  setHovered: (a: string | null) => void;
}) {
  const W = 640;
  const H = 480;
  const PAD = { top: 24, right: 24, bottom: 34, left: 40 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  // dominio da tutti i punti + tail, centrato a 100, con padding simmetrico
  const { xLo, xHi, yLo, yHi } = useMemo(() => {
    let xmin = 100, xmax = 100, ymin = 100, ymax = 100;
    for (const p of points) {
      for (const t of [...p.tail, { ratio: p.ratio, momentum: p.momentum }]) {
        xmin = Math.min(xmin, t.ratio); xmax = Math.max(xmax, t.ratio);
        ymin = Math.min(ymin, t.momentum); ymax = Math.max(ymax, t.momentum);
      }
    }
    const dx = Math.max(xmax - 100, 100 - xmin, 0.6) * 1.18;
    const dy = Math.max(ymax - 100, 100 - ymin, 0.6) * 1.18;
    return { xLo: 100 - dx, xHi: 100 + dx, yLo: 100 - dy, yHi: 100 + dy };
  }, [points]);

  const xS = (v: number) => PAD.left + ((v - xLo) / (xHi - xLo)) * innerW;
  const yS = (v: number) => PAD.top + (1 - (v - yLo) / (yHi - yLo)) * innerH;
  const cx = xS(100);
  const cy = yS(100);

  // 4 rettangoli-quadrante (soft)
  const quads: { q: RrgQuadrant; x: number; y: number; w: number; h: number }[] = [
    { q: "leading", x: cx, y: PAD.top, w: PAD.left + innerW - cx, h: cy - PAD.top },
    { q: "weakening", x: cx, y: cy, w: PAD.left + innerW - cx, h: PAD.top + innerH - cy },
    { q: "improving", x: PAD.left, y: PAD.top, w: cx - PAD.left, h: cy - PAD.top },
    { q: "lagging", x: PAD.left, y: cy, w: cx - PAD.left, h: PAD.top + innerH - cy },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      style={{ maxWidth: "100%", display: "block", borderRadius: 8 }}
      onMouseLeave={() => setHovered(null)}>
      {/* sfondi quadranti */}
      {quads.map((r) => (
        <rect key={r.q} x={r.x} y={r.y} width={Math.max(0, r.w)} height={Math.max(0, r.h)}
          fill={QUAD_COLOR[r.q]} opacity={0.06} />
      ))}
      {/* etichette quadranti (angoli) */}
      <text x={PAD.left + innerW - 6} y={PAD.top + 14} fontSize="11" fontWeight={600}
        fill={QUAD_COLOR.leading} textAnchor="end" opacity={0.8}>Leading</text>
      <text x={PAD.left + innerW - 6} y={PAD.top + innerH - 6} fontSize="11" fontWeight={600}
        fill={QUAD_COLOR.weakening} textAnchor="end" opacity={0.8}>Weakening</text>
      <text x={PAD.left + 6} y={PAD.top + 14} fontSize="11" fontWeight={600}
        fill={QUAD_COLOR.improving} textAnchor="start" opacity={0.8}>Improving</text>
      <text x={PAD.left + 6} y={PAD.top + innerH - 6} fontSize="11" fontWeight={600}
        fill={QUAD_COLOR.lagging} textAnchor="start" opacity={0.8}>Lagging</text>

      {/* croce centrale a 100 */}
      <line x1={cx} y1={PAD.top} x2={cx} y2={PAD.top + innerH} stroke="var(--border)" strokeDasharray="3 3" />
      <line x1={PAD.left} y1={cy} x2={PAD.left + innerW} y2={cy} stroke="var(--border)" strokeDasharray="3 3" />

      {/* assi */}
      <text x={PAD.left + innerW / 2} y={H - 8} fontSize="10.5" fill="var(--muted)" textAnchor="middle">
        RS-Ratio (forza relativa) →
      </text>
      <text x={12} y={PAD.top + innerH / 2} fontSize="10.5" fill="var(--muted)" textAnchor="middle"
        transform={`rotate(-90 12 ${PAD.top + innerH / 2})`}>
        RS-Momentum ↑
      </text>

      {/* asset: tail + testa (label rese a parte, de-clutterate) */}
      {points.map((p) => {
        const col = QUAD_COLOR[p.quadrant];
        const dim = hovered && hovered !== p.asset;
        const path = p.tail.map((t) => `${xS(t.ratio)},${yS(t.momentum)}`).join(" ");
        return (
          <g key={p.asset} opacity={dim ? 0.15 : 1}
            onMouseEnter={() => setHovered(p.asset)} style={{ cursor: "pointer" }}>
            {p.tail.length > 1 && (
              <polyline points={path} fill="none" stroke={col} strokeWidth={1.4}
                opacity={0.45} strokeLinejoin="round" strokeLinecap="round" />
            )}
            {p.tail.map((t, i) => (
              <circle key={i} cx={xS(t.ratio)} cy={yS(t.momentum)} r={1.5}
                fill={col} opacity={0.2 + 0.5 * (i / Math.max(1, p.tail.length - 1))} />
            ))}
            <circle cx={xS(p.ratio)} cy={yS(p.momentum)} r={hovered === p.asset ? 6 : 4.5}
              fill={col} stroke="var(--bg)" strokeWidth={1.5} />
          </g>
        );
      })}

      {/* label de-clutterate: separazione verticale greedy per lato, con leader-line */}
      {(() => {
        const items = points.map((p) => ({
          asset: p.asset, label: p.label, color: QUAD_COLOR[p.quadrant],
          px: xS(p.ratio), py: yS(p.momentum), side: xS(p.ratio) >= cx ? 1 : 0,
        }));
        const GAP = 12.5;
        for (const side of [0, 1]) {
          const col = items.filter((it) => it.side === side).sort((a, b) => a.py - b.py);
          let lastY = -Infinity;
          for (const it of col) {
            let ly = it.py + 3.5;
            if (ly - lastY < GAP) ly = lastY + GAP;
            (it as { ly?: number }).ly = ly;
            lastY = ly;
          }
        }
        return items.map((it) => {
          const ly = (it as { ly?: number }).ly ?? it.py + 3.5;
          const anchor = it.side === 1 ? "start" : "end";
          const lx = it.side === 1 ? it.px + 8 : it.px - 8;
          const dim = hovered && hovered !== it.asset;
          const moved = Math.abs(ly - (it.py + 3.5)) > 3;
          return (
            <g key={`lbl-${it.asset}`} opacity={dim ? 0.15 : 1}
              onMouseEnter={() => setHovered(it.asset)} style={{ cursor: "pointer" }}>
              {moved && (
                <line x1={it.px} y1={it.py} x2={lx} y2={ly - 3} stroke={it.color}
                  strokeWidth={0.8} opacity={0.4} />
              )}
              <text x={lx} y={ly} fontSize="10.5" textAnchor={anchor}
                fill="var(--text)" fontWeight={hovered === it.asset ? 700 : 500}
                style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 3 }}>
                {it.label}
              </text>
            </g>
          );
        });
      })()}
    </svg>
  );
}

export function FlowsRrgPanel() {
  const [data, setData] = useState<RrgResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [windowWeeks, setWindowWeeks] = useState(52);
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setErr(null);
    api.flowsRrg(windowWeeks, 8)
      .then((r) => { if (alive) { setData(r); setLoading(false); } })
      .catch((e) => { if (alive) { setErr(String(e)); setLoading(false); } });
    return () => { alive = false; };
  }, [windowWeeks]);

  const byQuad = useMemo(() => {
    const m: Record<RrgQuadrant, RrgPoint[]> = { leading: [], weakening: [], lagging: [], improving: [] };
    for (const p of data?.points ?? []) m[p.quadrant].push(p);
    return m;
  }, [data]);

  return (
    <div className="card">
      <h2>Flussi — Relative Rotation Graph</h2>
      <p style={{ color: "var(--muted)", fontSize: 13, marginTop: -4, maxWidth: 640 }}>
        Dove stanno andando i soldi tra settori e asset class, rispetto al basket equal-weight.
        Rotazione oraria: <span style={{ color: QUAD_COLOR.improving }}>Improving</span> →{" "}
        <span style={{ color: QUAD_COLOR.leading }}>Leading</span> →{" "}
        <span style={{ color: QUAD_COLOR.weakening }}>Weakening</span> →{" "}
        <span style={{ color: QUAD_COLOR.lagging }}>Lagging</span>. Segnale informativo, non un trader
        (non batte l'S&P long-only).
      </p>

      {/* segmented finestra */}
      <div className="segmented" style={{ display: "inline-flex", gap: 2, background: "var(--surface-2, var(--bg))",
        border: "1px solid var(--border)", borderRadius: 999, padding: 2, marginBottom: 12 }}>
        {WINDOWS.map((w) => (
          <button key={w} onClick={() => setWindowWeeks(w)}
            style={{
              border: "none", cursor: "pointer", borderRadius: 999, padding: "5px 14px",
              fontSize: 12.5, fontWeight: 600,
              background: windowWeeks === w ? QUAD_COLOR.leading : "transparent",
              color: windowWeeks === w ? "#fff" : "var(--muted)",
            }}>
            {w}w
          </button>
        ))}
      </div>

      {loading && <p style={{ color: "var(--muted)" }}>Calcolo la rotazione…</p>}
      {err && <p style={{ color: "var(--deflation, #ef4444)" }}>Errore: {err}</p>}

      {data && !loading && (
        <>
          <ReliabilityBadge r={data.reliability} />
          <p style={{ color: "var(--muted)", fontSize: 12, margin: "0 0 8px" }}>
            Aggiornato al {data.as_of} · finestra {data.window_weeks} settimane · coda {data.tail} punti
          </p>
          <RrgChart points={data.points} hovered={hovered} setHovered={setHovered} />

          {/* legenda quadranti + conteggi */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, margin: "12px 0" }}>
            {(["leading", "weakening", "lagging", "improving"] as RrgQuadrant[]).map((q) => (
              <div key={q} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5 }}>
                <span style={{ width: 10, height: 10, borderRadius: 3, background: QUAD_COLOR[q] }} />
                <strong style={{ color: "var(--text)" }}>{QUAD_LABEL[q]}</strong>
                <span style={{ color: "var(--muted)" }}>{QUAD_HINT[q]} · {byQuad[q].length}</span>
              </div>
            ))}
          </div>

          {/* tabella compatta (mobile-friendly / accessibile) */}
          <ScrollShadow>
            <table>
              <thead>
                <tr><th>Asset</th><th>Quadrante</th><th className="num">RS-Ratio</th><th className="num">RS-Mom</th></tr>
              </thead>
              <tbody>
                {data.points.map((p) => (
                  <tr key={p.asset}
                    onMouseEnter={() => setHovered(p.asset)} onMouseLeave={() => setHovered(null)}
                    style={{ background: hovered === p.asset ? "var(--surface-2, rgba(127,127,127,0.08))" : undefined }}>
                    <td>{p.label}</td>
                    <td><span style={{ color: QUAD_COLOR[p.quadrant], fontWeight: 600 }}>{QUAD_LABEL[p.quadrant]}</span></td>
                    <td className="num">{p.ratio.toFixed(2)}</td>
                    <td className="num">{p.momentum.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollShadow>
        </>
      )}
    </div>
  );
}
