import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { EuropeEnergy, EuropeEnergyItem } from "../types";

// S50 (2026-10-12): area euro ed energia. Vista informativa dall'ultimo refresh;
// il classificatore usa questi dati solo dietro flag (criteri S50).

const UP = "var(--stagflation, #ef4444)";
const DOWN = "var(--reflation, #10b981)";

function fmt(it: EuropeEnergyItem): string {
  const v = it.value;
  if (v === null || Number.isNaN(v)) return "—";
  const signed = it.unit === "pp" || it.key.endsWith("_yoy") || it.key === "brent_physical_premium";
  const digits = it.unit === "$" || it.unit === "€/MWh" ? 1 : Math.abs(v) >= 100 ? 0 : 2;
  const num = `${signed && v > 0 ? "+" : ""}${v.toFixed(digits)}`;
  if (it.unit === "$") return `${num} $`;
  if (it.unit === "€/MWh") return `${num} €/MWh`;
  return it.unit === "%" ? `${num}%` : `${num} pp`;
}

// Le variazioni si colorano (su = rosso, giu' = verde per inflazione/energia); i livelli no.
function tone(it: EuropeEnergyItem): string {
  if (it.value === null) return "var(--muted)";
  const isChange = it.key.endsWith("_yoy") || it.key.endsWith("_change_6m") || it.key.endsWith("_change_12m");
  if (!isChange || it.key === "ea_hicp_yoy" || it.key.startsWith("ea_core") || it.key.startsWith("ea_services")) {
    return "var(--text)";
  }
  return it.value > 0 ? UP : DOWN;
}

export function EuropeEnergyPanel() {
  const [data, setData] = useState<EuropeEnergy | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api.europeEnergy()
      .then((d) => { if (alive) setData(d); })
      .catch((e) => { if (alive) setErr(e instanceof Error ? e.message : String(e)); });
    return () => { alive = false; };
  }, []);

  if (err) return null; // pannello informativo: se l'endpoint manca, non disturba la dashboard
  if (!data || !data.available) return null;

  const pct = data.sovereign_spread_pctile;

  return (
    <div className="card">
      <h2>Europa ed energia</h2>
      <p style={{ color: "var(--muted)", fontSize: 13, marginTop: -4, maxWidth: 660 }}>
        Area euro e prezzi dell'energia, dalle fonti ufficiali (BCE, ICE via Yahoo, EIA via FRED).
        Aggiornato al <span style={{ whiteSpace: "nowrap" }}>{data.as_of ?? "—"}</span>. Sono dati
        di contesto: non cambiano il regime.
      </p>
      <div style={{ display: "grid", gap: 12,
        gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))" }}>
        {data.groups.map((g) => (
          <div key={g.id} style={{ border: "1px solid var(--border)", borderRadius: 12,
            padding: "10px 12px", background: "var(--surface-2, transparent)" }}>
            <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.3,
              color: "var(--muted)", textTransform: "uppercase", marginBottom: 6 }}>
              {g.title}
            </div>
            {g.items.map((it) => (
              <div key={it.key} style={{ display: "flex", justifyContent: "space-between",
                alignItems: "baseline", gap: 10, padding: "4px 0",
                borderTop: "1px solid var(--border)" }}>
                <span style={{ fontSize: 13, color: "var(--text)", minWidth: 0, flex: 1 }}>{it.label}</span>
                <span style={{ fontSize: 13.5, fontWeight: 600, whiteSpace: "nowrap", flexShrink: 0,
                  fontVariantNumeric: "tabular-nums", color: tone(it) }}>
                  {fmt(it)}
                </span>
              </div>
            ))}
            {g.id === "tassi" && pct !== null && (
              <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 6 }}>
                Spread sovrano = curva di tutti i rating meno AAA: al {pct.toFixed(0)}° percentile dal 2004
                {pct >= 90 ? " — zona di stress" : pct >= 75 ? " — sopra la norma" : ""}.
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
