import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DataQuality } from "../types";

// S51 (2026-10-13): dopo ogni refresh i numeri del sito si confrontano con le fonti ufficiali
// (BLS, Treasury.gov, FRED, BCE, Yahoo). Qui l'esito: una riga, il dettaglio a richiesta.

const OK = "var(--reflation, #10b981)";
const KO = "var(--deflation, #e5484d)"; // il rosso del tema (errori), non il viola della stagflazione

function quando(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("it-IT", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

export function DataQualityStrip() {
  const [q, setQ] = useState<DataQuality | null>(null);

  useEffect(() => {
    let alive = true;
    api.dataQuality().then((d) => { if (alive) setQ(d); }).catch(() => { /* informativo: se manca, niente */ });
    return () => { alive = false; };
  }, []);

  if (!q || !q.available) return null;
  const totali = q.n_ok + q.n_fail + q.n_na;
  const male = q.n_fail > 0;

  return (
    <details className="card" style={{ padding: "12px 16px" }} open={male}>
      <summary style={{ cursor: "pointer", display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
        listStyle: "none", fontSize: 14 }}>
        <span aria-hidden="true" style={{ width: 10, height: 10, borderRadius: 999, flexShrink: 0,
          background: male ? KO : OK }} />
        <strong style={{ color: "var(--text)" }}>
          {male
            ? `${q.n_fail} ${q.n_fail === 1 ? "numero non torna" : "numeri non tornano"} con la fonte ufficiale`
            : `Dati verificati: ${q.n_ok} su ${totali} con le fonti ufficiali`}
        </strong>
        <span style={{ color: "var(--muted)", fontSize: 12.5 }}>
          controllo del {quando(q.checked_at)}{q.n_na > 0 ? ` · ${q.n_na} non giudicabili (fonte non raggiungibile)` : ""}
        </span>
      </summary>
      <div style={{ marginTop: 10, overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
          <thead>
            <tr style={{ color: "var(--muted)", textAlign: "left", whiteSpace: "nowrap" }}>
              <th style={{ padding: "4px 6px", fontWeight: 600 }}>Indicatore</th>
              <th style={{ padding: "4px 6px", fontWeight: 600, textAlign: "right" }}>Sito</th>
              <th style={{ padding: "4px 6px", fontWeight: 600, textAlign: "right" }}>Fonte</th>
            </tr>
          </thead>
          <tbody>
            {q.results.map((r) => (
              <tr key={r.key} style={{ borderTop: "1px solid var(--border)" }}>
                <td style={{ padding: "5px 6px", color: "var(--text)" }}>
                  <span aria-label={r.status} style={{ marginRight: 6,
                    color: r.status === "ok" ? OK : r.status === "fail" ? KO : "var(--muted)" }}>
                    {r.status === "ok" ? "✓" : r.status === "fail" ? "✗" : "–"}
                  </span>
                  {r.name}
                  <div style={{ fontSize: 11, color: "var(--muted)", marginLeft: 18 }}>{r.source}</div>
                </td>
                <td style={{ padding: "5px 6px", textAlign: "right", fontVariantNumeric: "tabular-nums",
                  whiteSpace: "nowrap", verticalAlign: "top" }}>
                  {r.site === null ? "—" : r.site.toFixed(2)}
                </td>
                <td style={{ padding: "5px 6px", textAlign: "right", fontVariantNumeric: "tabular-nums",
                  whiteSpace: "nowrap", verticalAlign: "top" }}>
                  {r.source_value === null ? "—" : r.source_value.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
