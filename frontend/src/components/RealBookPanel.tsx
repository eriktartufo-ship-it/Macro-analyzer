import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RealBookMovement, RealBookOverview, RealBookStrategy } from "../types";

/**
 * Libro reale (2026-10-21): lo stesso bot, ma con un portafoglio che si potrebbe tenere davvero
 * (ETF UCITS in euro, quote intere, apertura delle 09:00, commissioni, cedole).
 * Vista pubblica per asset class; strumento, ISIN e quantità arrivano solo col token admin
 * (il backend non li manda proprio, non è il frontend a nasconderli).
 */

type StrategyType = "model_driven" | "llm_driven" | "risk_managed_sp";

const SIDE_IT: Record<string, string> = { buy: "Acquisto", sell: "Vendita", dividend: "Cedola" };
const STATUS_COLOR: Record<string, string> = {
  filled: "var(--reflation)",
  pending: "var(--goldilocks)",
  cancelled: "var(--muted)",
  skipped: "var(--deflation)",
};

function eur(v: number | null | undefined, digits = 2): string {
  if (v == null) return "—";
  return v.toLocaleString("it-IT", { style: "currency", currency: "EUR", minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function pctIt(v: number | null | undefined, digits = 1, sign = false): string {
  if (v == null) return "—";
  const s = (v * 100).toFixed(digits).replace(".", ",");
  return `${sign && v > 0 ? "+" : ""}${s}%`;
}
function signColor(v: number | null | undefined): string {
  if (v == null) return "var(--muted)";
  return v >= 0 ? "var(--reflation)" : "var(--deflation)";
}
function whenFilled(iso: string | null): string {
  if (!iso) return "";
  return new Date(iso).toLocaleString("it-IT", {
    timeZone: "Europe/Rome", weekday: "short", day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}
function day(d: string | null): string {
  if (!d) return "";
  return new Date(`${d}T12:00:00Z`).toLocaleDateString("it-IT", { timeZone: "Europe/Rome", weekday: "short", day: "2-digit", month: "2-digit" });
}
function sourceLabel(src: string | null): string {
  if (!src) return "";
  if (src.startsWith("template")) {
    const g = src.match(/guardia: (.*)\)/);
    return g ? `testo dalle cause della decisione (prosa scartata: ${g[1]})` : "testo dalle cause della decisione";
  }
  return `testo di ${src}, controllato sui fatti`;
}

export function RealBookPanel({ strategy, label, color }: { strategy: StrategyType; label: string; color: string }) {
  const [overview, setOverview] = useState<RealBookOverview | null>(null);
  const [journal, setJournal] = useState<RealBookMovement[]>([]);
  const [showAll, setShowAll] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setError(null);
    setShowAll(false);
    Promise.all([api.realBookOverview(), api.realBookJournal(strategy, 200)])
      .then(([ov, j]) => {
        if (!alive) return;
        setOverview(ov);
        setJournal(j);
      })
      .catch((e) => alive && setError(e instanceof Error ? e.message : "Errore caricamento libro reale"));
    return () => {
      alive = false;
    };
  }, [strategy]);

  const book: RealBookStrategy | null | undefined = overview?.strategies?.[strategy];
  const admin = overview?.admin ?? false;
  const shown = showAll ? journal : journal.slice(0, 15);

  return (
    <div className="card" style={{ marginTop: 12, borderLeft: `3px solid ${color}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
        <h2 style={{ margin: 0, minWidth: 0 }}>
          📒 Libro reale · <span style={{ color }}>{label}</span>
        </h2>
        <span style={{ fontSize: 11, color: "var(--muted)", flex: "0 0 auto" }}>
          {admin ? "con strumenti (admin)" : "vista pubblica, per asset class"}
        </span>
      </div>
      <p style={{ fontSize: 12, lineHeight: 1.5, color: "var(--muted)", margin: "6px 0 10px" }}>
        Lo stesso bot con un portafoglio vero in euro: ETF quotati su Xetra o Milano, quote intere, ordine
        al mercato per l'apertura delle 09:00, commissione 0,99 € e spread stimato, cedole accreditate.
        {book ? <> Partito il {day(book.started_on)} con {eur(book.initial_capital_eur, 0)}.</> : null} Tasse e interessi sulla
        liquidità non sono ancora simulati. Le soglie d'uscita i bot le misurano sull'ETF americano, in dollari: sul nostro
        strumento in euro scattano a livelli un po' diversi.
      </p>

      {error && <div className="error">{error}</div>}
      {!error && overview && !book && (
        <div style={{ fontSize: 13, color: "var(--muted)", fontStyle: "italic" }}>
          Il libro parte al prossimo giro notturno dei bot: prima crea gli ordini, poi li esegue all'apertura.
        </div>
      )}

      {book && (
        <>
          {book.nav ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 8 }}>
              <Tile title="Valore del libro" value={eur(book.nav.nav_eur, 0)} sub={pctIt(book.nav.cumulative_return_pct, 2, true)} subColor={signColor(book.nav.cumulative_return_pct)} />
              <Tile title="Il bot sulla carta" value={pctIt(book.nav.bot_return_pct, 2, true)} valueColor={signColor(book.nav.bot_return_pct)} sub="stesso periodo, in dollari" />
              <Tile title="Costo della realtà" value={book.nav.reality_gap_pct == null ? "—" : `${(book.nav.reality_gap_pct * 100).toFixed(2).replace(".", ",")} pp`} valueColor={signColor(book.nav.reality_gap_pct)} sub="libro − bot di carta" />
              <Tile title={overview?.benchmark_label ?? "S&P 500 in euro"} value={pctIt(book.nav.benchmark_return_pct, 2, true)} valueColor={signColor(book.nav.benchmark_return_pct)} sub="comprato il primo giorno" />
              <Tile title="Costi pagati" value={eur(book.nav.fees_cum_eur + book.nav.spread_cum_eur)} sub={`commissioni ${eur(book.nav.fees_cum_eur)} · spread ${eur(book.nav.spread_cum_eur)}`} />
              <Tile title="Liquidità" value={eur(book.nav.cash_eur, 0)} sub={`${pctIt(book.nav.nav_eur ? book.nav.cash_eur / book.nav.nav_eur : null)} · cedole ${eur(book.nav.dividends_cum_eur)}`} />
            </div>
          ) : (
            <div style={{ fontSize: 13, color: "var(--muted)" }}>
              Primo valore alla chiusura di borsa dopo la prima esecuzione.
              {book.pending_orders > 0 && <> {book.pending_orders} ordini in attesa dell'apertura.</>}
            </div>
          )}

          {book.concentration.map((c) => (
            <div key={c.assets.join("+")} style={{ marginTop: 10, padding: "10px 12px", background: "var(--warn-bg)", borderLeft: "3px solid var(--goldilocks)", borderRadius: 4, fontSize: 12, lineHeight: 1.5 }}>
              ⚠️ <strong>{c.assets.join(" + ")}</strong> finiscono sullo stesso ETF: {pctIt(c.weight_pct)} del libro
              {c.weight_pct > 0.25 ? ", sopra il tetto del 25% per asset" : ""}. Il bot li vede come due asset diversi;
              il libro li copia così come sono.
            </div>
          ))}

          {book.positions.length > 0 && (
            <>
              <h3 style={{ margin: "14px 0 6px", fontSize: 14 }}>Posizioni alla chiusura{book.nav ? ` del ${day(book.nav.date)}` : ""}</h3>
              <div style={{ display: "grid", gap: 6 }}>
                {book.positions.map((p) => (
                  <div key={p.asset_class} style={{ padding: "8px 10px", background: "var(--surface-sunk)", borderRadius: 6, border: "1px solid var(--stroke)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                      <span style={{ fontWeight: 600, fontSize: 13, minWidth: 0 }}>{p.asset_label}</span>
                      <span style={{ fontSize: 13, fontWeight: 600, color: signColor(p.pnl_pct), flex: "0 0 auto" }}>{pctIt(p.pnl_pct, 1, true)}</span>
                    </div>
                    <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>
                      {eur(p.value_eur, 0)} · {pctIt(p.weight_pct)} del libro
                      {p.tranches_total ? ` · tranche ${p.tranches_filled}/${p.tranches_total}` : ""}
                    </div>
                    {admin && p.instrument && (
                      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
                        {p.qty} × {p.instrument.ticker} ({p.instrument.exchange}) · {p.instrument.isin} · {eur(p.price_eur)} (medio {eur(p.avg_price_eur)})
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}

          <h3 style={{ margin: "16px 0 6px", fontSize: 14 }}>Diario delle operazioni</h3>
          {journal.length === 0 ? (
            <div style={{ fontSize: 13, color: "var(--muted)", fontStyle: "italic" }}>Nessuna operazione ancora.</div>
          ) : (
            <div style={{ display: "grid", gap: 8 }}>
              {shown.map((m) => (
                <JournalEntry key={m.id} m={m} admin={admin} />
              ))}
              {journal.length > shown.length && (
                <button className="btn" type="button" onClick={() => setShowAll(true)} style={{ justifySelf: "start" }}>
                  Mostra tutte ({journal.length})
                </button>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Tile({ title, value, sub, valueColor, subColor }: { title: string; value: string; sub?: string; valueColor?: string; subColor?: string }) {
  return (
    <div style={{ padding: "10px 12px", background: "var(--surface-sunk)", border: "1px solid var(--stroke)", borderRadius: 8, minWidth: 0 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>{title}</div>
      <div style={{ fontSize: 18, fontWeight: 700, marginTop: 2, color: valueColor ?? "var(--text)" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: subColor ?? "var(--muted)", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function JournalEntry({ m, admin }: { m: RealBookMovement; admin: boolean }) {
  const when = m.status === "filled" ? whenFilled(m.fill_at) : m.status === "pending" ? "in attesa dell'apertura" : day(m.decided_on);
  const late = m.status === "filled" && m.fill_date && m.fill_date !== m.decided_on && m.side !== "dividend";
  const tranche = m.side === "buy" && m.tranches_total > 0 && m.kind !== "floor" ? ` · tranche ${m.tranche_index} di ${m.tranches_total}` : "";
  return (
    <div style={{ padding: "10px 12px", background: "var(--surface-sunk)", border: "1px solid var(--stroke)", borderRadius: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontSize: 12, color: "var(--muted)", minWidth: 0 }}>{when}</span>
        <span style={{ fontSize: 11, fontWeight: 700, color: STATUS_COLOR[m.status] ?? "var(--muted)", flex: "0 0 auto" }}>
          {m.kind_label}{m.status !== "filled" ? ` · ${m.status_label}` : ""}
        </span>
      </div>
      <div style={{ fontSize: 14, fontWeight: 700, marginTop: 2 }}>
        {SIDE_IT[m.side] ?? m.side} · {m.asset_label}
        <span style={{ fontWeight: 400, color: "var(--muted)" }}>{tranche}</span>
      </div>
      {m.status === "filled" && m.side !== "dividend" && (
        <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>
          {eur(m.gross_eur)} · commissione {eur(m.fee_eur)}
          {m.weight_after_pct != null && m.side === "buy" ? ` · peso dopo ${pctIt(m.weight_after_pct)}` : ""}
          {m.realized_pnl_eur != null && (
            <span style={{ color: signColor(m.realized_pnl_eur) }}> · risultato {eur(m.realized_pnl_eur)} ({pctIt(m.realized_pnl_pct, 1, true)})</span>
          )}
        </div>
      )}
      {admin && m.instrument && m.side !== "dividend" && (
        <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
          {m.quantity} × {m.instrument.ticker} ({m.instrument.exchange}) · {m.instrument.isin}
          {m.fill_price_eur ? ` · a ${eur(m.fill_price_eur, 3)}` : m.est_price_eur ? ` · stimato ${eur(m.est_price_eur, 3)}` : ""}
        </div>
      )}
      {m.narrative && (
        <div style={{ fontSize: 13, lineHeight: 1.5, marginTop: 6 }}>
          <strong>Perché.</strong> {m.narrative}
        </div>
      )}
      {m.exit_plan && (
        <div style={{ fontSize: 12, lineHeight: 1.5, marginTop: 4, color: "var(--muted)" }}>
          <strong>Uscita.</strong> {m.exit_plan}
        </div>
      )}
      <div style={{ fontSize: 11, color: "var(--subtle)", marginTop: 6 }}>
        {late && <>Deciso dal bot {day(m.decided_on)}, eseguito alla prima apertura utile. </>}
        {sourceLabel(m.narrative_source)}
        {admin && m.note ? ` · ${m.note}` : ""}
      </div>
    </div>
  );
}
