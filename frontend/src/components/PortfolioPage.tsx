/** Portafogli personali — cockpit gestionale a prezzi di mercato + semaforo macro.
 *
 * Mobile-first, "nonno-proof": login semplice, hero col totale + KPI,
 * barra di allocazione, semaforo verde/giallo/rosso per capire se è un buon
 * momento per accumulare. Desktop = 2 colonne (contenuto + rail analisi),
 * stile gestionale come il cockpit CognitiveOS ma con l'identità Macro.
 * La strategia (card dedicata) appare solo per gli utenti che ce l'hanno
 * (per Erik la scrive Claude Code via canale admin).
 */
import { useCallback, useEffect, useState } from "react";
import {
  PtApiError,
  getToken,
  ptApi,
  setToken,
  type PtAdvice,
  type PtPortfolio,
  type PtPosition,
  type PtStrategy,
  type PtTransaction,
  type PtTransactionIn,
} from "../api/pt";

const eur = (v: number | null | undefined, digits = 0) =>
  v == null
    ? "—"
    : v.toLocaleString("it-IT", {
        style: "currency",
        currency: "EUR",
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });

const num = (v: number, max = 4) =>
  v.toLocaleString("it-IT", { maximumFractionDigits: max });

const pnlClass = (v: number | null | undefined) =>
  v == null ? "" : v >= 0 ? "pt-pos" : "pt-neg";

const LIGHT_LABEL: Record<string, string> = {
  green: "Momento favorevole",
  yellow: "Neutro",
  red: "Meglio aspettare",
};

const ASSET_ORDER: ("gold" | "silver" | "bitcoin")[] = ["gold", "silver", "bitcoin"];
const ASSET_LABEL = { gold: "Oro", silver: "Argento", bitcoin: "Bitcoin" } as const;

/** Colori segmenti allocazione: semantici per metalli/crypto (token regime),
 * sfumature di accent per i ticker generici. Solo token del design system. */
function segmentColor(p: PtPosition, tickerIndex: number): { background: string; opacity?: number } {
  if (p.asset_key === "gold") return { background: "var(--goldilocks)" };
  if (p.asset_key === "silver") return { background: "var(--subtle)" };
  if (p.asset_key === "bitcoin") return { background: "var(--stagflation)" };
  const fades = [1, 0.7, 0.45, 0.28];
  return { background: "var(--accent)", opacity: fades[tickerIndex % fades.length] };
}

function segmentStyles(positions: PtPosition[]): Map<string, { background: string; opacity?: number }> {
  const map = new Map<string, { background: string; opacity?: number }>();
  let tickerIdx = 0;
  for (const p of positions) {
    if (p.asset_key === "gold" || p.asset_key === "silver" || p.asset_key === "bitcoin") {
      map.set(p.asset_key, segmentColor(p, 0));
    } else {
      map.set(p.asset_key, segmentColor(p, tickerIdx));
      tickerIdx += 1;
    }
  }
  return map;
}

interface Preset {
  asset_key: string;
  label: string;
  units: string[];
}

const PRESETS: Preset[] = [
  { asset_key: "gold", label: "Oro", units: ["g", "oz"] },
  { asset_key: "silver", label: "Argento", units: ["g", "oz"] },
  { asset_key: "bitcoin", label: "Bitcoin", units: ["btc"] },
  { asset_key: "custom", label: "Altro (ticker)", units: ["share"] },
];

const UNIT_LABEL: Record<string, string> = {
  g: "grammi",
  oz: "once",
  btc: "BTC",
  share: "quote",
  unit: "quote",
};

export function PortfolioPage() {
  const [session, setSession] = useState<{ username: string; display_name: string } | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    (async () => {
      if (!getToken()) {
        setChecking(false);
        return;
      }
      try {
        const me = await ptApi.me();
        setSession(me);
        // sessione rolling: rinnova il token a ogni apertura
        ptApi.refreshToken().then((r) => setToken(r.token)).catch(() => undefined);
      } catch {
        setToken(null);
      } finally {
        setChecking(false);
      }
    })();
  }, []);

  if (checking) return <div className="loading">Caricamento…</div>;
  if (!session) return <LoginCard onLogin={setSession} />;
  return (
    <PortfolioDashboard
      displayName={session.display_name}
      onLogout={() => {
        setToken(null);
        setSession(null);
      }}
    />
  );
}

function LoginCard({
  onLogin,
}: {
  onLogin: (s: { username: string; display_name: string }) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const s = await ptApi.login(username, password);
      setToken(s.token);
      onLogin({ username: s.username, display_name: s.display_name });
    } catch (err) {
      setError(err instanceof PtApiError ? err.message : "Errore di rete, riprova");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="pt-login-wrap">
      <form className="card pt-login pt-rise" onSubmit={submit}>
        <h2>Il mio portafoglio</h2>
        <p className="pt-login-sub">Entra per vedere i tuoi investimenti</p>
        <label className="pt-field">
          <span>Email</span>
          <input
            type="email"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            autoCapitalize="none"
            inputMode="email"
            required
          />
        </label>
        <label className="pt-field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </label>
        {error && <div className="pt-form-error">{error}</div>}
        <button className="btn pt-btn-full" disabled={busy}>
          {busy ? "Un attimo…" : "Entra"}
        </button>
      </form>
    </div>
  );
}

function PortfolioDashboard({
  displayName,
  onLogout,
}: {
  displayName: string;
  onLogout: () => void;
}) {
  const [portfolio, setPortfolio] = useState<PtPortfolio | null>(null);
  const [transactions, setTransactions] = useState<PtTransaction[]>([]);
  const [advice, setAdvice] = useState<PtAdvice | null>(null);
  const [strategy, setStrategy] = useState<PtStrategy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [pf, txs] = await Promise.all([ptApi.portfolio(), ptApi.transactions()]);
      setPortfolio(pf);
      setTransactions(txs.transactions);
    } catch (err) {
      if (err instanceof PtApiError && err.status === 401) return onLogout();
      setError("Non riesco a caricare il portafoglio, riprova tra poco");
    } finally {
      setLoading(false);
    }
    ptApi.latestAdvice().then((r) => setAdvice(r.advice)).catch(() => undefined);
    ptApi.strategy().then((r) => setStrategy(r.strategy)).catch(() => undefined);
  }, [onLogout]);

  useEffect(() => {
    load();
  }, [load]);

  const onDelete = async (id: number) => {
    if (!window.confirm("Cancellare questo movimento?")) return;
    try {
      await ptApi.deleteTransaction(id);
      await load();
    } catch {
      setError("Cancellazione non riuscita, riprova");
    }
  };

  if (loading) return <div className="loading">Caricamento portafoglio…</div>;

  return (
    <div className="pt-page">
      <div className="pt-topbar">
        <h2>Ciao, {displayName}</h2>
        <button className="btn-ghost" onClick={onLogout}>
          Esci
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="pt-grid">
        <div className="pt-area-hero pt-rise">
          {portfolio && <HeroCard portfolio={portfolio} />}
        </div>

        <div className="pt-area-side">
          <div className="pt-rise pt-rise-1">
            <AdviceCard advice={advice} onRefreshed={setAdvice} />
          </div>
          {strategy && portfolio && (
            <div className="pt-rise pt-rise-2">
              <StrategyCard strategy={strategy} portfolio={portfolio} />
            </div>
          )}
        </div>

        <div className="pt-area-positions pt-rise pt-rise-2">
          {portfolio && portfolio.positions.length > 0 && (
            <PositionsCard portfolio={portfolio} />
          )}
        </div>

        <div className="pt-area-form pt-rise pt-rise-3">
          {showForm ? (
            <TransactionForm
              onSaved={async () => {
                setShowForm(false);
                await load();
              }}
              onCancel={() => setShowForm(false)}
            />
          ) : (
            <button className="btn pt-btn-full pt-add" onClick={() => setShowForm(true)}>
              + Aggiungi movimento
            </button>
          )}
        </div>

        <div className="pt-area-tx pt-rise pt-rise-3">
          {transactions.length > 0 && (
            <TransactionsCard transactions={transactions} onDelete={onDelete} />
          )}
        </div>
      </div>
    </div>
  );
}

function HeroCard({ portfolio }: { portfolio: PtPortfolio }) {
  const t = portfolio.totals;
  const styles = segmentStyles(portfolio.positions);
  const allocated = portfolio.positions.filter(
    (p) => p.allocation_pct != null && p.allocation_pct > 0,
  );

  return (
    <div className="card pt-hero">
      <div className="pt-totals-label">Valore totale</div>
      <div className="pt-totals-value">{eur(t.market_value_eur)}</div>
      {t.pnl_eur != null && (
        <div className={`pt-hero-badge ${pnlClass(t.pnl_eur)}`}>
          {t.pnl_eur >= 0 ? "▲" : "▼"} {eur(Math.abs(t.pnl_eur))}
          {t.pnl_pct != null && ` (${t.pnl_pct >= 0 ? "+" : ""}${num(t.pnl_pct, 1)}%)`}
        </div>
      )}

      <div className="pt-kpis">
        <div className="pt-kpi">
          <div className="v">{eur(t.invested_eur)}</div>
          <div className="l">Investiti</div>
        </div>
        <div className="pt-kpi">
          <div className={`v ${pnlClass(t.pnl_eur)}`}>
            {t.pnl_eur != null ? `${t.pnl_eur >= 0 ? "+" : ""}${eur(t.pnl_eur)}` : "—"}
          </div>
          <div className="l">Guadagno</div>
          {t.realized_pnl_eur !== 0 && (
            <div className="hint">di cui incassati {eur(t.realized_pnl_eur)}</div>
          )}
        </div>
        <div className="pt-kpi">
          <div className={`v ${pnlClass(t.pnl_pct)}`}>
            {t.pnl_pct != null ? `${t.pnl_pct >= 0 ? "+" : ""}${num(t.pnl_pct, 1)}%` : "—"}
          </div>
          <div className="l">Rendimento</div>
        </div>
        <div className="pt-kpi">
          <div className="v">{portfolio.positions.length}</div>
          <div className="l">Posizioni</div>
        </div>
      </div>

      {allocated.length > 0 && (
        <>
          <div className="pt-alloc-bar" role="img" aria-label="Ripartizione del portafoglio">
            {allocated.map((p) => (
              <div
                key={p.asset_key}
                className="pt-alloc-seg"
                style={{ width: `${p.allocation_pct}%`, ...styles.get(p.asset_key) }}
                title={`${p.label} ${num(p.allocation_pct!, 1)}%`}
              />
            ))}
          </div>
          <div className="pt-alloc-legend">
            {allocated.map((p) => (
              <span className="pt-chip" key={p.asset_key}>
                <span className="pt-chip-dot" style={styles.get(p.asset_key)} />
                {p.label} <strong>{num(p.allocation_pct!, 1)}%</strong>
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function AdviceCard({
  advice,
  onRefreshed,
}: {
  advice: PtAdvice | null;
  onRefreshed: (a: PtAdvice | null) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await ptApi.refreshAdvice();
      onRefreshed(r.advice);
    } catch (err) {
      setMsg(err instanceof PtApiError ? err.message : "Aggiornamento non riuscito");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card pt-advice">
      <h3>Conviene comprare adesso?</h3>
      {!advice && (
        <p className="pt-muted">
          Nessuna analisi disponibile ancora. Premi "Aggiorna analisi" per generarla.
        </p>
      )}
      {advice && (
        <>
          {ASSET_ORDER.map((key) => {
            const a = advice.assets[key];
            if (!a) return null;
            return (
              <details className="pt-light-row" key={key}>
                <summary>
                  <span className={`pt-dot pt-dot-${a.light}`} />
                  <span className="pt-light-asset">{ASSET_LABEL[key]}</span>
                  <span className="pt-light-verdict">{LIGHT_LABEL[a.light]}</span>
                </summary>
                <div className="pt-light-detail">
                  <p className="pt-light-headline">{a.headline}</p>
                  <ul>
                    {a.reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                  {a.what_to_watch && (
                    <p className="pt-watch">👁 Da tenere d'occhio: {a.what_to_watch}</p>
                  )}
                </div>
              </details>
            );
          })}
          {advice.macro_summary && (
            <p className="pt-macro-summary">{advice.macro_summary}</p>
          )}
          <div className="pt-advice-meta">
            Aggiornata il{" "}
            {new Date(advice.generated_at + "Z").toLocaleDateString("it-IT", {
              day: "numeric",
              month: "long",
            })}
          </div>
        </>
      )}
      {msg && <div className="pt-form-error">{msg}</div>}
      <button className="btn-ghost pt-refresh-advice" onClick={refresh} disabled={busy}>
        {busy ? "Analizzo…" : "Aggiorna analisi"}
      </button>
      <p className="pt-disclaimer">
        Analisi automatica a scopo informativo, non è una consulenza finanziaria.
      </p>
    </div>
  );
}

function StrategyCard({
  strategy,
  portfolio,
}: {
  strategy: PtStrategy;
  portfolio: PtPortfolio;
}) {
  const actualFor = (keys?: string[]) => {
    if (!keys?.length) return null;
    const sum = portfolio.positions
      .filter((p) => keys.includes(p.asset_key))
      .reduce((acc, p) => acc + (p.allocation_pct ?? 0), 0);
    return sum > 0 ? sum : null;
  };

  return (
    <div className="card pt-strategy">
      <h3>{strategy.title ?? "La mia strategia"}</h3>
      {strategy.comment && <p className="pt-muted">{strategy.comment}</p>}
      {strategy.targets?.map((t) => {
        const actual = actualFor(t.asset_keys);
        return (
          <div className="pt-target-row" key={t.label}>
            <div className="pt-target-head">
              <span>{t.label}</span>
              <span className="pt-target-nums">
                {actual != null && <strong>{num(actual, 1)}%</strong>}
                <span className="pt-muted"> / target {num(t.target_pct, 0)}%</span>
              </span>
            </div>
            <div className="pt-bar">
              <div
                className="pt-bar-target"
                style={{ width: `${Math.min(t.target_pct, 100)}%` }}
              />
              {actual != null && (
                <div
                  className="pt-bar-actual"
                  style={{ width: `${Math.min(actual, 100)}%` }}
                />
              )}
            </div>
          </div>
        );
      })}
      {strategy.next_moves && strategy.next_moves.length > 0 && (
        <>
          <h4>Prossime mosse</h4>
          <ul className="pt-moves">
            {strategy.next_moves.map((m, i) => (
              <li key={i}>
                <span>{m.label}</span>
                {m.amount_eur != null && <strong>{eur(m.amount_eur)}</strong>}
              </li>
            ))}
          </ul>
        </>
      )}
      {strategy.check_note && <p className="pt-disclaimer">{strategy.check_note}</p>}
    </div>
  );
}

function PositionsCard({ portfolio }: { portfolio: PtPortfolio }) {
  const styles = segmentStyles(portfolio.positions);
  return (
    <div className="card">
      <h3>Le mie posizioni</h3>
      {portfolio.positions.map((p) => (
        <div className="pt-pos-row" key={p.asset_key}>
          <div className="pt-pos-top">
            <div className="pt-pos-main">
              <span className="pt-pos-label">
                <span className="pt-chip-dot" style={styles.get(p.asset_key)} />
                {p.label}
              </span>
              <span className="pt-pos-qty">
                {num(p.quantity)} {UNIT_LABEL[p.unit] ?? p.unit}
                {p.avg_cost_eur > 0 && ` · PMC ${eur(p.avg_cost_eur, 2)}`}
              </span>
            </div>
            <div className="pt-pos-side">
              <span className="pt-pos-value">{eur(p.market_value_eur)}</span>
              <span className={`pt-pos-pnl ${pnlClass(p.pnl_eur)}`}>
                {p.pnl_eur != null
                  ? `${p.pnl_eur >= 0 ? "+" : ""}${eur(p.pnl_eur)}${
                      p.pnl_pct != null ? ` (${p.pnl_pct >= 0 ? "+" : ""}${num(p.pnl_pct, 1)}%)` : ""
                    }`
                  : "prezzo n.d."}
              </span>
            </div>
          </div>
          {p.allocation_pct != null && (
            <div className="pt-pos-allocbar">
              <div
                className="pt-pos-allocfill"
                style={{ width: `${Math.min(p.allocation_pct, 100)}%`, ...styles.get(p.asset_key) }}
              />
              <span className="pt-pos-alloc">{num(p.allocation_pct, 1)}%</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function TransactionForm({
  onSaved,
  onCancel,
}: {
  onSaved: () => Promise<void>;
  onCancel: () => void;
}) {
  const [preset, setPreset] = useState<Preset>(PRESETS[0]);
  const [ticker, setTicker] = useState("");
  const [label, setLabel] = useState("");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [quantity, setQuantity] = useState("");
  const [unit, setUnit] = useState(PRESETS[0].units[0]);
  const [price, setPrice] = useState("");
  const [fee, setFee] = useState("");
  const [tradeDate, setTradeDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const pickPreset = (p: Preset) => {
    setPreset(p);
    setUnit(p.units[0]);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const isCustom = preset.asset_key === "custom";
    if (isCustom && !ticker.trim()) {
      setError("Inserisci il ticker (es. VUAA.DE)");
      return;
    }
    const tx: PtTransactionIn = {
      asset_key: isCustom ? ticker.trim().toUpperCase() : preset.asset_key,
      side,
      quantity: parseFloat(quantity.replace(",", ".")),
      unit,
      unit_price_eur: parseFloat(price.replace(",", ".")),
      fee_eur: fee ? parseFloat(fee.replace(",", ".")) : 0,
      trade_date: tradeDate,
      note: note.trim() || null,
      label: isCustom ? label.trim() || null : null,
    };
    if (!(tx.quantity > 0) || !(tx.unit_price_eur >= 0)) {
      setError("Controlla quantità e prezzo");
      return;
    }
    setBusy(true);
    try {
      await ptApi.addTransaction(tx);
      await onSaved();
    } catch (err) {
      setError(err instanceof PtApiError ? err.message : "Salvataggio non riuscito");
      setBusy(false);
    }
  };

  return (
    <form className="card pt-form" onSubmit={submit}>
      <h3>{side === "buy" ? "Nuovo acquisto" : "Nuova vendita"}</h3>

      <div className="pt-segmented">
        {PRESETS.map((p) => (
          <button
            type="button"
            key={p.asset_key}
            className={preset.asset_key === p.asset_key ? "active" : ""}
            onClick={() => pickPreset(p)}
          >
            {p.label}
          </button>
        ))}
      </div>

      {preset.asset_key === "custom" && (
        <div className="pt-form-grid">
          <label className="pt-field">
            <span>Ticker (Yahoo Finance)</span>
            <input
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="es. VUAA.DE"
            />
          </label>
          <label className="pt-field">
            <span>Nome (opzionale)</span>
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="es. ETF S&P 500"
            />
          </label>
        </div>
      )}

      <div className="pt-segmented pt-segmented-2">
        <button
          type="button"
          className={side === "buy" ? "active" : ""}
          onClick={() => setSide("buy")}
        >
          Compro
        </button>
        <button
          type="button"
          className={side === "sell" ? "active" : ""}
          onClick={() => setSide("sell")}
        >
          Vendo
        </button>
      </div>

      <div className="pt-form-grid">
        <label className="pt-field">
          <span>Quantità</span>
          <input
            inputMode="decimal"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            placeholder="es. 10"
            required
          />
        </label>
        {preset.units.length > 1 ? (
          <label className="pt-field">
            <span>Unità</span>
            <select value={unit} onChange={(e) => setUnit(e.target.value)}>
              {preset.units.map((u) => (
                <option key={u} value={u}>
                  {UNIT_LABEL[u] ?? u}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <label className="pt-field">
            <span>Unità</span>
            <input value={UNIT_LABEL[unit] ?? unit} disabled />
          </label>
        )}
        <label className="pt-field">
          <span>Prezzo per {UNIT_LABEL[unit] ?? unit} (€)</span>
          <input
            inputMode="decimal"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
            placeholder="es. 75,50"
            required
          />
        </label>
        <label className="pt-field">
          <span>Commissioni (€, opzionale)</span>
          <input
            inputMode="decimal"
            value={fee}
            onChange={(e) => setFee(e.target.value)}
            placeholder="0"
          />
        </label>
        <label className="pt-field">
          <span>Data</span>
          <input
            type="date"
            value={tradeDate}
            onChange={(e) => setTradeDate(e.target.value)}
            required
          />
        </label>
        <label className="pt-field">
          <span>Nota (opzionale)</span>
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="es. moneta Krugerrand"
          />
        </label>
      </div>

      {error && <div className="pt-form-error">{error}</div>}

      <div className="pt-form-actions">
        <button type="button" className="pt-btn-cancel" onClick={onCancel}>
          Annulla
        </button>
        <button className="btn" disabled={busy}>
          {busy ? "Salvo…" : "Salva"}
        </button>
      </div>
    </form>
  );
}

function TransactionsCard({
  transactions,
  onDelete,
}: {
  transactions: PtTransaction[];
  onDelete: (id: number) => void;
}) {
  return (
    <div className="card">
      <h3>Movimenti</h3>
      {transactions.map((t) => (
        <div className="pt-tx-row" key={t.id}>
          <div className="pt-tx-main">
            <span className="pt-tx-label">
              {t.side === "buy" ? "Acquisto" : "Vendita"} {t.label}
            </span>
            <span className="pt-tx-detail">
              {new Date(t.trade_date).toLocaleDateString("it-IT")} · {num(t.quantity)}{" "}
              {UNIT_LABEL[t.unit] ?? t.unit} a {eur(t.unit_price_eur, 2)}
              {t.fee_eur > 0 && ` · fee ${eur(t.fee_eur, 2)}`}
              {t.note && ` · ${t.note}`}
            </span>
          </div>
          <div className="pt-tx-side">
            <span className={t.side === "buy" ? "" : "pt-pos"}>
              {t.side === "buy" ? "-" : "+"}
              {eur(t.quantity * t.unit_price_eur + (t.side === "buy" ? t.fee_eur : -t.fee_eur), 2)}
            </span>
            <button
              className="pt-tx-delete"
              onClick={() => onDelete(t.id)}
              aria-label="Cancella movimento"
              title="Cancella"
            >
              ✕
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
