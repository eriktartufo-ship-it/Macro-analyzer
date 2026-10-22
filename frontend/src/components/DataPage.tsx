import { useEffect, useState } from "react";
import { api } from "../api/client";
import { AssetCalibrationPanel } from "./AssetCalibrationPanel";
import { BacktestPanel } from "./BacktestPanel";
import { DedollarComparisonPanel } from "./DedollarComparisonPanel";
import { FactorRegimePanel } from "./FactorRegimePanel";
import { FOMCSentimentPanel } from "./FOMCSentimentPanel";
import { MonteCarloPanel } from "./MonteCarloPanel";
import { RegimeDynamicsPanel } from "./RegimeDynamicsPanel";
import { RegimeEnsemblePanel } from "./RegimeEnsemblePanel";
import { SmoothedIndicatorsPanel } from "./SmoothedIndicatorsPanel";
import { TermPremiumPanel } from "./TermPremiumPanel";
import { ScrollShadow } from "./ScrollShadow";
import { LazyDetails } from "./LazyDetails";
import type { DataSnapshot, Regime } from "../types";
import { assetName, fmt, REGIME_LABEL as REGIME_LABELS } from "../lib/assets";

const REGIMES: Regime[] = ["reflation", "stagflation", "deflation", "goldilocks"];

export type DataSection = "ragiona" | "modelli" | "dati";

type RagionaView = "regime" | "asset" | "dollaro";
// S55: «Come ragiona» era una pagina sola di 8.700 px; ora tre viste, una per domanda.
const RAGIONA_VIEWS: { key: RagionaView; label: string }[] = [
  { key: "regime", label: "Il regime" },
  { key: "asset", label: "Gli asset" },
  { key: "dollaro", label: "Il dollaro" },
];

function formatName(name: string): string {
  return name.replace(/_/g, " ");
}

function formatValue(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return fmt(v, digits);
}

function formatPercent(v: number): string {
  return `${fmt(v * 100, 1)}%`;
}

/**
 * S54: la vecchia pagina «Data» (21.600 px, 24 schermate) è divisa nelle schede di
 * Metodo: «Come ragiona» (formule, pesi, matrice), «Modelli» (ognuno si apre a
 * richiesta) e «Dati grezzi» (le due tabelle, con ricerca).
 */
export function DataPage({ section = "ragiona" }: { section?: DataSection }) {
  if (section === "modelli") return <ModelsList />;
  return <SnapshotView section={section} />;
}

const MODELS: { title: string; sub: string; render: () => JSX.Element }[] = [
  { title: "Tre modelli a confronto", sub: "regole, HMM e MS-VAR votano il regime", render: () => <RegimeEnsemblePanel /> },
  { title: "Tono della Fed", sub: "comunicati FOMC letti dall'AI: falco o colomba", render: () => <FOMCSentimentPanel /> },
  { title: "Monte Carlo e scenari di shock", sub: "dove può andare il regime, e cosa succede con uno shock", render: () => <MonteCarloPanel /> },
  { title: "Indicatori filtrati", sub: "filtro di Kalman: il segnale senza il rumore", render: () => <SmoothedIndicatorsPanel /> },
  { title: "Term premium", sub: "scomposizione Adrian-Crump-Moench dei rendimenti", render: () => <TermPremiumPanel /> },
  { title: "Transizioni fra regimi", sub: "matrice empirica e classificatore HMM", render: () => <RegimeDynamicsPanel /> },
  { title: "Calibrazione degli asset", sub: "ipotesi iniziali contro dati misurati", render: () => <AssetCalibrationPanel /> },
  { title: "Fattori per regime", sub: "Fama-French: quali stili rendono in quale regime", render: () => <FactorRegimePanel /> },
  { title: "Backtest e anticipo sulle recessioni", sub: "come avrebbe allocato in passato", render: () => <BacktestPanel /> },
];

function ModelsList() {
  return (
    <div className="models-list">
      <p className="muted-s" style={{ margin: "0 0 4px" }}>
        I modelli di supporto al classificatore. Ognuno si apre e si carica solo quando lo tocchi.
      </p>
      {MODELS.map((m) => (
        <LazyDetails
          key={m.title}
          className="card model-acc"
          summary={
            <span className="model-sum">
              <span>{m.title}</span>
              <span className="muted-s">{m.sub}</span>
            </span>
          }
        >
          {m.render()}
        </LazyDetails>
      ))}
    </div>
  );
}

function SnapshotView({ section }: { section: "ragiona" | "dati" }) {
  const [query, setQuery] = useState("");
  const [view, setView] = useState<RagionaView>("regime");
  const [condRegime, setCondRegime] = useState<Regime | null>(null);
  const [data, setData] = useState<DataSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api
      .dataSnapshot()
      .then((d) => {
        if (alive) setData(d);
      })
      .catch((e) => {
        if (alive) setError(e instanceof Error ? e.message : "Errore caricamento dati");
      });
    return () => {
      alive = false;
    };
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!data) return <div className="loading">Caricamento dati…</div>;

  if (section === "dati") {
    const q = query.trim().toLowerCase().replace(/\s+/g, "_");
    const match = ([k]: [string, number]) => !q || k.toLowerCase().includes(q);
    const macro = Object.entries(data.current_indicators).sort(([a], [b]) => a.localeCompare(b)).filter(match);
    const dollar = Object.entries(data.current_dedollar_indicators).sort(([a], [b]) => a.localeCompare(b)).filter(match);
    return (
      <div>
        <div className="card">
          <h2>Dati grezzi</h2>
          <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 12 }}>
            I valori che entrano nel modello, aggiornati al {data.date ?? "—"}.
          </div>
          <input
            type="search"
            className="search-input"
            placeholder="Cerca un indicatore (es. cpi, spread, gold)"
            aria-label="Cerca un indicatore"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <RawTable title={`Indicatori macro (${macro.length})`} rows={macro} />
        <RawTable title={`Indicatori del dollaro (${dollar.length})`} rows={dollar} />
      </div>
    );
  }

  const assetsInMatrix = Object.keys(data.asset_regime_data);
  // Una tabella di condizioni alla volta (prima 4 impilate, 74 righe): parte dal regime attuale.
  const shownRegime: Regime = condRegime ?? (data.regime as Regime | null) ?? "reflation";
  const conds = data.regime_conditions[shownRegime] ?? [];
  const totalWeight = conds.reduce((acc, c) => acc + c.weight, 0);

  return (
    <div>
      <div className="card">
        <h2>Come ragiona il modello</h2>
        <div style={{ fontSize: 13, color: "var(--muted)" }}>
          Le formule e i pesi che decidono il regime e la classifica degli asset.
          Aggiornato al {data.date ?? "—"}.
        </div>
        <div className="segmented" style={{ marginTop: 12 }} role="tablist" aria-label="Parte del modello">
          {RAGIONA_VIEWS.map((v) => (
            <button
              key={v.key}
              role="tab"
              aria-selected={view === v.key}
              className={view === v.key ? "active" : ""}
              onClick={() => setView(v.key)}
            >
              {v.label}
            </button>
          ))}
        </div>
      </div>

      {view === "regime" && (
        <>
          <div className="card">
            <h2>Dagli indicatori al regime</h2>
            <div style={{ display: "grid", gap: 14 }}>
              <FormulaBlock
                title="Quanto ogni regime si adatta ai dati"
                body={
                  <>
                    <code>
                      fit(r) = Σ<sub>i</sub> peso(c<sub>i</sub>) · eval(c<sub>i</sub>, x<sub>i</sub>) − penalità(r, x)
                    </code>
                    <p>
                      <em>eval</em> misura quanto è soddisfatta ogni condizione, con una curva a S o a campana.
                      La <em>penalità</em> abbassa il fit quando un indicatore chiave contraddice il regime
                      (es. inflazione sopra il 3,5% penalizza deflazione e goldilocks; PIL sotto zero penalizza
                      la reflazione).
                    </p>
                  </>
                }
              />
              <FormulaBlock
                title="Dal fit alle probabilità"
                body={
                  <>
                    <code>prob(r) = max(0, fit(r)) / Σ<sub>r</sub> max(0, fit(r))</code>
                    <p>
                      I fit negativi valgono zero, poi si divide per la somma: le quattro probabilità fanno 100%.
                      Se tutti i fit sono negativi, ogni regime vale 25%.
                    </p>
                  </>
                }
              />
            </div>
            <div className="ragiona-now">
              {REGIMES.map((r) => (
                <div key={r} className={r === data.regime ? "is-current" : ""}>
                  <span className={`regime-${r}`}>{REGIME_LABELS[r]}</span>
                  <strong>{formatPercent(data.current_probabilities[r] ?? 0)}</strong>
                  <span>fit {formatValue(data.current_fit_scores[r])}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h2>Le condizioni di ogni regime e il loro peso</h2>
            <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
              I pesi di un regime non sommano per forza a 1: ogni condizione aggiunge al fit in proporzione al suo peso.
            </div>
            <div className="segmented" style={{ marginBottom: 12 }} role="tablist" aria-label="Regime">
              {REGIMES.map((r) => (
                <button
                  key={r}
                  role="tab"
                  aria-selected={shownRegime === r}
                  className={shownRegime === r ? "active" : ""}
                  onClick={() => setCondRegime(r)}
                >
                  {REGIME_LABELS[r]}
                </button>
              ))}
            </div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 6 }}>
              {conds.length} condizioni · somma dei pesi {formatValue(totalWeight)}
              {shownRegime === data.regime && " · è il regime attuale"}
            </div>
            <table className="table" style={{ minWidth: 0 }}>
              <thead>
                <tr>
                  <th>Condizione</th>
                  <th className="num">Peso</th>
                </tr>
              </thead>
              <tbody>
                {conds.map((c) => (
                  <tr key={c.name} title={c.name}>
                    <td style={{ fontSize: 13 }}>{c.description}</td>
                    <td className="num">{formatValue(c.weight)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {view === "asset" && (
        <>
          <div className="card">
            <h2>Lo score di un asset</h2>
            <FormulaBlock
              title="Media dei regimi, pesata per la loro probabilità"
              body={
                <>
                  <code>
                    score(a) = Σ<sub>r</sub> prob(r) · [0,25·hit(a,r) + 0,50·rend_reale_norm(a,r) + 0,25·sharpe_norm(a,r)] · 100
                    + bonus_dollaro(a) + notizie(a) − penalità(a)
                  </code>
                  <p>
                    <em>rend_reale_norm = clamp((rendimento reale + 0,30) / 0,60, 0, 1)</em>: da −30% a +30% reale
                    l'anno. <em>sharpe_norm = clamp((sharpe + 1) / 3, 0, 1)</em>. Il rendimento reale pesa metà
                    per non premiare la liquidità, che ha uno Sharpe alto anche quando rende poco.
                  </p>
                </>
              }
            />
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 12 }}>
              La classifica di oggi, con questi score, è in Mercati › Classifica.
            </div>
          </div>

          <div className="card">
            <h2>Come è andato ogni asset in ogni regime</h2>
            <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
              Batte infl. = quante volte ha battuto l'inflazione · Rend. = rendimento reale medio annuo ·
              Vol. = volatilità · Sharpe. Dati storici fissati nel codice: sono il punto di partenza dello score.
            </div>
            <ScrollShadow innerClassName="table-wrap">
              <table className="table table-nowrap">
                <thead>
                  <tr>
                    <th>Asset</th>
                    {REGIMES.map((r) => (
                      <th key={r} colSpan={4} className="num" style={{ textAlign: "center" }}>
                        {REGIME_LABELS[r]}
                      </th>
                    ))}
                  </tr>
                  <tr>
                    <th></th>
                    {REGIMES.flatMap((r) => [
                      <th key={`${r}-hr`} className="num" style={{ fontSize: 11 }}>Batte infl.</th>,
                      <th key={`${r}-ret`} className="num" style={{ fontSize: 11 }}>Rend.%</th>,
                      <th key={`${r}-vol`} className="num" style={{ fontSize: 11 }}>Vol.%</th>,
                      <th key={`${r}-sh`} className="num" style={{ fontSize: 11 }}>Sharpe</th>,
                    ])}
                  </tr>
                </thead>
                <tbody>
                  {assetsInMatrix.map((asset) => {
                    const row = data.asset_regime_data[asset];
                    return (
                      <tr key={asset}>
                        <td>{assetName(asset)}</td>
                        {REGIMES.flatMap((r) => {
                          const s = row[r];
                          return [
                            <td key={`${r}-hr`} className="num">{s ? formatPercent(s.hit_rate) : "—"}</td>,
                            <td key={`${r}-ret`} className="num">{s ? formatValue(s.avg_return * 100, 1) : "—"}</td>,
                            <td key={`${r}-vol`} className="num">{s ? formatValue(s.vol * 100, 1) : "—"}</td>,
                            <td key={`${r}-sh`} className="num">{s ? formatValue(s.sharpe, 2) : "—"}</td>,
                          ];
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </ScrollShadow>
          </div>
        </>
      )}

      {view === "dollaro" && (
        <>
          <div className="card">
            <h2>Il punteggio di de-dollarizzazione</h2>
            <FormulaBlock
              title="Media pesata degli orizzonti"
              body={
                <>
                  <code>
                    complessivo = 0,20 · ciclico + 0,25 · strutturale + 0,25 · decennale + 0,20 · ventennale
                    + 0,10 · max(0, accelerazione) / 2
                  </code>
                  <p>
                    Ogni componente va da 0 a 1. Se mancano i dati a 20 anni i pesi diventano 0,25 · 0,30 · 0,30
                    e 0,15 sull'accelerazione. L'<em>accelerazione</em> confronta il ritmo di oggi con quello di un
                    anno fa (dollaro, oro e aree del mondo) e conta solo quando è positiva. Il punteggio{" "}
                    <em>geopolitico</em> si mostra a parte e non entra nel complessivo. Il bonus di ogni asset è{" "}
                    <code>sensibilità(a) · complessivo</code>.
                  </p>
                </>
              }
            />
            <div style={{ fontSize: 13, marginTop: 12 }}>
              Oggi il punteggio complessivo è <strong>{formatValue(data.current_dedollar_combined, 3)}</strong>.
            </div>
          </div>
          <DedollarComparisonPanel />
        </>
      )}
    </div>
  );
}

function RawTable({ title, rows }: { title: string; rows: [string, number][] }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      {rows.length === 0 ? (
        <div style={{ fontSize: 13, color: "var(--muted)" }}>Nessun indicatore con questo nome.</div>
      ) : (
        // S55: griglia a colonne invece di una tabella a due colonne lunga 155 righe.
        <dl className="raw-grid">
          {rows.map(([k, v]) => (
            <div key={k}>
              <dt>{formatName(k)}</dt>
              <dd>{formatValue(v, 4)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

function FormulaBlock({ title, body }: { title: string; body: React.ReactNode }) {
  return (
    <div
      style={{
        padding: 12,
        border: "1px solid var(--border)",
        borderRadius: 8,
        background: "var(--bg)",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{title}</div>
      <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.5 }}>{body}</div>
    </div>
  );
}
