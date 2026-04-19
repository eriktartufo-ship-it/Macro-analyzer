import React, { useState, useEffect } from "react";
import type { Dedollarization, PlayerScore, PlayerSignal } from "../types";
import { api } from "../api/client";
import { ScrollShadow } from "./ScrollShadow";

interface Props {
  data: Dedollarization;
  rawIndicators?: Record<string, number>;
  onBack?: () => void;
}

function scoreColor(score: number): string {
  if (score >= 0.7) return "var(--deflation)";
  if (score >= 0.5) return "var(--goldilocks)";
  return "var(--reflation)";
}

function scoreLabel(score: number): string {
  if (score >= 0.75) return "Extreme";
  if (score >= 0.6) return "High";
  if (score >= 0.4) return "Moderate";
  if (score >= 0.25) return "Low";
  return "Minimal";
}

function renderInline(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const regex = /\*\*([^*]+)\*\*/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let idx = 0;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    parts.push(
      <strong key={`b${idx++}`} style={{ color: "var(--accent)" }}>
        {m[1]}
      </strong>
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function renderExplanation(text: string): React.ReactNode {
  const paragraphs = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  return paragraphs.map((p, i) => (
    <p key={i} style={{ margin: i === 0 ? "0 0 10px 0" : "0 0 10px 0" }}>
      {renderInline(p)}
    </p>
  ));
}

// ── Descrizioni statiche per ogni indicatore ──

const SIGNAL_HINTS: Record<string, string> = {
  gold_sp500_ratio: "Oro vs azioni: alto = fuga verso beni scarsi, sfiducia nel sistema",
  copper_gold_ratio: "Rame vs oro: basso = paura/recessione, alto = ciclo industriale sano",
  m2_roc_12m: "Crescita massa monetaria M2 a 12m: alta = stampa moneta, diluizione USD",
  yield_curve_10y2y: "Spread 10Y-2Y: inversione (<0) = recessione in arrivo, stress fiscale",
  real_yield_10y: "Rendimento reale 10Y (TIPS): negativo = erosione potere d'acquisto del dollaro",
  interest_tax_ratio: "Interessi sul debito / entrate fiscali: >20% = soglia di insostenibilita",
  foreign_treasury_roc_12m: "Variazione 12m holdings esteri di Treasury: calo = vendite nette di USD",
  btp_bund_spread: "Spread Italia-Germania: alto = stress/frammentazione eurozona",
  eur_chf: "Tasso EUR/CHF: basso = flight-to-quality verso il franco svizzero",
  japan_10y: "Rendimento JGB 10Y: alto = fine yield control BoJ, rischio carry unwind globale",
  jpy_usd_roc_3m: "Variazione yen 3m: apprezzamento rapido = panico, flight-to-safety",
  commodity_fx_strength: "Forza CAD+AUD a 12m: positiva = super-ciclo commodity, domanda reale",
  em_hy_oas: "Spread HY (proxy EM): basso = risk-on verso emergenti, capitali escono da USD",
  defense_gdp_pct: "Spesa difesa / PIL: alta = riarmo, frammentazione geopolitica",
};

const PLAYER_HINTS: Record<string, string> = {
  system: "Fiducia nel sistema monetario: oro vs azioni, liquidita globale",
  usa: "Stress fiscale USA: curva rendimenti, costi debito, domanda estera di Treasury",
  europe: "Stabilita eurozona: spread periferia, flussi verso CHF",
  japan: "Rischio carry trade: normalizzazione BoJ, forza yen",
  commodity_fx: "Ciclo materie prime: forza valute legate a risorse naturali",
  em: "Attrattivita emergenti: spread creditizio, risk-on vs USD",
  defense: "Pressione geopolitica: spesa militare e riarmo",
};

type HorizonMeta = Record<string, { label: string; hint: string }>;

const CYCLICAL_META: HorizonMeta = {
  usd_weakness: { label: "USD Weakness (12m)", hint: "Variazione dollaro 12m: calo = dedollarizzazione ciclica" },
  gold_strength: { label: "Gold Strength (12m)", hint: "Crescita oro 12m: alta = domanda di safe-haven" },
  gold_oil_ratio: { label: "Gold / Oil Ratio", hint: "Oro/petrolio: alto = preferenza per riserva di valore vs energia" },
  debt_burden: { label: "Debt Burden", hint: "Debito/PIL USA: >100% = zona critica per sostenibilita" },
  real_rate_signal: { label: "Real Rate Signal", hint: "Tasso reale (FedFunds - CPI): negativo = svalutazione monetaria" },
  monetary_debasement: { label: "M2 Debasement (12m)", hint: "Crescita M2 12m: alta = diluizione del valore del dollaro" },
};

const STRUCTURAL_META: HorizonMeta = {
  usd_secular: { label: "USD Decline (5Y)", hint: "Trend dollaro su 5 anni: calo persistente = declino strutturale" },
  gold_secular: { label: "Gold Rise (5Y)", hint: "Trend oro su 5 anni: crescita = sfiducia strutturale nel fiat" },
  debt_trajectory: { label: "Debt Trajectory (5Y)", hint: "Variazione Debito/PIL su 5 anni: +20pp = espansione insostenibile" },
  m2_cumulative: { label: "M2 Cumulative (5Y)", hint: "Crescita M2 cumulata 5 anni: alta = debasement persistente" },
};

const DECADE_META: HorizonMeta = {
  usd_decade: { label: "USD Decline (10Y)", hint: "Trend dollaro su 10 anni: calo secolare confermato" },
  gold_decade: { label: "Gold Rise (10Y)", hint: "Trend oro su 10 anni: super-ciclo rialzista" },
  debt_decade: { label: "Debt Growth (10Y)", hint: "Crescita debito/PIL in un decennio: +30pp = esplosione" },
  m2_decade: { label: "M2 Expansion (10Y)", hint: "Espansione M2 decennale: strutturale e difficile da invertire" },
};

const TWENTY_META: HorizonMeta = {
  usd_20y: { label: "USD Decline (20Y)", hint: "Trend dollaro su 20 anni: declino generazionale" },
  gold_20y: { label: "Gold Rise (20Y)", hint: "Trend oro su 20 anni: crescita secolare plurigenerazionale" },
  debt_20y: { label: "Debt Growth (20Y)", hint: "Crescita debito/PIL su 20 anni: traiettoria storica" },
  m2_20y: { label: "M2 Expansion (20Y)", hint: "Espansione M2 su 20 anni: misura la stampa monetaria cumulata" },
};

// ── Componenti UI ──

function HorizonScore({ label, value }: { label: string; value: number | null | undefined }) {
  if (value === null || value === undefined) {
    return (
      <div>
        <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", marginBottom: 4 }}>
          {label}
        </div>
        <div style={{ fontSize: 22, fontWeight: 700, color: "var(--muted)" }}>—</div>
      </div>
    );
  }
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: scoreColor(value) }}>
        {(value * 100).toFixed(0)}%
      </div>
    </div>
  );
}

function SignalRow({ signal }: { signal: PlayerSignal }) {
  const hasValue = signal.value !== null && signal.score !== null;
  const hint = SIGNAL_HINTS[signal.key];
  return (
    <div
      style={{
        padding: "8px 12px",
        background: "var(--bg)",
        borderRadius: 6,
        borderLeft: signal.red_flag
          ? "3px solid var(--deflation)"
          : hasValue
          ? `3px solid ${scoreColor(signal.score!)}`
          : "3px solid var(--muted)",
        opacity: hasValue ? 1 : 0.55,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 2 }}>
        <span style={{ fontSize: 13, fontWeight: 500, flex: 1 }}>
          {signal.label}
          {signal.red_flag && (
            <span
              style={{
                marginLeft: 8,
                fontSize: 9,
                fontWeight: 700,
                padding: "1px 6px",
                borderRadius: 3,
                background: "rgba(239,68,68,0.2)",
                color: "var(--deflation)",
              }}
            >
              RED FLAG
            </span>
          )}
        </span>
        {hasValue && (
          <span style={{ fontSize: 12, color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>
            w: {(signal.weight * 100).toFixed(0)}%
          </span>
        )}
        {hasValue && (
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: scoreColor(signal.score!),
              width: 50,
              textAlign: "right",
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {(signal.score! * 100).toFixed(0)}%
          </span>
        )}
      </div>
      {hint && (
        <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 3, fontStyle: "italic" }}>{hint}</div>
      )}
      {hasValue && (
        <div className="prob-bar" style={{ height: 4, marginBottom: 4 }}>
          <div
            className="prob-fill"
            style={{ width: `${(signal.score! * 100).toFixed(0)}%`, background: scoreColor(signal.score!) }}
          />
        </div>
      )}
      <div style={{ fontSize: 12, color: "var(--muted)" }}>{signal.interpret}</div>
    </div>
  );
}

function PlayerSection({ playerId, player }: { playerId: string; player: PlayerScore }) {
  const [expanded, setExpanded] = useState(true);
  const redFlags = player.signals.filter((s) => s.red_flag).length;
  const validSignals = player.signals.filter((s) => s.value !== null).length;
  const hint = PLAYER_HINTS[playerId];

  return (
    <div className="player-card">
      <button
        onClick={() => setExpanded((e) => !e)}
        className="player-header"
        aria-expanded={expanded}
      >
        <span className="player-header-caret">{expanded ? "▼" : "▶"}</span>
        <span className="player-header-title">{player.label}</span>
        <span
          className="player-header-score"
          style={{ color: scoreColor(player.score) }}
        >
          {(player.score * 100).toFixed(0)}%
        </span>
        <span className="player-header-sub">
          {hint && <span style={{ flex: 1, minWidth: 140 }}>{hint}</span>}
          {redFlags > 0 && (
            <span className="chip chip-danger" data-nowrap>
              {redFlags} RED FLAG{redFlags > 1 ? "S" : ""}
            </span>
          )}
          <span data-nowrap>
            {validSignals}/{player.signals.length} segnali
          </span>
        </span>
      </button>
      {expanded && (
        <div className="player-body">
          {player.signals.map((s) => (
            <SignalRow key={s.key} signal={s} />
          ))}
        </div>
      )}
    </div>
  );
}

function HorizonBars({
  label,
  items,
  meta,
}: {
  label: string;
  items: Record<string, number>;
  meta: HorizonMeta;
}) {
  const sorted = Object.entries(items).sort((a, b) => b[1] - a[1]);
  if (sorted.length === 0) return null;
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
        {label}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {sorted.map(([key, value]) => {
          const m = meta[key];
          return (
            <div key={key} style={{ padding: "6px 10px", background: "var(--bg)", borderRadius: 6 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 2,
                }}
              >
                <span style={{ fontSize: 13 }}>{m?.label ?? key}</span>
                <span style={{ fontSize: 13, fontWeight: 600, color: scoreColor(value) }}>
                  {(value * 100).toFixed(0)}%
                </span>
              </div>
              {m?.hint && (
                <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 3, fontStyle: "italic" }}>
                  {m.hint}
                </div>
              )}
              <div className="prob-bar" style={{ height: 4 }}>
                <div
                  className="prob-fill"
                  style={{ width: `${(value * 100).toFixed(0)}%`, background: scoreColor(value) }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Matrice Player × Orizzonte ──

const HORIZON_COLS = [
  { key: "now", label: "Oggi" },
  { key: "1y", label: "1Y" },
  { key: "5y", label: "5Y" },
  { key: "10y", label: "10Y" },
  { key: "20y", label: "20Y" },
];

const cellStyle: React.CSSProperties = {
  padding: "8px 6px",
  background: "var(--card-bg)",
  fontSize: 13,
};

function PlayerHorizonRow({
  playerId,
  label,
  currentScore,
  history,
  acceleration,
}: {
  playerId: string;
  label: string;
  currentScore: number;
  history: Record<string, Record<string, number>>;
  acceleration?: number;
}) {
  const hint = PLAYER_HINTS[playerId];
  const scores: (number | null)[] = HORIZON_COLS.map((col) => {
    if (col.key === "now") return currentScore;
    return history?.[col.key]?.[playerId] ?? null;
  });

  return (
    <>
      <div style={{ ...cellStyle, display: "flex", flexDirection: "column", justifyContent: "center", gap: 2 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 500, flex: 1 }}>{label}</span>
          {acceleration !== undefined && (
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                color: acceleration > 0.02 ? "var(--deflation)" : acceleration < -0.02 ? "var(--reflation)" : "var(--muted)",
              }}
            >
              {acceleration > 0 ? "+" : ""}{(acceleration * 100).toFixed(0)}pp
            </span>
          )}
        </div>
        {hint && (
          <div style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.2 }}>{hint}</div>
        )}
      </div>
      {scores.map((s, i) => (
        <div
          key={HORIZON_COLS[i].key}
          style={{
            ...cellStyle,
            textAlign: "center",
            fontWeight: 600,
            fontVariantNumeric: "tabular-nums",
            color: s !== null ? scoreColor(s) : "var(--muted)",
          }}
        >
          {s !== null ? `${(s * 100).toFixed(0)}%` : "—"}
        </div>
      ))}
    </>
  );
}

function accelLabel(accel: number): { text: string; color: string } {
  if (accel > 0.3) return { text: "Accelerating fast", color: "var(--deflation)" };
  if (accel > 0.1) return { text: "Accelerating", color: "var(--goldilocks)" };
  if (accel > -0.1) return { text: "Stable pace", color: "var(--muted)" };
  if (accel > -0.3) return { text: "Decelerating", color: "var(--reflation)" };
  return { text: "Reversing", color: "var(--reflation)" };
}

// ── Pagina principale ──

export function DedollarizationPage({ data }: Props) {
  const [view, setView] = useState<"horizon" | "players" | "ai">("players");
  const [explanation, setExplanation] = useState<string | null>(data.explanation ?? null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth <= 640);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  const generateAi = async () => {
    setAiLoading(true);
    setAiError(null);
    try {
      const res = await api.generateDedollarExplanation();
      setExplanation(res.explanation);
    } catch (e: any) {
      setAiError(e?.message || "Errore generazione analisi AI");
    } finally {
      setAiLoading(false);
    }
  };

  const playerEntries = Object.entries(data.by_player);
  const totalRedFlags = playerEntries.reduce(
    (sum, [, p]) => sum + p.signals.filter((s) => s.red_flag).length,
    0,
  );
  const accel = accelLabel(data.acceleration);

  return (
    <div className="card">
      <h2 style={{ marginTop: 0, marginBottom: 16 }}>Dedollarization — Analisi completa</h2>

      {/* Top scores grid */}
      <div className="grid grid-6" style={{ gap: 12, marginBottom: 20 }}>
        <HorizonScore label="Cyclical (1Y)" value={data.score} />
        <HorizonScore label="Structural (5Y)" value={data.structural_score} />
        <HorizonScore label="Decade (10Y)" value={data.decade_score} />
        <HorizonScore label="20-Year" value={data.twenty_year_score} />
        <HorizonScore label="Geopolitical" value={data.geopolitical_score} />
        <div>
          <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", marginBottom: 4 }}>
            Acceleration
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: accel.color }}>
            {data.acceleration > 0 ? "+" : ""}{(data.acceleration * 100).toFixed(0)}%
          </div>
          <div style={{ fontSize: 11, color: accel.color }}>{accel.text}</div>
        </div>
      </div>

      <div className="surface" style={{ marginBottom: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={{ fontSize: 14, fontWeight: 500 }}>Combined Score</span>
          <span style={{ fontSize: 20, fontWeight: 700, color: scoreColor(data.combined_score) }}>
            {(data.combined_score * 100).toFixed(0)}% — {scoreLabel(data.combined_score)}
          </span>
          {totalRedFlags > 0 && (
            <span className="chip chip-danger">
              {totalRedFlags} RED FLAG{totalRedFlags > 1 ? "S" : ""} ATTIVI
            </span>
          )}
        </div>
        <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 6 }}>
          {data.twenty_year_score !== null
            ? "Media pesata: ciclico 20% + strutturale 5Y 25% + decennale 10Y 25% + ventennale 20Y 20% + accelerazione 10%"
            : "Media pesata: ciclico 25% + strutturale 5Y 30% + decennale 10Y 30% + accelerazione 15%"}
        </div>
      </div>

      {/* View switcher */}
      <div className="segmented" style={{ marginBottom: 18 }}>
        <button
          onClick={() => setView("players")}
          className={view === "players" ? "active" : ""}
        >
          Per Macro-Player
        </button>
        <button
          onClick={() => setView("horizon")}
          className={view === "horizon" ? "active" : ""}
        >
          Per Orizzonte
        </button>
        <button
          onClick={() => setView("ai")}
          className={view === "ai" ? "active" : ""}
        >
          Analisi AI
        </button>
      </div>

      {view === "players" && (
        <div>
          {playerEntries.length === 0 && (
            <div style={{ color: "var(--muted)", padding: "20px 0", textAlign: "center" }}>
              Nessun dato player-based disponibile (in attesa del prossimo refresh).
            </div>
          )}
          {playerEntries.map(([id, player]) => (
            <PlayerSection key={id} playerId={id} player={player} />
          ))}
        </div>
      )}

      {view === "horizon" && (
        <div>
          {/* Matrice Player x Orizzonte */}
          {playerEntries.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 10, fontWeight: 600 }}>
                MACRO PLAYER — EVOLUZIONE TEMPORALE
              </div>
              {isMobile ? (
                <div className="asset-mobile-list" style={{ marginTop: 0 }}>
                  {playerEntries
                    .sort((a, b) => b[1].score - a[1].score)
                    .map(([id, p]) => {
                      const accelVal = data.player_acceleration?.[id];
                      return (
                        <div key={id} className="asset-mobile-card" style={{ padding: 14 }}>
                          <div className="asset-mobile-header" style={{ borderBottom: "1px solid var(--stroke)", paddingBottom: 10, marginBottom: 10 }}>
                            <span className="asset-name" style={{ fontSize: 15 }}>{p.label}</span>
                            {accelVal !== undefined && (
                              <span style={{ fontSize: 13, fontWeight: 700, color: accelVal > 0.02 ? "var(--deflation)" : accelVal < -0.02 ? "var(--reflation)" : "var(--muted)" }}>
                                {accelVal > 0 ? "+" : ""}{(accelVal * 100).toFixed(0)}pp
                              </span>
                            )}
                          </div>
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 4 }}>
                            {HORIZON_COLS.map((col) => {
                              const s = col.key === "now" ? p.score : data.player_history?.[col.key]?.[id] ?? null;
                              return (
                                <div key={col.key} style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end" }}>
                                  <span style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", marginBottom: 2 }}>{col.label}</span>
                                  <span style={{ fontSize: 12, fontWeight: 700, fontVariantNumeric: "tabular-nums", color: s !== null ? scoreColor(s) : "var(--muted)" }}>
                                    {s !== null ? `${(s * 100).toFixed(0)}%` : "—"}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                </div>
              ) : (
                <>
                  <div className="scroll-label">← Scorri per vedere tutti gli orizzonti →</div>
                  <ScrollShadow
                    innerClassName="scroll-x"
                    innerStyle={{
                      display: "grid",
                      gridTemplateColumns: "minmax(180px, 1fr) repeat(5, minmax(60px, 70px))",
                      gap: "1px",
                      background: "var(--divider)",
                      borderRadius: 8,
                      border: "1px solid var(--divider)",
                    }}
                  >
                    {/* Header */}
                    <div style={{ ...cellStyle, fontWeight: 600, fontSize: 11, color: "var(--muted)" }}>Player</div>
                    {HORIZON_COLS.map((col) => (
                      <div key={col.key} style={{ ...cellStyle, fontWeight: 600, fontSize: 11, color: "var(--muted)", textAlign: "center" }}>
                        {col.label}
                      </div>
                    ))}
                    {/* Rows */}
                    {playerEntries
                      .sort((a, b) => b[1].score - a[1].score)
                      .map(([id, p]) => {
                        const accelVal = data.player_acceleration?.[id];
                        return (
                          <PlayerHorizonRow
                            key={id}
                            playerId={id}
                            label={p.label}
                            currentScore={p.score}
                            history={data.player_history}
                            acceleration={accelVal}
                          />
                        );
                      })}
                  </ScrollShadow>
                </>
              )}
            </div>
          )}
          <HorizonBars label="SEGNALI CICLICI (12 MESI)" items={data.components} meta={CYCLICAL_META} />
          <HorizonBars label="TREND STRUTTURALE (5 ANNI)" items={data.structural} meta={STRUCTURAL_META} />
          <HorizonBars label="VISIONE DECENNALE (10 ANNI)" items={data.decade} meta={DECADE_META} />
          <HorizonBars label="VENTENNALE (20 ANNI)" items={data.twenty_year} meta={TWENTY_META} />
        </div>
      )}

      {view === "ai" && (
        <div>
          <div
            style={{
              fontSize: 13,
              color: "var(--muted)",
              marginBottom: 14,
              lineHeight: 1.6,
            }}
          >
            L'analisi narrativa viene generata su richiesta via Gemini 2.5 Flash sui dati
            attualmente in pipeline. Clicca il pulsante per generarla (o rigenerarla).
          </div>

          <button
            onClick={generateAi}
            disabled={aiLoading}
            className="btn"
            style={{ marginBottom: 16 }}
          >
            {aiLoading
              ? "Generazione in corso…"
              : explanation
              ? "Rigenera analisi AI"
              : "Genera analisi AI"}
          </button>

          {aiError && <div className="error">{aiError}</div>}

          {explanation ? (
            <div
              style={{
                padding: "16px 18px",
                background: "var(--accent-bg)",
                border: "1px solid #c6dafc",
                borderRadius: 12,
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: 0.6,
                  color: "var(--accent)",
                  marginBottom: 8,
                }}
              >
                Analisi AI — Gemini 2.5 Flash
              </div>
              <div style={{ fontSize: 14, lineHeight: 1.65, color: "var(--text)" }}>
                {renderExplanation(explanation)}
              </div>
            </div>
          ) : (
            !aiLoading && (
              <div style={{ color: "var(--muted)", fontSize: 13 }}>
                Nessuna analisi generata. Clicca il pulsante per produrne una.
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}
