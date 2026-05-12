import { useState } from "react";
import type { InsiderActivity, InsiderTickerVolume } from "../types";

interface Props {
  data: InsiderActivity;
}

function formatUsd(usd: number): string {
  const abs = Math.abs(usd);
  const sign = usd < 0 ? "-" : "";
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}k`;
  return `${sign}$${abs.toFixed(0)}`;
}

function scoreColor(score: number): { bg: string; fg: string; label: string } {
  if (score > 0.3) return { bg: "var(--success-bg)", fg: "var(--reflation)", label: "BULLISH" };
  if (score < -0.3) return { bg: "var(--danger-bg)", fg: "var(--deflation)", label: "BEARISH" };
  return { bg: "var(--warn-bg)", fg: "var(--warn-text)", label: "NEUTRAL" };
}

export function InsiderActivityTile({ data }: Props) {
  const [expanded, setExpanded] = useState(false);

  const score = data.insider_score;
  const sc = scoreColor(score);
  const isEmpty = data.n_transactions === 0;

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          gap: 12,
          flexWrap: "wrap",
          marginBottom: 6,
        }}
      >
        <h2 style={{ margin: 0 }}>Insider Activity (SEC Form 4)</h2>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>
          {data.period_start && data.period_end
            ? `${data.period_start} → ${data.period_end}`
            : "—"}
        </span>
      </div>

      {isEmpty ? (
        <div
          style={{
            padding: "14px 16px",
            background: "var(--surface-sunk)",
            borderRadius: 6,
            border: "1px dashed var(--stroke)",
            fontSize: 13,
            color: "var(--muted)",
            marginTop: 8,
          }}
        >
          Nessuna transazione disponibile nella finestra. SEC pubblica i daily
          index a fine giornata business — riprova D+1, oppure usa un{" "}
          <code>end_date</code> esplicito storico.
          {data.notes.length > 0 && (
            <ul style={{ margin: "6px 0 0 18px", padding: 0 }}>
              {data.notes.slice(0, 3).map((n, i) => (
                <li key={i} style={{ fontSize: 11 }}>
                  {n}
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <>
          <div
            style={{
              display: "flex",
              gap: 10,
              flexWrap: "wrap",
              alignItems: "center",
              marginBottom: 12,
            }}
          >
            <span
              className="chip"
              style={{
                background: sc.bg,
                color: sc.fg,
                fontSize: 12,
                padding: "5px 12px",
                fontWeight: 700,
                letterSpacing: 0.5,
              }}
            >
              SMART MONEY {sc.label}
            </span>
            <span
              style={{
                fontSize: 13,
                fontWeight: 600,
                fontVariantNumeric: "tabular-nums",
                color: sc.fg,
              }}
            >
              score {score >= 0 ? "+" : ""}
              {score.toFixed(3)}
            </span>
            <span
              style={{
                fontSize: 12,
                color: "var(--muted)",
              }}
            >
              {data.n_transactions} txs · {data.n_buy_filings} buy / {data.n_sell_filings} sell
            </span>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap: 10,
              marginBottom: 12,
            }}
          >
            <Metric label="Total Buy" value={formatUsd(data.total_buy_usd)} color="var(--reflation)" />
            <Metric label="Total Sell" value={formatUsd(data.total_sell_usd)} color="var(--deflation)" />
            <Metric
              label="Net Flow"
              value={formatUsd(data.net_flow_usd)}
              color={data.net_flow_usd >= 0 ? "var(--reflation)" : "var(--deflation)"}
            />
          </div>

          <ScoreBar score={score} />

          {(data.top_buy_tickers.length > 0 || data.top_sell_tickers.length > 0) && (
            <div className="grid grid-2" style={{ gap: 16, marginTop: 12, marginBottom: 0 }}>
              <TickerList
                title="Top Buys"
                accent="var(--reflation)"
                items={data.top_buy_tickers.slice(0, 5)}
              />
              <TickerList
                title="Top Sells"
                accent="var(--deflation)"
                items={data.top_sell_tickers.slice(0, 5)}
              />
            </div>
          )}

          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            style={{
              marginTop: 12,
              fontSize: 12,
              padding: "6px 12px",
              background: "transparent",
              border: "1px solid var(--stroke)",
              borderRadius: 6,
              cursor: "pointer",
              color: "var(--muted)",
            }}
          >
            {expanded ? "Nascondi dettagli" : "Mostra dettagli"}
          </button>

          {expanded && (
            <div
              style={{
                marginTop: 12,
                padding: 12,
                background: "var(--surface-sunk)",
                borderRadius: 6,
                fontSize: 12,
                color: "var(--muted)",
                lineHeight: 1.6,
              }}
            >
              <div>
                <strong>Buy/Sell ratio:</strong>{" "}
                {data.buy_sell_ratio !== null
                  ? data.buy_sell_ratio.toFixed(2)
                  : "—"}{" "}
                ({data.buy_sell_ratio !== null && data.buy_sell_ratio > 1
                  ? "buying dominates"
                  : data.buy_sell_ratio !== null && data.buy_sell_ratio < 1
                  ? "selling dominates"
                  : "—"})
              </div>
              <div style={{ marginTop: 4 }}>
                <strong>Score formula:</strong> tanh(net_flow_usd / $100M) → bounded [-1, +1].
                {" "}Threshold |score|&gt;0.3 = smart money signal forte (7° classifier pillar).
              </div>
              <div style={{ marginTop: 4 }}>
                <strong>Esclusioni:</strong> option exercises (M), compensation grants (A),
                tax withholdings (F) — NON market signal.
              </div>
              {data.notes.length > 0 && (
                <div style={{ marginTop: 4 }}>
                  <strong>Notes:</strong>
                  <ul style={{ margin: "4px 0 0 18px", padding: 0 }}>
                    {data.notes.map((n, i) => (
                      <li key={i}>{n}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Metric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div
      style={{
        padding: "8px 12px",
        background: "var(--surface-sunk)",
        borderRadius: 6,
        border: "1px solid var(--stroke)",
      }}
    >
      <div
        style={{
          fontSize: 10,
          fontWeight: 700,
          color: "var(--muted)",
          textTransform: "uppercase",
          letterSpacing: 0.5,
          marginBottom: 2,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 15,
          fontWeight: 700,
          fontVariantNumeric: "tabular-nums",
          color,
        }}
      >
        {value}
      </div>
    </div>
  );
}

function ScoreBar({ score }: { score: number }) {
  // Score range [-1, +1] → posizione 0-100%. 0.5 = center.
  const clamped = Math.max(-1, Math.min(1, score));
  const center = 50;
  const offset = clamped * 50;
  const filled = Math.abs(offset);
  const isBullish = clamped >= 0;
  return (
    <div>
      <div
        style={{
          position: "relative",
          height: 8,
          background: "var(--surface-2)",
          borderRadius: 4,
          overflow: "hidden",
        }}
      >
        {/* Center marker */}
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: "50%",
            width: 1,
            background: "var(--stroke)",
          }}
        />
        {/* Filled bar */}
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: isBullish ? `${center}%` : `${center - filled}%`,
            width: `${filled}%`,
            background: isBullish ? "var(--reflation)" : "var(--deflation)",
            borderRadius: 2,
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 10,
          color: "var(--muted)",
          marginTop: 4,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        <span>−1 (extreme selling)</span>
        <span>0</span>
        <span>+1 (extreme buying)</span>
      </div>
    </div>
  );
}

function TickerList({
  title,
  accent,
  items,
}: {
  title: string;
  accent: string;
  items: InsiderTickerVolume[];
}) {
  if (items.length === 0) {
    return (
      <div>
        <div
          style={{
            fontSize: 11,
            fontWeight: 700,
            color: accent,
            textTransform: "uppercase",
            letterSpacing: 0.5,
            marginBottom: 8,
          }}
        >
          {title}
        </div>
        <div style={{ fontSize: 12, color: "var(--muted)", fontStyle: "italic" }}>
          Nessun ticker
        </div>
      </div>
    );
  }
  const maxUsd = Math.max(...items.map((i) => i.usd));
  return (
    <div>
      <div
        style={{
          fontSize: 11,
          fontWeight: 700,
          color: accent,
          textTransform: "uppercase",
          letterSpacing: 0.5,
          marginBottom: 8,
        }}
      >
        {title} ({items.length})
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {items.map((p) => {
          const pct = (p.usd / maxUsd) * 100;
          return (
            <div
              key={p.ticker}
              style={{
                display: "grid",
                gridTemplateColumns: "60px 1fr 70px",
                alignItems: "center",
                gap: 8,
                padding: "6px 10px",
                background: "var(--bg)",
                borderRadius: 4,
              }}
            >
              <span style={{ fontSize: 12, fontWeight: 600 }}>{p.ticker}</span>
              <div
                style={{
                  height: 4,
                  background: "var(--surface-2)",
                  borderRadius: 2,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${pct}%`,
                    height: "100%",
                    background: accent,
                  }}
                />
              </div>
              <span
                style={{
                  fontSize: 11,
                  color: "var(--muted)",
                  fontVariantNumeric: "tabular-nums",
                  textAlign: "right",
                }}
              >
                {formatUsd(p.usd)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
