import { useState } from "react";
import type {
  CrisisRiskAssessment,
  CrisisRiskLevel,
  CrisisType,
  PositioningStance,
} from "../types";

interface Props {
  data: CrisisRiskAssessment;
}

const RISK_COLORS: Record<CrisisRiskLevel, { bg: string; fg: string; label: string }> = {
  low: { bg: "var(--success-bg)", fg: "var(--reflation)", label: "BASSO" },
  elevated: { bg: "var(--warn-bg)", fg: "var(--warn-text)", label: "ELEVATO" },
  high: { bg: "var(--danger-bg)", fg: "var(--deflation)", label: "ALTO" },
  extreme: { bg: "var(--danger-bg)", fg: "var(--deflation)", label: "ESTREMO" },
};

const CRISIS_TYPE_META: Record<CrisisType, { label: string; tag: string; tagBg: string }> = {
  no_crisis: {
    label: "Nessuna crisi imminente",
    tag: "OK",
    tagBg: "var(--success-bg)",
  },
  deflation_crash: {
    label: "Deflation Crash (2008-style)",
    tag: "DEFL",
    tagBg: "var(--danger-bg)",
  },
  stagflation_debasement: {
    label: "Stagflation / Debasement (1973-style)",
    tag: "STAG",
    tagBg: "var(--warn-bg)",
  },
  bubble_goldilocks: {
    label: "Bubble / Goldilocks complacency (2000-style)",
    tag: "BUBBLE",
    tagBg: "var(--accent-bg)",
  },
};

function formatAsset(name: string): string {
  return name.replace(/_/g, " ");
}

function stanceWeight(s: PositioningStance): number {
  if (s === "overweight") return 1;
  if (s === "underweight") return -1;
  return 0;
}

function stanceColor(s: PositioningStance): string {
  if (s === "overweight") return "var(--reflation)";
  if (s === "underweight") return "var(--deflation)";
  return "var(--muted)";
}

function stanceLabel(s: PositioningStance): string {
  if (s === "overweight") return "OW";
  if (s === "underweight") return "UW";
  return "Neutral";
}

export function CrisisRiskPanel({ data }: Props) {
  const [expanded, setExpanded] = useState(false);

  const risk = RISK_COLORS[data.risk_level];
  const crisisMeta = CRISIS_TYPE_META[data.crisis_type];

  const positioningSorted = Object.entries(data.positioning)
    .map(([asset, stance]) => ({ asset, stance, weight: stanceWeight(stance) }))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight) || a.asset.localeCompare(b.asset));

  const overweights = positioningSorted.filter((p) => p.stance === "overweight");
  const underweights = positioningSorted.filter((p) => p.stance === "underweight");
  const isNoCrisis = data.crisis_type === "no_crisis";

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
        <h2 style={{ margin: 0 }}>Crisis Risk Indicator</h2>
        <span style={{ fontSize: 12, color: "var(--muted)" }}>As of {data.date}</span>
      </div>

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
            background: risk.bg,
            color: risk.fg,
            fontSize: 12,
            padding: "5px 12px",
            fontWeight: 700,
            letterSpacing: 0.5,
          }}
        >
          RISCHIO {risk.label}
        </span>
        <span
          className="chip"
          style={{
            background: crisisMeta.tagBg,
            color: "var(--text, inherit)",
            fontSize: 11,
            padding: "5px 10px",
          }}
        >
          {crisisMeta.tag}
        </span>
        <span style={{ fontSize: 13, color: "var(--muted)", fontWeight: 500 }}>
          {crisisMeta.label}
        </span>
        {data.historical_analog && (
          <span
            style={{
              fontSize: 11,
              color: "var(--muted)",
              fontStyle: "italic",
            }}
          >
            analog: {data.historical_analog}
          </span>
        )}
      </div>

      <div
        style={{
          fontSize: 13,
          color: "var(--muted)",
          lineHeight: 1.5,
          marginBottom: 14,
        }}
      >
        {data.summary}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: 10,
          marginBottom: 12,
        }}
      >
        <Metric label="Risk score" value={`${(data.risk_score * 100).toFixed(0)}%`} />
        <Metric
          label="Confidence tipo"
          value={`${(data.crisis_type_confidence * 100).toFixed(0)}%`}
        />
        <Metric
          label="Triggers fired"
          value={`${data.triggers_fired.length} / ${data.triggers_all.length}`}
        />
      </div>

      {!isNoCrisis && (overweights.length > 0 || underweights.length > 0) && (
        <div className="grid grid-2" style={{ gap: 16, marginBottom: 8 }}>
          <PositioningList
            title="Overweight"
            accent="var(--reflation)"
            items={overweights.slice(0, 6)}
          />
          <PositioningList
            title="Underweight"
            accent="var(--deflation)"
            items={underweights.slice(0, 6)}
          />
        </div>
      )}

      <button
        type="button"
        className="btn-ghost"
        onClick={() => setExpanded((v) => !v)}
        style={{
          marginTop: 4,
          fontSize: 12,
          padding: "6px 12px",
          background: "transparent",
          border: "1px solid var(--stroke)",
          borderRadius: 6,
          cursor: "pointer",
          color: "var(--muted)",
        }}
      >
        {expanded ? "Nascondi dettagli" : "Mostra dettagli (triggers + positioning completo)"}
      </button>

      {expanded && (
        <div style={{ marginTop: 14 }}>
          <SectionTitle>Triggers ({data.triggers_fired.length} attivi)</SectionTitle>
          <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 14 }}>
            {data.triggers_all.map((t) => (
              <div
                key={t.name}
                style={{
                  display: "grid",
                  gridTemplateColumns: "auto 1fr auto auto",
                  alignItems: "center",
                  gap: 10,
                  padding: "6px 10px",
                  background: t.fired ? "var(--danger-bg)" : "var(--surface-sunk)",
                  borderRadius: 4,
                  opacity: t.fired ? 1 : 0.55,
                }}
              >
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    color: t.fired ? "var(--deflation)" : "var(--muted)",
                    minWidth: 30,
                  }}
                >
                  {t.fired ? "FIRE" : "—"}
                </span>
                <span style={{ fontSize: 12 }}>{t.description}</span>
                <span
                  style={{
                    fontSize: 11,
                    color: "var(--muted)",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {t.value.toFixed(2)} {t.operator === "ge" ? "≥" : "≤"} {t.threshold}
                </span>
                <code
                  style={{
                    fontSize: 10,
                    color: "var(--muted)",
                    fontFamily: "ui-monospace, monospace",
                  }}
                >
                  {t.name}
                </code>
              </div>
            ))}
          </div>

          <SectionTitle>Positioning completo (15 asset class)</SectionTitle>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
              gap: 6,
              marginBottom: 8,
            }}
          >
            {positioningSorted.map((p) => (
              <div
                key={p.asset}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "6px 10px",
                  background: "var(--surface-sunk)",
                  borderRadius: 4,
                  fontSize: 12,
                }}
              >
                <span style={{ textTransform: "capitalize" }}>{formatAsset(p.asset)}</span>
                <span
                  style={{
                    fontWeight: 700,
                    fontSize: 11,
                    color: stanceColor(p.stance),
                  }}
                >
                  {stanceLabel(p.stance)}
                </span>
              </div>
            ))}
          </div>

          <div
            style={{
              fontSize: 11,
              color: "var(--muted)",
              marginTop: 8,
              fontStyle: "italic",
            }}
          >
            Dedollar combined: {data.dedollar_combined.toFixed(2)}
            {data.notes.length > 0 && <> · {data.notes.join(" · ")}</>}
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
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
      <div style={{ fontSize: 15, fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 11,
        fontWeight: 700,
        color: "var(--muted)",
        textTransform: "uppercase",
        letterSpacing: 0.5,
        marginBottom: 8,
      }}
    >
      {children}
    </div>
  );
}

interface PositioningListProps {
  title: string;
  accent: string;
  items: { asset: string; stance: PositioningStance }[];
}

function PositioningList({ title, accent, items }: PositioningListProps) {
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
          Nessun asset
        </div>
      </div>
    );
  }
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
        {items.map((p) => (
          <div
            key={p.asset}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "6px 10px",
              background: "var(--bg)",
              borderRadius: 4,
              fontSize: 13,
            }}
          >
            <span style={{ textTransform: "capitalize" }}>{formatAsset(p.asset)}</span>
            <span style={{ color: accent, fontSize: 11, fontWeight: 700 }}>
              {stanceLabel(p.stance)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
