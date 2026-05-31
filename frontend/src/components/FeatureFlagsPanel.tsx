import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ConfigFlags } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
}

const T5_FLAGS: { name: string; label: string; tier: string; description: string }[] = [
  {
    name: "USE_DEDOLLAR_PILLAR",
    label: "Dedollar pillar",
    tier: "T5.2",
    description: "Dedollar combined entra come pillar nel classifier (peso 0.02-0.03).",
  },
  {
    name: "USE_CRISIS_MODULATION",
    label: "Crisis modulation",
    tier: "T5.1",
    description: "Modula probs regime post-blend con crisis_type + risk_score.",
  },
  {
    name: "USE_NEWS_PILLAR",
    label: "News pillar",
    tier: "T5.3",
    description: "News avg sentiment entra come pillar classifier (peso 0.02).",
  },
  {
    name: "USE_DFM_ASSET_BONUS",
    label: "DFM asset bonus",
    tier: "T5.4",
    description: "gdp_yoy_dfm modula scoring asset (pro-cyclical bid, bonds penalty).",
  },
  {
    name: "USE_ASSET_FEEDBACK",
    label: "Asset feedback (sperimentale)",
    tier: "T5.6",
    description: "Bayesian update regime da 6m asset returns. Rischio lookahead.",
  },
];

const LEGACY_FLAGS: { name: string; label: string; description: string }[] = [
  {
    name: "USE_DEDOLLAR_BONUS",
    label: "Dedollar bonus (legacy)",
    description: "Secular bonus dedollar in asset scoring + trajectory.",
  },
  {
    name: "USE_CALIBRATED_SCORING",
    label: "Calibrated scoring",
    description: "Bayesian shrinkage da calibrated_asset_regime.json.",
  },
];

// T11-T17 (council 2026-05-28/31): features classifier accuracy + sizing + hedging
const COUNCIL_FLAGS: { name: string; label: string; tier: string; description: string }[] = [
  {
    name: "USE_MOMENTUM_PILLARS",
    label: "Momentum pillars",
    tier: "T11",
    description: "Pillar inflation_accelerating / core_pce_accelerating / gdp_decelerating. Default ON.",
  },
  {
    name: "USE_LIQUIDITY_SURGE_OVERRIDE",
    label: "Liquidity surge override",
    tier: "T11",
    description: "Force reflation se M2>+10% + WALCL ROC 3m>+12% + real rate accomm. (anti-inflation gate). Default ON.",
  },
  {
    name: "USE_UNCERTAINTY_GATE",
    label: "Uncertainty gate",
    tier: "T11",
    description: "Skip rebalance se conf < 0.30 (bypass su liquidity surge). Default ON.",
  },
  {
    name: "USE_FORWARD_INFLATION_PILLAR",
    label: "Forward inflation pillar",
    tier: "T13",
    description: "Pillar MICH/sticky CPI/T5YIFR fix structural inflation blindness. Default ON.",
  },
  {
    name: "USE_LABOR_CREDIT_PILLAR",
    label: "Labor + credit pillar",
    tier: "T17",
    description: "Pillar wage growth/JOLTS quits/avg hours/HY OAS spread (alpha alla radice). Default ON.",
  },
];

export function FeatureFlagsPanel({ open, onClose }: Props) {
  const [flags, setFlags] = useState<ConfigFlags | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    api
      .getConfigFlags()
      .then((data) => {
        if (!cancelled) {
          setFlags(data);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Fetch failed");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const toggle = async (name: string, currentValue: boolean) => {
    try {
      const result = await api.setConfigFlag(name, !currentValue);
      setFlags(result.flags);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Toggle failed");
    }
  };

  const clear = async (name: string) => {
    try {
      const result = await api.setConfigFlag(name, null);
      setFlags(result.flags);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Clear failed");
    }
  };

  if (!open) return null;

  return (
    <>
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "rgba(0,0,0,0.35)",
          zIndex: 100,
        }}
      />
      <div
        style={{
          position: "fixed",
          top: "10vh",
          left: "50%",
          transform: "translateX(-50%)",
          width: "min(560px, 92vw)",
          maxHeight: "80vh",
          overflowY: "auto",
          background: "var(--bg)",
          border: "1px solid var(--stroke)",
          borderRadius: 12,
          padding: 18,
          zIndex: 101,
          backdropFilter: "blur(20px)",
          boxShadow: "0 12px 48px rgba(0,0,0,0.3)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 12,
          }}
        >
          <h2 style={{ margin: 0 }}>Feature Flags</h2>
          <button
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              fontSize: 22,
              cursor: "pointer",
              color: "var(--muted)",
            }}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {loading && <div style={{ color: "var(--muted)" }}>Loading…</div>}
        {error && <div style={{ color: "var(--deflation)" }}>{error}</div>}

        {flags && (
          <>
            <SectionTitle>Council T11-T17 (accuracy + sizing + hedging)</SectionTitle>
            {COUNCIL_FLAGS.map((f) => (
              <FlagRow
                key={f.name}
                name={f.name}
                label={f.label}
                tier={f.tier}
                description={f.description}
                state={flags[f.name]}
                onToggle={toggle}
                onClear={clear}
              />
            ))}

            <SectionTitle>Tier 5 LOOP CLOSURE</SectionTitle>
            {T5_FLAGS.map((f) => (
              <FlagRow
                key={f.name}
                name={f.name}
                label={f.label}
                tier={f.tier}
                description={f.description}
                state={flags[f.name]}
                onToggle={toggle}
                onClear={clear}
              />
            ))}

            <SectionTitle>Legacy</SectionTitle>
            {LEGACY_FLAGS.map((f) => (
              <FlagRow
                key={f.name}
                name={f.name}
                label={f.label}
                description={f.description}
                state={flags[f.name]}
                onToggle={toggle}
                onClear={clear}
              />
            ))}

            <div
              style={{
                marginTop: 14,
                padding: "10px 12px",
                background: "var(--surface-sunk)",
                borderRadius: 6,
                fontSize: 11,
                color: "var(--muted)",
                lineHeight: 1.5,
              }}
            >
              <strong>Source priority</strong>: runtime &gt; env &gt; default.
              I toggle qui sono <strong>process-local</strong> (durano finché il backend
              resta up). Per persistenza permanente: env var nel container.
              Refresh data dopo aver toggleato per vedere l'effetto sul regime.
            </div>
          </>
        )}
      </div>
    </>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 10,
        fontWeight: 700,
        color: "var(--muted)",
        textTransform: "uppercase",
        letterSpacing: 0.6,
        margin: "16px 0 8px",
      }}
    >
      {children}
    </div>
  );
}

interface FlagRowProps {
  name: string;
  label: string;
  tier?: string;
  description: string;
  state: { value: boolean; source: string } | undefined;
  onToggle: (name: string, current: boolean) => void;
  onClear: (name: string) => void;
}

function FlagRow({ name, label, tier, description, state, onToggle, onClear }: FlagRowProps) {
  if (!state) return null;
  const active = state.value;
  const isRuntime = state.source === "runtime";
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto",
        gap: 10,
        padding: "10px 12px",
        background: active ? "var(--success-bg)" : "var(--surface-sunk)",
        border: `1px solid ${active ? "var(--reflation)" : "var(--stroke)"}`,
        borderRadius: 6,
        marginBottom: 6,
      }}
    >
      <div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          {tier && (
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                color: "var(--accent)",
                background: "var(--accent-bg)",
                padding: "2px 6px",
                borderRadius: 3,
              }}
            >
              {tier}
            </span>
          )}
          {label}
          <code style={{ fontSize: 10, color: "var(--muted)", marginLeft: 4 }}>{name}</code>
        </div>
        <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2, lineHeight: 1.4 }}>
          {description}
        </div>
        <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>
          source: <strong>{state.source}</strong>
          {isRuntime && (
            <>
              {" · "}
              <button
                onClick={() => onClear(name)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--accent)",
                  cursor: "pointer",
                  fontSize: 10,
                  padding: 0,
                  textDecoration: "underline",
                }}
              >
                clear override
              </button>
            </>
          )}
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center" }}>
        <button
          onClick={() => onToggle(name, active)}
          style={{
            background: active ? "var(--reflation)" : "var(--surface-2)",
            color: active ? "white" : "var(--muted)",
            border: "1px solid var(--stroke)",
            borderRadius: 999,
            padding: "6px 14px",
            fontSize: 11,
            fontWeight: 700,
            cursor: "pointer",
            minWidth: 60,
          }}
        >
          {active ? "ON" : "OFF"}
        </button>
      </div>
    </div>
  );
}
