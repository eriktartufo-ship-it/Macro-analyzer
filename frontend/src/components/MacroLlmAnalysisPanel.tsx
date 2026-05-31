import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { MacroLlmAnalysis } from "../types";

const REGIME_COLORS: Record<string, string> = {
  reflation: "var(--reflation)",
  stagflation: "var(--stagflation)",
  deflation: "var(--deflation)",
  goldilocks: "var(--goldilocks)",
};

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "var(--reflation)",
  medium: "var(--accent)",
  low: "var(--stagflation)",
};

function timeAgo(iso: string): string {
  try {
    const ts = new Date(iso).getTime();
    const diffMs = Date.now() - ts;
    const hours = Math.floor(diffMs / 3600_000);
    if (hours < 1) {
      const minutes = Math.max(1, Math.floor(diffMs / 60_000));
      return `${minutes} min fa`;
    }
    if (hours < 24) return `${hours}h fa`;
    return `${Math.floor(hours / 24)}g fa`;
  } catch {
    return "—";
  }
}

export function MacroLlmAnalysisPanel() {
  const [data, setData] = useState<MacroLlmAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = async (forceRefresh = false) => {
    if (forceRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const result = await api.macroLlmAnalysis(forceRefresh);
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analisi non disponibile");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    load(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading) {
    return (
      <div className="card" style={{ marginTop: 16 }}>
        <h2 style={{ margin: 0, marginBottom: 6 }}>🧠 Analisi macro AI</h2>
        <div style={{ color: "var(--muted)", fontSize: 13 }}>Generazione analisi in corso…</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="card" style={{ marginTop: 16 }}>
        <h2 style={{ margin: 0, marginBottom: 6 }}>🧠 Analisi macro AI</h2>
        <div style={{ color: "var(--muted)", fontSize: 13 }}>
          {error ?? "Analisi non disponibile (Gemini API key mancante o errore di rete)."}
        </div>
      </div>
    );
  }

  const regimeColor = REGIME_COLORS[data.regime] || "var(--accent)";
  const confColor = CONFIDENCE_COLORS[data.confidence_qualitative.toLowerCase()] || "var(--muted)";

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 12,
          marginBottom: 4,
        }}
      >
        <h2 style={{ margin: 0 }}>
          🧠 Analisi macro AI
          <span
            style={{
              fontSize: 11,
              fontWeight: 500,
              color: "var(--muted)",
              marginLeft: 8,
              textTransform: "none",
            }}
          >
            Gemini 2.5 Flash · {timeAgo(data.cached_at)}
            {data.stale && " · stale"}
          </span>
        </h2>
        <button
          onClick={() => load(true)}
          disabled={refreshing}
          style={{
            background: "var(--surface-sunk)",
            border: "1px solid var(--stroke)",
            color: "var(--text)",
            borderRadius: 6,
            padding: "4px 10px",
            fontSize: 11,
            cursor: refreshing ? "default" : "pointer",
            opacity: refreshing ? 0.5 : 1,
          }}
          title="Forza re-generazione (bypass cache 24h)"
        >
          {refreshing ? "..." : "↻ Rigenera"}
        </button>
      </div>

      {/* Headline */}
      <div
        style={{
          fontSize: 16,
          fontWeight: 600,
          lineHeight: 1.4,
          padding: "10px 12px",
          background: "var(--surface-sunk)",
          borderLeft: `3px solid ${regimeColor}`,
          borderRadius: 6,
          marginTop: 8,
          marginBottom: 12,
        }}
      >
        {data.headline}
      </div>

      {/* Regime thesis */}
      {data.regime_thesis && (
        <p
          style={{
            fontSize: 13,
            color: "var(--muted)",
            lineHeight: 1.5,
            margin: "0 0 14px 0",
          }}
        >
          {data.regime_thesis}
        </p>
      )}

      {/* 3 colonne: drivers / risks / opportunities */}
      <div className="grid grid-3" style={{ gap: 12, marginBottom: 10 }}>
        <AnalysisColumn
          title="Driver chiave"
          items={data.key_drivers}
          accent="var(--reflation)"
          icon="📊"
        />
        <AnalysisColumn
          title="Rischi"
          items={data.risks}
          accent="var(--deflation)"
          icon="⚠️"
        />
        <AnalysisColumn
          title="Opportunità"
          items={data.opportunities}
          accent="var(--goldilocks)"
          icon="💡"
        />
      </div>

      {/* Footer: horizon + confidence */}
      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          flexWrap: "wrap",
          fontSize: 11,
          color: "var(--muted)",
          paddingTop: 10,
          borderTop: "1px solid var(--stroke)",
        }}
      >
        <span>
          Orizzonte: <strong style={{ color: "var(--text)" }}>{data.time_horizon}</strong>
        </span>
        <span>
          Confidence:{" "}
          <strong style={{ color: confColor }}>{data.confidence_qualitative}</strong>
        </span>
        <span style={{ marginLeft: "auto", fontSize: 10 }}>
          Cache 24h · {data.cache_hit ? "served from cache" : "fresh"}
        </span>
      </div>
    </div>
  );
}

interface ColumnProps {
  title: string;
  items: string[];
  accent: string;
  icon: string;
}

function AnalysisColumn({ title, items, accent, icon }: ColumnProps) {
  return (
    <div
      style={{
        padding: "10px 12px",
        background: "var(--surface-sunk)",
        borderRadius: 6,
        borderTop: `2px solid ${accent}`,
      }}
    >
      <div
        style={{
          fontSize: 11,
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: 0.6,
          color: "var(--muted)",
          marginBottom: 8,
        }}
      >
        {icon} {title}
      </div>
      {items.length === 0 ? (
        <div style={{ fontSize: 12, color: "var(--muted)", fontStyle: "italic" }}>—</div>
      ) : (
        <ul style={{ margin: 0, padding: "0 0 0 16px", fontSize: 12, lineHeight: 1.5 }}>
          {items.map((item, i) => (
            <li key={i} style={{ marginBottom: 4 }}>
              {item}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
