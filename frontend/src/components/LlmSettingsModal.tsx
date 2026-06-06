import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { LlmSettings } from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  onSaved?: () => void;
}

export function LlmSettingsModal({ open, onClose, onSaved }: Props) {
  const [state, setState] = useState<LlmSettings | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [modelInput, setModelInput] = useState("");
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .llmSettings()
      .then((d) => {
        if (cancelled) return;
        setState(d);
        setApiKeyInput("");
        setModelInput(d.model);
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

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const payload: { api_key?: string; model?: string } = {};
      if (apiKeyInput.trim()) payload.api_key = apiKeyInput.trim();
      if (modelInput && modelInput !== state?.model) payload.model = modelInput;
      if (Object.keys(payload).length === 0) {
        setSaving(false);
        return;
      }
      const updated = await api.llmSettingsUpdate(payload);
      setState(updated);
      setApiKeyInput("");
      if (onSaved) onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const handleClearKey = async () => {
    if (!confirm("Cancellare la API key runtime? Tornerà a usare la chiave env var.")) return;
    setSaving(true);
    setError(null);
    try {
      const updated = await api.llmSettingsUpdate({ api_key: "" });
      setState(updated);
      setApiKeyInput("");
      if (onSaved) onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Clear failed");
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  return (
    <>
      <div
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.35)",
          zIndex: 100,
        }}
      />
      <div
        style={{
          position: "fixed",
          top: "8vh",
          left: "50%",
          transform: "translateX(-50%)",
          width: "min(520px, 92vw)",
          maxHeight: "84vh",
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
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>🧠 LLM Settings — Gemini Analisi Macro</h2>
          <button
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              fontSize: 22,
              color: "var(--muted)",
              cursor: "pointer",
              lineHeight: 1,
            }}
          >
            ×
          </button>
        </div>

        {loading && <div style={{ color: "var(--muted)" }}>Caricamento…</div>}
        {error && (
          <div style={{ color: "var(--deflation)", marginBottom: 10, fontSize: 13 }}>{error}</div>
        )}

        {state && !loading && (
          <>
            <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5, marginBottom: 16 }}>
              Modifica runtime di API key e modello Gemini per l'analisi macro AI. Le impostazioni
              sono persistite su disco. Priorità: <strong>runtime → env var → globale</strong>.
            </div>

            {/* API KEY section */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 12, fontWeight: 700, display: "block", marginBottom: 6, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>
                API Key (Gemini)
              </label>
              <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
                Stato: {state.api_key_set ? (
                  <>
                    <span style={{ color: "var(--reflation)" }}>● configurata</span>{" "}
                    <span style={{ fontFamily: "monospace", fontSize: 11 }}>
                      ({state.api_key_masked})
                    </span>{" "}
                    via <strong>{sourceLabel(state.source_api_key)}</strong>
                  </>
                ) : (
                  <span style={{ color: "var(--deflation)" }}>● mancante</span>
                )}
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <input
                  type={showKey ? "text" : "password"}
                  placeholder="AIza... (incolla nuova API key)"
                  value={apiKeyInput}
                  onChange={(e) => setApiKeyInput(e.target.value)}
                  style={{
                    flex: 1,
                    padding: "8px 10px",
                    background: "var(--surface-sunk)",
                    border: "1px solid var(--stroke)",
                    borderRadius: 6,
                    color: "var(--text)",
                    fontSize: 13,
                    fontFamily: "monospace",
                  }}
                />
                <button
                  onClick={() => setShowKey((v) => !v)}
                  style={{
                    background: "var(--surface-sunk)",
                    border: "1px solid var(--stroke)",
                    borderRadius: 6,
                    padding: "0 10px",
                    color: "var(--muted)",
                    cursor: "pointer",
                    fontSize: 12,
                  }}
                  title={showKey ? "Nascondi" : "Mostra"}
                >
                  {showKey ? "🙈" : "👁"}
                </button>
              </div>
              {state.source_api_key === "runtime" && (
                <button
                  onClick={handleClearKey}
                  disabled={saving}
                  style={{
                    marginTop: 6,
                    background: "transparent",
                    border: "none",
                    color: "var(--deflation)",
                    fontSize: 11,
                    cursor: "pointer",
                    textDecoration: "underline",
                  }}
                >
                  Cancella runtime key (torna a env var)
                </button>
              )}
            </div>

            {/* MODEL section */}
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 12, fontWeight: 700, display: "block", marginBottom: 6, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>
                Modello Gemini
              </label>
              <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
                Attuale: <strong style={{ color: "var(--text)" }}>{state.model}</strong> via{" "}
                <strong>{sourceLabel(state.source_model)}</strong>
              </div>
              <select
                value={modelInput}
                onChange={(e) => setModelInput(e.target.value)}
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  background: "var(--surface-sunk)",
                  border: "1px solid var(--stroke)",
                  borderRadius: 6,
                  color: "var(--text)",
                  fontSize: 13,
                }}
              >
                {state.models_available.map((m) => (
                  <option key={m} value={m}>
                    {m}
                    {m === state.model_default ? " (default)" : ""}
                  </option>
                ))}
              </select>
            </div>

            {/* Actions */}
            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 16 }}>
              <button
                onClick={onClose}
                style={{
                  background: "transparent",
                  border: "1px solid var(--stroke)",
                  borderRadius: 6,
                  padding: "8px 16px",
                  color: "var(--muted)",
                  cursor: "pointer",
                }}
              >
                Annulla
              </button>
              <button
                onClick={handleSave}
                disabled={saving || (!apiKeyInput.trim() && modelInput === state.model)}
                style={{
                  background: "var(--accent)",
                  border: "none",
                  borderRadius: 6,
                  padding: "8px 16px",
                  color: "white",
                  cursor: saving ? "default" : "pointer",
                  opacity: saving || (!apiKeyInput.trim() && modelInput === state.model) ? 0.5 : 1,
                  fontWeight: 600,
                }}
              >
                {saving ? "Salvando…" : "Salva"}
              </button>
            </div>

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
              <strong>Caching intelligente</strong>: l'analisi viene rigenerata SOLO quando i dati
              macro cambiano effettivamente (hash MD5 di regime + indicators + scoreboard +
              modello). Se i dati sono uguali, la cache risponde senza chiamare Gemini.
            </div>
          </>
        )}
      </div>
    </>
  );
}

function sourceLabel(src: string): string {
  switch (src) {
    case "runtime":
      return "runtime (file disco)";
    case "env_macro":
      return "env GEMINI_*_MACRO";
    case "env_global":
      return "env GEMINI_API_KEY";
    case "default":
      return "default (hardcoded)";
    case "none":
      return "nessuna fonte";
    default:
      return src;
  }
}
