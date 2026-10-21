/**
 * Interruttore on/off del Macro Analyzer (mai checkbox: convenzione UI della workspace).
 * Estratto da SettingsMenu il 2026-10-21 per riusarlo nel libro reale (toggle «Tasse»).
 */
export function Switch({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      role="switch"
      aria-checked={on}
      aria-label={label}
      style={{
        width: 36,
        height: 20,
        flex: "0 0 auto",
        background: on ? "var(--reflation)" : "var(--stroke)",
        border: "none",
        borderRadius: 12,
        position: "relative",
        cursor: "pointer",
        transition: "background 0.2s",
        padding: 0,
      }}
    >
      <span
        style={{
          position: "absolute",
          top: 2,
          left: on ? 18 : 2,
          width: 16,
          height: 16,
          background: "white",
          borderRadius: 8,
          transition: "left 0.2s",
          boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
        }}
      />
    </button>
  );
}
