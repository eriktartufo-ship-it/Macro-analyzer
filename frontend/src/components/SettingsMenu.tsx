import { useEffect, useRef, useState } from "react";
import { useDedollarBonus } from "../hooks/useDedollarBonus";
import { FeatureFlagsPanel } from "./FeatureFlagsPanel";
import { LlmSettingsModal } from "./LlmSettingsModal";

/** Settings cog button + dropdown menu in alto a destra.
 * Sostituisce i 3 vecchi "pallini" (T5, $, ⚙ disperse).
 */
export function SettingsMenu() {
  const [open, setOpen] = useState(false);
  const [llmOpen, setLlmOpen] = useState(false);
  const [flagsOpen, setFlagsOpen] = useState(false);
  const [dedollarOn, setDedollarOn] = useDedollarBonus();
  const wrapRef = useRef<HTMLDivElement | null>(null);

  // Click outside to close dropdown
  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [open]);

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <button
        className="theme-toggle glass-active"
        onClick={() => setOpen((v) => !v)}
        aria-label="Impostazioni"
        title="Impostazioni"
        aria-expanded={open}
      >
        ⚙
      </button>

      {open && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 8px)",
            right: 0,
            minWidth: 240,
            background: "var(--bg)",
            border: "1px solid var(--stroke)",
            borderRadius: 10,
            padding: 6,
            zIndex: 90,
            boxShadow: "0 10px 32px rgba(0,0,0,0.25)",
            backdropFilter: "blur(20px)",
          }}
          role="menu"
        >
          <MenuItem
            label="🧠 LLM Settings"
            sub="API key + modello Gemini"
            onClick={() => {
              setLlmOpen(true);
              setOpen(false);
            }}
          />
          <MenuItem
            label="🚩 Feature Flags"
            sub="Toggle pilastri classifier"
            onClick={() => {
              setFlagsOpen(true);
              setOpen(false);
            }}
          />
          <MenuToggle
            label="💵 Dedollar bonus"
            sub={dedollarOn ? "ATTIVO: score con bias dedollar" : "Score puro data-driven"}
            on={dedollarOn}
            onChange={(v) => setDedollarOn(v)}
          />
        </div>
      )}

      <LlmSettingsModal open={llmOpen} onClose={() => setLlmOpen(false)} />
      <FeatureFlagsPanel open={flagsOpen} onClose={() => setFlagsOpen(false)} />
    </div>
  );
}

interface MenuItemProps {
  label: string;
  sub?: string;
  onClick: () => void;
}

function MenuItem({ label, sub, onClick }: MenuItemProps) {
  return (
    <button
      onClick={onClick}
      role="menuitem"
      style={{
        display: "block",
        width: "100%",
        textAlign: "left",
        padding: "10px 12px",
        background: "transparent",
        border: "none",
        borderRadius: 6,
        color: "var(--text)",
        cursor: "pointer",
        fontSize: 13,
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--surface-sunk)")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
    >
      <div style={{ fontWeight: 600 }}>{label}</div>
      {sub && (
        <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{sub}</div>
      )}
    </button>
  );
}

interface MenuToggleProps {
  label: string;
  sub?: string;
  on: boolean;
  onChange: (v: boolean) => void;
}

function MenuToggle({ label, sub, on, onChange }: MenuToggleProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 12px",
        borderRadius: 6,
      }}
    >
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 600, fontSize: 13, color: "var(--text)" }}>{label}</div>
        {sub && (
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{sub}</div>
        )}
      </div>
      <button
        onClick={() => onChange(!on)}
        role="switch"
        aria-checked={on}
        style={{
          width: 36,
          height: 20,
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
    </div>
  );
}
