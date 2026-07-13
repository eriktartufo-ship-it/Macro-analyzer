import { useEffect, useRef, useState } from "react";
import { useDedollarBonus } from "../hooks/useDedollarBonus";
import { ptClearSession, usePtSession } from "../hooks/usePtSession";
import type { Page } from "./Header";
import { FeatureFlagsPanel } from "./FeatureFlagsPanel";
import { LlmSettingsModal } from "./LlmSettingsModal";

/** Settings cog button + dropdown: account (utente Portafogli + esci) +
 * impostazioni macro (LLM, flag, dedollar). Riusato in topbar (mobile) e
 * sidebar (desktop, placement="up"). */
export function SettingsMenu({
  onNavigate,
  placement = "down",
}: {
  onNavigate?: (p: Page) => void;
  placement?: "down" | "up";
}) {
  const [open, setOpen] = useState(false);
  const [llmOpen, setLlmOpen] = useState(false);
  const [flagsOpen, setFlagsOpen] = useState(false);
  const [dedollarOn, setDedollarOn] = useDedollarBonus();
  const user = usePtSession();
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

  // "up" = sidebar footer (in basso a sinistra) → apri verso l'alto ancorato a
  // sinistra. "down" = topbar mobile (in alto a destra) → verso il basso a destra.
  const posStyle =
    placement === "up"
      ? { bottom: "calc(100% + 8px)" as const, left: 0 as const }
      : { top: "calc(100% + 8px)" as const, right: 0 as const };

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <button
        className={placement === "up" ? "mac-foot-btn" : "theme-toggle glass-active"}
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
            ...posStyle,
            minWidth: 250,
            background: "var(--bg)",
            border: "1px solid var(--stroke)",
            borderRadius: 12,
            padding: 6,
            zIndex: 120,
            boxShadow: "0 10px 32px rgba(0,0,0,0.25)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
          }}
          role="menu"
        >
          {/* Account (area riservata = Portafogli) */}
          {user ? (
            <div className="mac-menu-account">
              <span className="mac-avatar" aria-hidden="true">
                {user.display_name.slice(0, 1).toUpperCase()}
              </span>
              <span className="mac-account-txt">
                <span className="mac-account-name">{user.display_name}</span>
                <span className="mac-account-mail">{user.username}</span>
              </span>
              <button
                className="mac-logout"
                onClick={() => {
                  ptClearSession();
                  setOpen(false);
                }}
              >
                Esci
              </button>
            </div>
          ) : (
            <MenuItem
              label="◆ Accedi ai Portafogli"
              sub="Il tuo portafoglio personale"
              onClick={() => {
                onNavigate?.("portfolio");
                setOpen(false);
              }}
            />
          )}

          <div className="mac-menu-divider" />

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
