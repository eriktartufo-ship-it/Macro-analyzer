/** Sidebar gestionale (desktop ≥900px) — brand + nav verticale + footer account.
 * Stessa struttura di CognitiveOS/RGV. Su mobile è nascosta (shell usa Header). */
import { TABS, type Page, type Theme } from "./Header";
import { SettingsMenu } from "./SettingsMenu";
import { ptClearSession, usePtSession } from "../hooks/usePtSession";

interface Props {
  date?: string;
  page: Page;
  onPageChange: (p: Page) => void;
  theme: Theme;
  onThemeToggle: () => void;
  onRefresh: () => void;
  refreshing: boolean;
}

export function Sidebar({
  date,
  page,
  onPageChange,
  theme,
  onThemeToggle,
  onRefresh,
  refreshing,
}: Props) {
  const user = usePtSession();

  return (
    <aside className="mac-sidebar">
      <button className="mac-brand" onClick={() => onPageChange("dashboard")}>
        <span className="mac-mark" aria-hidden="true" />
        <span className="mac-brand-txt">
          <span className="mac-wordmark">Macro Analyzer</span>
          <span className="mac-brandsub">
            {date ? `Aggiornato ${date}` : "Regime & scoring"}
          </span>
        </span>
      </button>

      <nav className="mac-nav" aria-label="Navigazione">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`mac-navlink ${page === t.id ? "on" : ""}`}
            aria-current={page === t.id ? "page" : undefined}
            onClick={() => onPageChange(t.id)}
          >
            <span className="mac-navic" aria-hidden="true">
              {t.icon}
            </span>
            {t.label}
          </button>
        ))}
      </nav>

      <div className="mac-side-foot">
        {user ? (
          <div className="mac-account">
            <span className="mac-avatar" aria-hidden="true">
              {user.display_name.slice(0, 1).toUpperCase()}
            </span>
            <span className="mac-account-txt">
              <span className="mac-account-name">{user.display_name}</span>
              <span className="mac-account-mail">{user.username}</span>
            </span>
            <button
              className="mac-logout"
              onClick={ptClearSession}
              title="Esci dai Portafogli"
            >
              Esci
            </button>
          </div>
        ) : (
          <button className="mac-signin" onClick={() => onPageChange("portfolio")}>
            <span className="mac-navic" aria-hidden="true">
              ◆
            </span>
            Accedi ai Portafogli
          </button>
        )}

        <div className="mac-foot-actions">
          <SettingsMenu onNavigate={onPageChange} placement="up" />
          <button
            className="mac-foot-btn"
            onClick={onThemeToggle}
            title={theme === "dark" ? "Tema chiaro" : "Tema scuro"}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
          <button
            className="mac-foot-btn"
            onClick={onRefresh}
            disabled={refreshing}
            title="Aggiorna dati"
          >
            {refreshing ? "…" : "⟳"}
          </button>
        </div>
      </div>
    </aside>
  );
}
