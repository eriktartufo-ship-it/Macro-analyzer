/** Sidebar gestionale (desktop ≥900px) — brand + 5 sezioni per domanda + footer account.
 * Stessa struttura di CognitiveOS/RGV. Su mobile è nascosta (shell usa Header). */
import { formatDate, type Theme } from "./Header";
import { SettingsMenu } from "./SettingsMenu";
import { Icon } from "./Icon";
import { ptClearSession, usePtSession } from "../hooks/usePtSession";
import { HOME, SECTIONS, to, type Route } from "../lib/nav";

interface Props {
  date?: string;
  route: Route;
  navigate: (r: Route) => void;
  theme: Theme;
  onThemeToggle: () => void;
  onRefresh?: () => void; // S51: assente per chi non ha il token admin
  refreshing: boolean;
}

export function Sidebar({ date, route, navigate, theme, onThemeToggle, onRefresh, refreshing }: Props) {
  const user = usePtSession();
  const openPortfolio = () => navigate(to("portafogli", "miei"));

  return (
    <aside className="mac-sidebar">
      <button className="mac-brand" onClick={() => navigate(HOME)}>
        <span className="mac-mark" aria-hidden="true" />
        <span className="mac-brand-txt">
          <span className="mac-wordmark">Macro Analyzer</span>
          <span className="mac-brandsub">{date ? `Dati del ${formatDate(date)}` : "Regime e asset"}</span>
        </span>
      </button>

      <nav className="mac-nav" aria-label="Navigazione">
        {SECTIONS.map((s) => {
          const on = route.section === s.id;
          return (
            <button
              key={s.id}
              className={`mac-navlink ${on ? "on" : ""}`}
              aria-current={on ? "page" : undefined}
              onClick={() => navigate(to(s.id))}
            >
              <Icon name={s.icon} size={20} />
              <span className="mac-navtxt">
                <span>{s.label}</span>
                <span className="mac-navsub">{s.sub}</span>
              </span>
            </button>
          );
        })}
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
            <button className="mac-logout" onClick={ptClearSession} title="Esci dai Portafogli">
              Esci
            </button>
          </div>
        ) : (
          <button className="mac-signin" onClick={openPortfolio}>
            <Icon name="user" size={18} />
            Accedi ai Portafogli
          </button>
        )}

        <div className="mac-foot-actions">
          <SettingsMenu onOpenPortfolio={openPortfolio} placement="up" />
          <button
            className="mac-foot-btn"
            onClick={onThemeToggle}
            aria-label={theme === "dark" ? "Tema chiaro" : "Tema scuro"}
            title={theme === "dark" ? "Tema chiaro" : "Tema scuro"}
          >
            <Icon name={theme === "dark" ? "sun" : "moon"} size={18} />
          </button>
          {onRefresh && (
            <button
              className="mac-foot-btn"
              onClick={onRefresh}
              disabled={refreshing}
              aria-label="Aggiorna i dati"
              title="Aggiorna i dati"
            >
              <Icon name="refresh" size={18} />
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}
