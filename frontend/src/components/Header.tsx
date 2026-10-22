import { SettingsMenu } from "./SettingsMenu";
import { Icon } from "./Icon";
import { SECTIONS, to, type Route } from "../lib/nav";

export type Theme = "light" | "dark";

interface Props {
  date?: string;
  onRefresh?: () => void; // S51: assente per chi non ha il token admin
  refreshing: boolean;
  route: Route;
  navigate: (r: Route) => void;
  theme: Theme;
  onThemeToggle: () => void;
}

/** Header MOBILE (topbar + barra in basso a 5 voci). Su desktop (≥900px) lo shell
 * usa la Sidebar e questo header è nascosto via CSS. */
export function Header({ date, onRefresh, refreshing, route, navigate, theme, onThemeToggle }: Props) {
  const current = SECTIONS.find((s) => s.id === route.section) ?? SECTIONS[0];

  return (
    <>
      <div className="header">
        <div className="header-titles">
          <h1>{current.label}</h1>
          <div className="subtitle">{date ? `Dati del ${formatDate(date)}` : "Macro Analyzer"}</div>
        </div>
        <div className="header-actions">
          <SettingsMenu onOpenPortfolio={() => navigate(to("portafogli", "miei"))} />
          <button
            className="theme-toggle glass-active"
            onClick={onThemeToggle}
            aria-label={theme === "dark" ? "Passa al tema chiaro" : "Passa al tema scuro"}
            title={theme === "dark" ? "Tema chiaro" : "Tema scuro"}
          >
            <Icon name={theme === "dark" ? "sun" : "moon"} size={18} />
          </button>
          {onRefresh && (
            <button
              className="theme-toggle glass-active"
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

      <nav className="nav-bottom" aria-label="Navigazione">
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            aria-current={route.section === s.id ? "page" : undefined}
            className={`nav-bottom-tab ${route.section === s.id ? "active" : ""}`}
            onClick={() => navigate(to(s.id))}
          >
            <Icon name={s.icon} size={21} />
            <span>{s.label}</span>
          </button>
        ))}
      </nav>
    </>
  );
}

/** "2026-10-22" → "22/10" */
export function formatDate(iso: string): string {
  const [, m, d] = iso.split("-");
  return d && m ? `${d}/${m}` : iso;
}
