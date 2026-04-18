export type Page = "dashboard" | "sentiment" | "dedollar" | "assets";
export type Theme = "light" | "dark";

interface Props {
  date?: string;
  onRefresh: () => void;
  refreshing: boolean;
  page: Page;
  onPageChange: (p: Page) => void;
  theme: Theme;
  onThemeToggle: () => void;
}

interface Tab {
  id: Page;
  label: string;
  short: string;
  icon: string;
}

const TABS: Tab[] = [
  { id: "dashboard", label: "Dashboard", short: "Home", icon: "▣" },
  { id: "sentiment", label: "Sentiment", short: "News", icon: "◈" },
  { id: "dedollar", label: "Dedollarizzazione", short: "USD", icon: "◉" },
  { id: "assets", label: "Asset Ranking", short: "Assets", icon: "≡" },
];

export function Header({ date, onRefresh, refreshing, page, onPageChange, theme, onThemeToggle }: Props) {
  return (
    <>
      <div className="header">
        <div style={{ minWidth: 0, flex: 1 }}>
          <h1>Macro Analyzer</h1>
          <div className="subtitle">
            {date ? `Last updated: ${date}` : "Regime classification & asset scoring"}
          </div>
        </div>
        <div className="header-actions">
          <button
            className="theme-toggle"
            onClick={onThemeToggle}
            aria-label={theme === "dark" ? "Passa al tema chiaro" : "Passa al tema scuro"}
            title={theme === "dark" ? "Tema chiaro" : "Tema scuro"}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
          <button className="btn" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? "Refreshing…" : "Refresh data"}
          </button>
        </div>
      </div>

      <div className="nav-tabs" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={page === tab.id}
            className={`nav-tab ${page === tab.id ? "active" : ""}`}
            onClick={() => onPageChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="nav-bottom" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={page === tab.id}
            className={`nav-bottom-tab ${page === tab.id ? "active" : ""}`}
            onClick={() => onPageChange(tab.id)}
          >
            <span className="icon">{tab.icon}</span>
            <span>{tab.short}</span>
          </button>
        ))}
      </div>
    </>
  );
}
