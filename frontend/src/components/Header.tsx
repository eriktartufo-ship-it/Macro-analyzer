import { useRef, useEffect, useState } from "react";
import { SettingsMenu } from "./SettingsMenu";

export type Page = "dashboard" | "ai-portfolio" | "flows" | "portfolio" | "sentiment" | "dedollar" | "assets" | "data";
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

export interface Tab {
  id: Page;
  label: string;
  short: string;
  icon: string;
}

export const TABS: Tab[] = [
  { id: "dashboard", label: "Dashboard", short: "Home", icon: "▣" },
  { id: "ai-portfolio", label: "AI Portfolio", short: "AI", icon: "🤖" },
  { id: "flows", label: "Flussi", short: "Flussi", icon: "⇄" },
  { id: "portfolio", label: "Portafogli", short: "Miei", icon: "◆" },
  { id: "sentiment", label: "Sentiment", short: "News", icon: "◈" },
  { id: "dedollar", label: "Dedollarizzazione", short: "USD", icon: "◉" },
  { id: "assets", label: "Asset Ranking", short: "Assets", icon: "≡" },
  { id: "data", label: "Data", short: "Data", icon: "▤" },
];

/** Header MOBILE (topbar + bottom-nav). Su desktop (≥900px) lo shell usa la
 * Sidebar e questo header è nascosto via CSS. */
export function Header({ date, onRefresh, refreshing, page, onPageChange, theme, onThemeToggle }: Props) {
  const [mobilePillStyle, setMobilePillStyle] = useState({ left: 0, width: 0, opacity: 0 });
  const mobileTabsRef = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    const activeIndex = TABS.findIndex((t) => t.id === page);
    const activeMobileTab = mobileTabsRef.current[activeIndex];
    const timeout = setTimeout(() => {
      if (activeMobileTab) {
        setMobilePillStyle({
          left: activeMobileTab.offsetLeft,
          width: activeMobileTab.clientWidth,
          opacity: 1,
        });
      }
    }, 10);
    return () => clearTimeout(timeout);
  }, [page]);

  return (
    <>
      <div className="header">
        <div className="header-titles">
          <h1>Macro Analyzer</h1>
          <div className="subtitle">
            {date ? `Last updated: ${date}` : "Regime classification & asset scoring"}
          </div>
        </div>
        <div className="header-actions">
          <SettingsMenu onNavigate={onPageChange} />
          <button
            className="theme-toggle glass-active"
            onClick={onThemeToggle}
            aria-label={theme === "dark" ? "Passa al tema chiaro" : "Passa al tema scuro"}
            title={theme === "dark" ? "Tema chiaro" : "Tema scuro"}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
          <button className="btn glass-btn" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? "Refreshing…" : "Refresh data"}
          </button>
        </div>
      </div>

      <nav className="nav-bottom" role="tablist" style={{ position: "fixed" }}>
        <div
          className="nav-bottom-highlight"
          style={{
            left: mobilePillStyle.left,
            width: mobilePillStyle.width,
            opacity: mobilePillStyle.opacity,
          }}
        />
        {TABS.map((tab, i) => (
          <button
            key={tab.id}
            ref={(el) => {
              mobileTabsRef.current[i] = el;
            }}
            role="tab"
            aria-selected={page === tab.id}
            className={`nav-bottom-tab ${page === tab.id ? "active" : ""}`}
            onClick={() => onPageChange(tab.id)}
          >
            <span className="icon">{tab.icon}</span>
            <span>{tab.short}</span>
          </button>
        ))}
      </nav>
    </>
  );
}
