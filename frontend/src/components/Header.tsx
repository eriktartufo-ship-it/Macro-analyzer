import { useRef, useEffect, useState } from "react";

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
  const [pillStyle, setPillStyle] = useState({ left: 0, width: 0, opacity: 0 });
  const [mobilePillStyle, setMobilePillStyle] = useState({ left: 0, width: 0, opacity: 0 });
  
  const tabsRef = useRef<(HTMLButtonElement | null)[]>([]);
  const mobileTabsRef = useRef<(HTMLButtonElement | null)[]>([]);

  // Update highlighter position when tab changes
  useEffect(() => {
    const activeIndex = TABS.findIndex(t => t.id === page);
    const activeTab = tabsRef.current[activeIndex];
    const activeMobileTab = mobileTabsRef.current[activeIndex];
    
    // We use a small timeout to ensure DOM layout is complete before measuring
    const timeout = setTimeout(() => {
        if (activeTab) {
          setPillStyle({
            left: activeTab.offsetLeft,
            width: activeTab.clientWidth,
            opacity: 1
          });
        }
        if (activeMobileTab) {
          setMobilePillStyle({
            left: activeMobileTab.offsetLeft,
            width: activeMobileTab.clientWidth,
            opacity: 1
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

      <div className="nav-tabs" role="tablist">
        <div 
          className="nav-tab-highlight" 
          style={{
            left: pillStyle.left,
            width: pillStyle.width,
            opacity: pillStyle.opacity
          }} 
        />
        {TABS.map((tab, i) => (
          <button
            key={tab.id}
            ref={el => { tabsRef.current[i] = el; }}
            role="tab"
            aria-selected={page === tab.id}
            className={`nav-tab ${page === tab.id ? "active" : ""}`}
            onClick={() => onPageChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="nav-bottom" role="tablist" style={{ position: "fixed" }}>
        <div 
          className="nav-bottom-highlight" 
          style={{
            left: mobilePillStyle.left,
            width: mobilePillStyle.width,
            opacity: mobilePillStyle.opacity
          }} 
        />
        {TABS.map((tab, i) => (
          <button
            key={tab.id}
            ref={el => { mobileTabsRef.current[i] = el; }}
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
