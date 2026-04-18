export type Page = "dashboard" | "sentiment" | "dedollar" | "assets";

interface Props {
  date?: string;
  onRefresh: () => void;
  refreshing: boolean;
  page: Page;
  onPageChange: (p: Page) => void;
}

const TABS: { id: Page; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "sentiment", label: "Sentiment" },
  { id: "dedollar", label: "Dedollarizzazione" },
  { id: "assets", label: "Asset Ranking" },
];

export function Header({ date, onRefresh, refreshing, page, onPageChange }: Props) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div className="header">
        <div>
          <h1>Macro Analyzer</h1>
          <div className="subtitle">
            {date ? `Last updated: ${date}` : "Regime classification & asset scoring"}
          </div>
        </div>
        <button className="btn" onClick={onRefresh} disabled={refreshing}>
          {refreshing ? "Refreshing..." : "Refresh data"}
        </button>
      </div>
      <div
        style={{
          display: "flex",
          gap: 4,
          marginTop: 12,
          borderBottom: "1px solid var(--border)",
        }}
      >
        {TABS.map((tab) => {
          const active = page === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onPageChange(tab.id)}
              style={{
                padding: "10px 18px",
                fontSize: 13,
                fontWeight: 600,
                background: "transparent",
                color: active ? "var(--accent)" : "var(--muted)",
                border: "none",
                borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
                marginBottom: -1,
                cursor: "pointer",
              }}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
