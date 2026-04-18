import type { NewsItem } from "../types";

interface Props {
  news: NewsItem[];
}

function sentimentColor(s: number): string {
  if (s > 0.3) return "var(--reflation)";
  if (s > 0.1) return "#6ee7b7";
  if (s > -0.1) return "var(--muted)";
  if (s > -0.3) return "var(--deflation)";
  return "var(--deflation)";
}

function sentimentLabel(s: number): string {
  if (s > 0.5) return "Very Bullish";
  if (s > 0.2) return "Bullish";
  if (s > -0.2) return "Neutral";
  if (s > -0.5) return "Bearish";
  return "Very Bearish";
}

function formatAsset(name: string): string {
  return name.replace(/_/g, " ");
}

const SOURCE_LABELS: Record<string, string> = {
  reuters_markets: "Reuters",
  cnbc_economy: "CNBC",
  fed_press: "Fed",
  ecb_press: "ECB",
  ft_markets: "FT",
};

export function NewsPanel({ news }: Props) {
  if (news.length === 0) {
    return (
      <div className="card">
        <h2>News Sentiment</h2>
        <div style={{ color: "var(--muted)", fontSize: 13, padding: "12px 0" }}>
          No news data available yet. Add your GROQ_API_KEY to .env and run a refresh.
        </div>
      </div>
    );
  }

  // Summary: average sentiment
  const avgSentiment = news.reduce((s, n) => s + n.sentiment * n.relevance, 0)
    / Math.max(news.reduce((s, n) => s + n.relevance, 0), 0.01);

  return (
    <div className="card">
      <h2>News Sentiment</h2>

      {/* Summary bar */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        marginBottom: 20,
        padding: "10px 12px",
        background: "var(--bg)",
        borderRadius: 6,
      }}>
        <span style={{ fontSize: 13, color: "var(--muted)" }}>Market Mood</span>
        <span style={{
          fontSize: 18,
          fontWeight: 700,
          color: sentimentColor(avgSentiment),
        }}>
          {sentimentLabel(avgSentiment)}
        </span>
        <span style={{
          fontSize: 13,
          color: "var(--muted)",
          fontVariantNumeric: "tabular-nums",
        }}>
          ({avgSentiment > 0 ? "+" : ""}{(avgSentiment * 100).toFixed(0)}%)
        </span>
        <span style={{ fontSize: 12, color: "var(--muted)", marginLeft: "auto" }}>
          {news.length} headlines analyzed
        </span>
      </div>

      {/* News list */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {news.map((n, i) => {
          const assets = Object.entries(n.affected_assets)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 4);

          return (
            <div
              key={i}
              style={{
                padding: "8px 12px",
                background: "var(--bg)",
                borderRadius: 6,
                borderLeft: `3px solid ${sentimentColor(n.sentiment)}`,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 3 }}>
                    {n.title}
                  </div>
                  {n.summary && (
                    <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>
                      {n.summary}
                    </div>
                  )}
                </div>
                <div style={{ textAlign: "right", flexShrink: 0 }}>
                  <div style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: sentimentColor(n.sentiment),
                  }}>
                    {n.sentiment > 0 ? "+" : ""}{(n.sentiment * 100).toFixed(0)}%
                  </div>
                  <div style={{ fontSize: 10, color: "var(--muted)" }}>
                    {SOURCE_LABELS[n.source] ?? n.source}
                  </div>
                </div>
              </div>

              {/* Affected assets chips */}
              {assets.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                  {assets.map(([asset, impact]) => (
                    <span
                      key={asset}
                      style={{
                        fontSize: 10,
                        padding: "2px 6px",
                        borderRadius: 4,
                        background: impact > 0 ? "rgba(16,185,129,0.15)" : "rgba(239,68,68,0.15)",
                        color: impact > 0 ? "var(--reflation)" : "var(--deflation)",
                      }}
                    >
                      {formatAsset(asset)} {impact > 0 ? "+" : ""}{(impact * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
