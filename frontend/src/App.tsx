import { useEffect, useState, useCallback } from "react";
import { api } from "./api/client";
import type { CurrentRegime, Dedollarization, NewsItem, RegimeExplain, Scoreboard } from "./types";
import { Header, type Page, type Theme } from "./components/Header";
import { RegimeCard } from "./components/RegimeCard";
import { ProbabilityBars } from "./components/ProbabilityBars";
import { AssetRankingTable } from "./components/AssetRankingTable";
import { DedollarizationPage } from "./components/DedollarizationPage";
import { AnalysisPanel } from "./components/AnalysisPanel";
import { ProjectedAssetsPanel } from "./components/ProjectedAssetsPanel";
import { NewsPanel } from "./components/NewsPanel";

const THEME_KEY = "macro-theme";

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // localStorage non disponibile, fallback a media query
  }
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    return "dark";
  }
  return "light";
}

async function safe<T>(p: Promise<T>): Promise<T | null> {
  try {
    return await p;
  } catch {
    return null;
  }
}

export default function App() {
  const [regime, setRegime] = useState<CurrentRegime | null>(null);
  const [scoreboard, setScoreboard] = useState<Scoreboard | null>(null);
  const [dedollar, setDedollar] = useState<Dedollarization | null>(null);
  const [explain, setExplain] = useState<RegimeExplain | null>(null);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState<Page>("dashboard");
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // ignore
    }
  }, [theme]);

  const load = useCallback(async () => {
    setError(null);
    const [r, s, e, d, n] = await Promise.all([
      safe(api.currentRegime()),
      safe(api.scoreboard()),
      safe(api.regimeExplain()),
      safe(api.dedollarization()),
      safe(api.news()),
    ]);
    setRegime(r);
    setScoreboard(s);
    setExplain(e);
    setDedollar(d);
    setNews(n ?? []);
    if (!r && !s) {
      setError("Dati non ancora disponibili — attendi il primo refresh oppure premi Refresh data.");
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const onRefresh = async () => {
    setRefreshing(true);
    try {
      await api.refresh();
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Refresh failed");
    } finally {
      setRefreshing(false);
    }
  };

  const ready = regime && scoreboard;

  return (
    <div className="app">
      <Header
        date={regime?.date}
        onRefresh={onRefresh}
        refreshing={refreshing}
        page={page}
        onPageChange={setPage}
        theme={theme}
        onThemeToggle={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      />

      {loading && <div className="loading">Loading macro data…</div>}
      {error && !loading && <div className="error">{error}</div>}

      {ready && page === "dashboard" && (
        <>
          <div className="grid grid-2">
            <RegimeCard data={regime} />
            <ProbabilityBars
              probabilities={regime.probabilities}
              projected={explain?.trajectory?.projected_probabilities}
            />
          </div>
          {explain && <AnalysisPanel explain={explain} />}
          {explain?.trajectory && (
            <ProjectedAssetsPanel
              trajectory={explain.trajectory}
              currentScores={scoreboard.scores}
            />
          )}
        </>
      )}

      {ready && page === "sentiment" && <NewsPanel news={news} />}

      {ready && page === "dedollar" && dedollar && (
        <DedollarizationPage
          data={dedollar}
          rawIndicators={explain?.dedollar_indicators}
        />
      )}

      {ready && page === "assets" && (
        <AssetRankingTable
          scores={scoreboard.scores}
          projected={explain?.trajectory?.projected_scores}
        />
      )}
    </div>
  );
}
