import { useEffect, useState, useCallback } from "react";
import { api } from "./api/client";
import type { CurrentRegime, Dedollarization, NewsItem, RegimeExplain, Scoreboard } from "./types";
import { Header, type Page } from "./components/Header";
import { RegimeCard } from "./components/RegimeCard";
import { ProbabilityBars } from "./components/ProbabilityBars";
import { AssetRankingTable } from "./components/AssetRankingTable";
import { DedollarizationPage } from "./components/DedollarizationPage";
import { AnalysisPanel } from "./components/AnalysisPanel";
import { ProjectedAssetsPanel } from "./components/ProjectedAssetsPanel";
import { NewsPanel } from "./components/NewsPanel";

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

  const load = useCallback(async () => {
    try {
      setError(null);
      const [r, s] = await Promise.all([api.currentRegime(), api.scoreboard()]);
      setRegime(r);
      setScoreboard(s);
      try { setExplain(await api.regimeExplain()); } catch { setExplain(null); }
      try { setDedollar(await api.dedollarization()); } catch { setDedollar(null); }
      try { setNews(await api.news()); } catch { setNews([]); }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load data");
    } finally {
      setLoading(false);
    }
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
      />

      {loading && <div className="loading">Loading macro data...</div>}
      {error && <div className="error">Error: {error}</div>}

      {ready && page === "dashboard" && (
        <>
          <div className="grid">
            <RegimeCard data={regime} />
            <ProbabilityBars probabilities={regime.probabilities} />
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

      {ready && page === "assets" && <AssetRankingTable scores={scoreboard.scores} />}
    </div>
  );
}
