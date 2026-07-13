import { useEffect, useState, useCallback, lazy, Suspense } from "react";
import { api } from "./api/client";
import { useDedollarBonus } from "./hooks/useDedollarBonus";
import type { CrisisHistory, CrisisRiskAssessment, CurrentRegime, Dedollarization, InsiderActivity, NewsItem, RegimeExplain, RegimeHistoryItem, Scoreboard } from "./types";
import { Header, type Page, type Theme } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { ptEnsureLoaded } from "./hooks/usePtSession";
import { RegimeCard } from "./components/RegimeCard";
import { ProbabilityBars } from "./components/ProbabilityBars";
import { RegimeTimelineChart } from "./components/RegimeTimelineChart";
import { CrisisRiskPanel } from "./components/CrisisRiskPanel";
import { LiveTrackRecordPanel } from "./components/LiveTrackRecordPanel";
import { MacroLlmAnalysisPanel } from "./components/MacroLlmAnalysisPanel";
import { ProjectedAssetsPanel } from "./components/ProjectedAssetsPanel";
import { LazyDetails } from "./components/LazyDetails";

// Lazy-loaded — heavy / off-dashboard / collapsible
const DedollarizationPage = lazy(() => import("./components/DedollarizationPage").then(m => ({ default: m.DedollarizationPage })));
const AssetRankingTable = lazy(() => import("./components/AssetRankingTable").then(m => ({ default: m.AssetRankingTable })));
const DataPage = lazy(() => import("./components/DataPage").then(m => ({ default: m.DataPage })));
const AiPortfolioPage = lazy(() => import("./components/AiPortfolioPage").then(m => ({ default: m.AiPortfolioPage })));
const NewsPanel = lazy(() => import("./components/NewsPanel").then(m => ({ default: m.NewsPanel })));
const AnalysisPanel = lazy(() => import("./components/AnalysisPanel").then(m => ({ default: m.AnalysisPanel })));
const CrisisHistoryTimeline = lazy(() => import("./components/CrisisHistoryTimeline").then(m => ({ default: m.CrisisHistoryTimeline })));
const InsiderActivityTile = lazy(() => import("./components/InsiderActivityTile").then(m => ({ default: m.InsiderActivityTile })));
const Tier9TransitionPanel = lazy(() => import("./components/Tier9TransitionPanel").then(m => ({ default: m.Tier9TransitionPanel })));
const Tier9ForecastPanel = lazy(() => import("./components/Tier9ForecastPanel").then(m => ({ default: m.Tier9ForecastPanel })));
const FailureLearningsPanel = lazy(() => import("./components/FailureLearningsPanel").then(m => ({ default: m.FailureLearningsPanel })));
const PortfolioPage = lazy(() => import("./components/PortfolioPage").then(m => ({ default: m.PortfolioPage })));

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
  const [regimeHistory, setRegimeHistory] = useState<RegimeHistoryItem[]>([]);
  const [scoreboard, setScoreboard] = useState<Scoreboard | null>(null);
  const [dedollar, setDedollar] = useState<Dedollarization | null>(null);
  const [explain, setExplain] = useState<RegimeExplain | null>(null);
  const [crisis, setCrisis] = useState<CrisisRiskAssessment | null>(null);
  const [crisisHistory, setCrisisHistory] = useState<CrisisHistory | null>(null);
  const [insider, setInsider] = useState<InsiderActivity | null>(null);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [dedollarFlag] = useDedollarBonus();
  const [loading, setLoading] = useState(true);
  // Deep-link: macro.pieranlab.cloud/portfolio apre direttamente i Portafogli
  const [page, setPage] = useState<Page>(() =>
    typeof window !== "undefined" && window.location.pathname.startsWith("/portfolio")
      ? "portfolio"
      : "dashboard",
  );
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // ignore
    }
  }, [theme]);

  // Risolve una eventuale sessione Portafogli salvata (token) → visibile nella sidebar
  useEffect(() => {
    ptEnsureLoaded();
  }, []);

  const load = useCallback(async () => {
    setError(null);

    // PHASE 1 — critical above-the-fold (regime + scores + explain + crisis)
    const [r, s, e, c] = await Promise.all([
      safe(api.currentRegime()),
      safe(api.scoreboard()),
      safe(api.regimeExplain()),
      safe(api.crisisRisk()),
    ]);
    setRegime(r);
    setScoreboard(s);
    setExplain(e);
    setCrisis(c);
    if (!r && !s) {
      setError("Dati non ancora disponibili — attendi il primo refresh oppure premi Refresh data.");
    }
    setLoading(false);

    // PHASE 2 — secondary (timeline + dedollar + news), no blocking UI
    Promise.all([
      safe(api.regimeHistory(365 * 5)),
      safe(api.dedollarization()),
      safe(api.news()),
    ]).then(([h, d, n]) => {
      setRegimeHistory(h ?? []);
      setDedollar(d);
      setNews(n ?? []);
    });
  }, []);

  // Lazy fetch — only when the collapsible details opens
  const loadInsider = useCallback(async () => {
    if (insider) return;
    const i = await safe(api.insiderActivity({ days: 7, maxFilingsPerDay: 30 }));
    setInsider(i);
  }, [insider]);

  const loadCrisisHistory = useCallback(async () => {
    if (crisisHistory) return;
    // 10 anni invece di 30: la timeline storica resta significativa, payload molto minore
    const ch = await safe(api.crisisRiskHistory(365 * 10));
    setCrisisHistory(ch);
  }, [crisisHistory]);

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load, dedollarFlag]);

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
  const fallback = <div className="loading">Caricamento…</div>;

  const themeToggle = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  return (
    <div className="app mac-shell">
      <Sidebar
        date={regime?.date}
        page={page}
        onPageChange={setPage}
        theme={theme}
        onThemeToggle={themeToggle}
        onRefresh={onRefresh}
        refreshing={refreshing}
      />
      <div className="mac-main">
        <Header
          date={regime?.date}
          onRefresh={onRefresh}
          refreshing={refreshing}
          page={page}
          onPageChange={setPage}
          theme={theme}
          onThemeToggle={themeToggle}
        />

      {loading && page !== "portfolio" && <div className="loading">Loading macro data…</div>}
      {error && !loading && page !== "portfolio" && <div className="error">{error}</div>}

      {ready && page === "dashboard" && (
        <>
          {/* HERO: regime corrente + probability — above the fold */}
          <div className="grid grid-2">
            <RegimeCard data={regime} trajectory={explain?.trajectory} />
            <ProbabilityBars
              probabilities={regime.probabilities}
              projected={explain?.trajectory?.projected_probabilities}
            />
          </div>

          {/* AI ANALYSIS — Gemini macro analysis (cached 24h, fisso) */}
          <MacroLlmAnalysisPanel />

          {/* PROOF — Live track record (validation reale) */}
          <LiveTrackRecordPanel />

          {/* TIMELINE — regime history visual (lazy: phase 2 data) */}
          {regimeHistory.length > 1 && (
            <RegimeTimelineChart
              history={regimeHistory}
              projected={
                explain?.trajectory?.projected_fit_scores ??
                explain?.trajectory?.projected_probabilities ??
                null
              }
            />
          )}

          {/* CRISIS RISK + Asset projection — key actionable info */}
          {crisis && <CrisisRiskPanel data={crisis} />}
          {explain?.trajectory && (
            <ProjectedAssetsPanel
              trajectory={explain.trajectory}
              currentScores={scoreboard.scores}
            />
          )}

          {/* DETAILS — lazy mount + lazy fetch (riduce TTI + payload) */}
          <LazyDetails summary="📊 Analytics & insights aggiuntive">
            <Suspense fallback={fallback}>
              <Tier9TransitionPanel />
              <Tier9ForecastPanel />
              <FailureLearningsPanel />
              {explain && <AnalysisPanel explain={explain} />}
              <CrisisHistoryOnOpen
                onMount={loadCrisisHistory}
                data={crisisHistory}
              />
              <InsiderOnOpen onMount={loadInsider} data={insider} />
            </Suspense>
          </LazyDetails>
        </>
      )}

      {ready && page === "sentiment" && (
        <Suspense fallback={fallback}>
          <NewsPanel news={news} />
        </Suspense>
      )}

      {ready && page === "dedollar" && dedollar && (
        <Suspense fallback={fallback}>
          <DedollarizationPage
            data={dedollar}
            rawIndicators={explain?.dedollar_indicators}
          />
        </Suspense>
      )}

      {ready && page === "assets" && (
        <Suspense fallback={fallback}>
          <AssetRankingTable scores={scoreboard.scores} />
        </Suspense>
      )}

      {ready && page === "data" && (
        <Suspense fallback={fallback}>
          <DataPage />
        </Suspense>
      )}

      {ready && page === "ai-portfolio" && (
        <Suspense fallback={fallback}>
          <AiPortfolioPage />
        </Suspense>
      )}

      {/* Portafogli personali: indipendente da `ready` (non serve il modello macro) */}
      {page === "portfolio" && (
        <Suspense fallback={fallback}>
          <PortfolioPage />
        </Suspense>
      )}
      </div>
    </div>
  );
}

// Wrapper interni che triggera fetch al primo mount (= primo open del details)
function CrisisHistoryOnOpen({ onMount, data }: { onMount: () => void; data: CrisisHistory | null }) {
  useEffect(() => {
    onMount();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (!data || data.n_points <= 1) return null;
  return <CrisisHistoryTimeline data={data} />;
}

function InsiderOnOpen({ onMount, data }: { onMount: () => void; data: InsiderActivity | null }) {
  useEffect(() => {
    onMount();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (!data) return null;
  return <InsiderActivityTile data={data} />;
}
