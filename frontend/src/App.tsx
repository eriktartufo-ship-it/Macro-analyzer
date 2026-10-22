import { useEffect, useState, useCallback, lazy, Suspense } from "react";
import { getAdminToken, onAdminTokenChange } from "./api/adminToken";
import { api } from "./api/client";
import { useDedollarBonus } from "./hooks/useDedollarBonus";
import { useRoute } from "./hooks/useRoute";
import type { CrisisHistory, CrisisRiskAssessment, CurrentRegime, Dedollarization, InsiderActivity, NewsItem, RegimeExplain, RegimeHistoryItem, Scoreboard } from "./types";
import { Header, type Theme } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { SectionTabs } from "./components/SectionTabs";
import { ptEnsureLoaded } from "./hooks/usePtSession";
import { OggiPage } from "./components/OggiPage";
import { LazyDetails } from "./components/LazyDetails";

// Lazy-loaded: tutto ciò che non è la home
const ClassificaPanel = lazy(() => import("./components/ClassificaPanel").then(m => ({ default: m.ClassificaPanel })));
const AssetDetailPage = lazy(() => import("./components/AssetDetailPage").then(m => ({ default: m.AssetDetailPage })));
const FlowsPage = lazy(() => import("./components/FlowsPage").then(m => ({ default: m.FlowsPage })));
const NewsPanel = lazy(() => import("./components/NewsPanel").then(m => ({ default: m.NewsPanel })));
const InsiderActivityTile = lazy(() => import("./components/InsiderActivityTile").then(m => ({ default: m.InsiderActivityTile })));
const MacroLlmAnalysisPanel = lazy(() => import("./components/MacroLlmAnalysisPanel").then(m => ({ default: m.MacroLlmAnalysisPanel })));
const RegimeTimelineChart = lazy(() => import("./components/RegimeTimelineChart").then(m => ({ default: m.RegimeTimelineChart })));
const Tier9TransitionPanel = lazy(() => import("./components/Tier9TransitionPanel").then(m => ({ default: m.Tier9TransitionPanel })));
const Tier9ForecastPanel = lazy(() => import("./components/Tier9ForecastPanel").then(m => ({ default: m.Tier9ForecastPanel })));
const AnalysisPanel = lazy(() => import("./components/AnalysisPanel").then(m => ({ default: m.AnalysisPanel })));
const CyclePage = lazy(() => import("./components/CyclePage").then(m => ({ default: m.CyclePage })));
const DedollarizationPage = lazy(() => import("./components/DedollarizationPage").then(m => ({ default: m.DedollarizationPage })));
const CrisisRiskPanel = lazy(() => import("./components/CrisisRiskPanel").then(m => ({ default: m.CrisisRiskPanel })));
const CrisisHistoryTimeline = lazy(() => import("./components/CrisisHistoryTimeline").then(m => ({ default: m.CrisisHistoryTimeline })));
const EuropeEnergyPanel = lazy(() => import("./components/EuropeEnergyPanel").then(m => ({ default: m.EuropeEnergyPanel })));
const AiPortfolioPage = lazy(() => import("./components/AiPortfolioPage").then(m => ({ default: m.AiPortfolioPage })));
const PortfolioPage = lazy(() => import("./components/PortfolioPage").then(m => ({ default: m.PortfolioPage })));
const LiveTrackRecordPanel = lazy(() => import("./components/LiveTrackRecordPanel").then(m => ({ default: m.LiveTrackRecordPanel })));
const DataQualityStrip = lazy(() => import("./components/DataQualityStrip").then(m => ({ default: m.DataQualityStrip })));
const FailureLearningsPanel = lazy(() => import("./components/FailureLearningsPanel").then(m => ({ default: m.FailureLearningsPanel })));
const DataPage = lazy(() => import("./components/DataPage").then(m => ({ default: m.DataPage })));

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
  // S51: il refresh vuole il token admin; senza, il bottone non compare (fallirebbe a ogni clic)
  const [isAdmin, setIsAdmin] = useState(() => Boolean(getAdminToken()));
  useEffect(() => onAdminTokenChange(() => setIsAdmin(Boolean(getAdminToken()))), []);
  const [dedollarFlag] = useDedollarBonus();
  const [loading, setLoading] = useState(true);
  const [route, navigate] = useRoute();
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

    // FASE 1 — ciò che serve alla home (regime + score + explain + crisi)
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
      setError("Dati non ancora disponibili — attendi l'aggiornamento giornaliero (o premi Aggiorna, se hai l'accesso admin).");
    }
    setLoading(false);

    // FASE 2 — storia + dollaro + notizie, senza bloccare la pagina
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

  // Caricati solo quando si apre la scheda che li mostra
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
      setError(e instanceof Error ? e.message : "Aggiornamento non riuscito");
    } finally {
      setRefreshing(false);
    }
  };

  const ready = regime && scoreboard;
  const fallback = <div className="loading">Caricamento…</div>;
  const themeToggle = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  // I Portafogli personali non dipendono dal modello macro
  const isMyPortfolio = route.section === "portafogli" && route.tab === "miei";

  const page = (() => {
    if (isMyPortfolio) return <PortfolioPage />;
    if (!ready) return null;

    switch (route.section) {
      case "oggi":
        return (
          <OggiPage
            regime={regime}
            explain={explain}
            scoreboard={scoreboard}
            crisis={crisis}
            history={regimeHistory}
            navigate={navigate}
          />
        );

      case "mercati":
        if (route.asset) {
          return (
            <AssetDetailPage
              asset={route.asset}
              scoreboard={scoreboard}
              explain={explain}
              crisis={crisis}
              news={news}
              history={regimeHistory}
              navigate={navigate}
            />
          );
        }
        if (route.tab === "rotazione") return <FlowsPage />;
        if (route.tab === "notizie") {
          return (
            <>
              <NewsPanel news={news} />
              <LazyDetails summary="Acquisti e vendite degli insider (SEC Form 4)">
                <InsiderOnOpen onMount={loadInsider} data={insider} />
              </LazyDetails>
            </>
          );
        }
        return <ClassificaPanel scoreboard={scoreboard} explain={explain} crisis={crisis} navigate={navigate} />;

      case "scenari":
        if (route.tab === "ciclo") return <CyclePage />;
        if (route.tab === "dollaro") {
          return dedollar ? <DedollarizationPage data={dedollar} rawIndicators={explain?.dedollar_indicators} /> : fallback;
        }
        if (route.tab === "crisi") {
          return (
            <>
              {crisis && <CrisisRiskPanel data={crisis} />}
              <CrisisHistoryOnOpen onMount={loadCrisisHistory} data={crisisHistory} />
            </>
          );
        }
        if (route.tab === "europa") return <EuropeEnergyPanel />;
        return (
          <>
            <MacroLlmAnalysisPanel />
            {regimeHistory.length > 1 && (
              <RegimeTimelineChart
                history={regimeHistory}
                projected={explain?.trajectory?.projected_fit_scores ?? explain?.trajectory?.projected_probabilities ?? null}
              />
            )}
            <Tier9TransitionPanel />
            <Tier9ForecastPanel />
            {explain && <AnalysisPanel explain={explain} />}
          </>
        );

      case "portafogli":
        return <AiPortfolioPage view={route.tab === "libro" ? "libro" : "bot"} />;

      case "metodo":
        if (route.tab === "ragiona") return <DataPage section="ragiona" />;
        if (route.tab === "modelli") return <DataPage section="modelli" />;
        if (route.tab === "dati") return <DataPage section="dati" />;
        return (
          <>
            <LiveTrackRecordPanel />
            <DataQualityStrip />
            <FailureLearningsPanel />
          </>
        );
    }
  })();

  return (
    <div className="app mac-shell">
      <Sidebar
        date={regime?.date}
        route={route}
        navigate={navigate}
        theme={theme}
        onThemeToggle={themeToggle}
        onRefresh={isAdmin ? onRefresh : undefined}
        refreshing={refreshing}
      />
      <div className="mac-main">
        <Header
          date={regime?.date}
          onRefresh={isAdmin ? onRefresh : undefined}
          refreshing={refreshing}
          route={route}
          navigate={navigate}
          theme={theme}
          onThemeToggle={themeToggle}
        />

        {route.section !== "oggi" && !route.asset && (
          <SectionTabs section={route.section} active={route.tab} navigate={navigate} />
        )}

        {loading && !isMyPortfolio && <div className="loading">Caricamento dei dati…</div>}
        {error && !loading && !isMyPortfolio && <div className="error">{error}</div>}

        <Suspense fallback={fallback}>{page}</Suspense>
      </div>
    </div>
  );
}

// Wrapper interni che avviano il caricamento al primo montaggio
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
  if (!data) return <div className="loading">Caricamento…</div>;
  return <InsiderActivityTile data={data} />;
}
