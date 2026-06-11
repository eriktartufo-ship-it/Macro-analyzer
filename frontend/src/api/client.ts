import { getDedollarBonusFlag } from "../hooks/useDedollarBonus";
import type {
  BacktestResult,
  CalibrationPayload,
  ConfigFlags,
  CrisisHistory,
  CrisisRiskAssessment,
  CurrentRegime,
  InsiderActivity,
  DataSnapshot,
  DedollarComparison,
  DedollarHistoryItem,
  Dedollarization,
  EnsembleResult,
  FactorRegimeReport,
  FOMCReport,
  HMMPrediction,
  AiPortfolioDecision,
  AiPortfolioPerformanceResponse,
  AiPortfolioPosition,
  LeadTimeReport,
  LlmSettings,
  MacroLlmAnalysis,
  MonteCarloForecast,
  ScenarioPreset,
  ScenarioResult,
  SmoothableIndicator,
  SmoothedIndicator,
  TermPremiumReport,
  MacroIndicatorsHistoryItem,
  NewsItem,
  PlayerHistoryItem,
  PredictionLogStats,
  PredictionLogRecent,
  ActiveModelInfo,
  RegimeExplain,
  RegimeHistoryItem,
  Scoreboard,
  SignalsHistoryItem,
  TransitionMatrix,
  TransitionCheckResponse,
  MLForecastResponse,
  // Regime2DResponse + SubRegimeResponse rimossi 2026-05-28
} from "../types";

const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function request<T>(
  path: string,
  init?: RequestInit,
  opts: { retries?: number } = {},
): Promise<T> {
  const retries = opts.retries ?? 3;
  let lastErr: unknown;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(`${BASE}${path}`, init);
      if (res.ok) return (await res.json()) as T;

      // 4xx (tranne 408/429) = errore del client, non ritentare.
      if (res.status >= 400 && res.status < 500 && res.status !== 408 && res.status !== 429) {
        throw new ApiError(res.status, `${res.status} ${res.statusText}`);
      }

      lastErr = new ApiError(res.status, `${res.status} ${res.statusText}`);
    } catch (e) {
      if (e instanceof ApiError && e.status >= 400 && e.status < 500 && e.status !== 408 && e.status !== 429) {
        throw e;
      }
      lastErr = e;
    }

    if (attempt < retries) {
      await sleep(400 * Math.pow(2, attempt));
    }
  }

  throw lastErr instanceof Error ? lastErr : new Error("Request failed");
}

/** Aggiunge `include_dedollar=true|false` se l'utente ha esplicitamente toggleato.
 * Se preferenza non settata, omette il param (backend usa env var di default). */
function dedollarParam(): string {
  return getDedollarBonusFlag() ? "include_dedollar=true" : "include_dedollar=false";
}

function withDedollar(qs: string): string {
  const sep = qs.includes("?") ? "&" : "?";
  return `${qs}${sep}${dedollarParam()}`;
}

export const api = {
  currentRegime: () => request<CurrentRegime>("/regime/current"),
  regimeHistory: (days = 180) => request<RegimeHistoryItem[]>(`/regime/history?days=${days}`),
  regimeExplain: () => request<RegimeExplain>("/regime/explain"),
  scoreboard: () => request<Scoreboard>(withDedollar("/scoreboard")),
  aiPortfolioPositions: () => request<AiPortfolioPosition[]>("/ai-portfolio/positions"),
  aiPortfolioDecisions: (days = 30) =>
    request<AiPortfolioDecision[]>(`/ai-portfolio/decisions?days=${days}`),
  aiPortfolioPerformance: (days = 180) =>
    request<AiPortfolioPerformanceResponse>(`/ai-portfolio/performance?days=${days}`),
  aiPortfolioManualScan: () =>
    request<{ scan_summary: Record<string, unknown>; performance: unknown }>(
      "/ai-portfolio/manual-scan",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" },
      { retries: 0 },
    ),
  macroLlmAnalysis: (forceRefresh = false) =>
    request<MacroLlmAnalysis>(`/macro-llm-analysis${forceRefresh ? "?force_refresh=true" : ""}`),
  llmSettings: () => request<LlmSettings>("/llm-settings", undefined, { retries: 0 }),
  llmSettingsUpdate: (payload: { api_key?: string; model?: string }) =>
    request<LlmSettings>(
      "/llm-settings",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
      { retries: 0 },
    ),
  dedollarization: () => request<Dedollarization>("/dedollarization"),
  dedollarizationHistory: (days = 365) =>
    request<DedollarHistoryItem[]>(`/dedollarization/history?days=${days}`),
  signalsHistory: (days = 365) =>
    request<SignalsHistoryItem[]>(`/signals/history?days=${days}`),
  macroIndicatorsHistory: (days = 365) =>
    request<MacroIndicatorsHistoryItem[]>(`/macro-indicators/history?days=${days}`),
  dedollarPlayerHistory: (days = 365) =>
    request<PlayerHistoryItem[]>(`/dedollarization/player-history?days=${days}`),
  dataSnapshot: () => request<DataSnapshot>("/data-snapshot"),
  transitionMatrix: (horizonDays = 30, projectSteps = 0) =>
    request<TransitionMatrix>(
      `/regime/transition-matrix?horizon_days=${horizonDays}&project_steps=${projectSteps}`,
    ),
  hmmPrediction: (nStates = 4) =>
    request<HMMPrediction>(`/regime/hmm?n_states=${nStates}`, undefined, { retries: 0 }),
  regimeEnsemble: () => request<EnsembleResult>("/regime/ensemble", undefined, { retries: 0 }),
  crisisRisk: () => request<CrisisRiskAssessment>("/regime/crisis-risk", undefined, { retries: 0 }),
  // T9-AUDIT endpoints
  transitionCheck: () => request<TransitionCheckResponse>("/regime/transition-check", undefined, { retries: 0 }),
  mlForecast: (horizons = "1,3,6", topN = 5) =>
    request<MLForecastResponse>(
      `/regime/ml-forecast?horizons=${encodeURIComponent(horizons)}&top_n=${topN}`,
      undefined,
      { retries: 0 },
    ),
  // subRegime + regime2D rimossi 2026-05-28 (code audit: no frontend consumer)
  // CRITICAL #1+#2 council 2026-05-27: live track record + model snapshot lock
  predictionLogStats: (horizon: "1m" | "3m" | "6m" = "3m") =>
    request<PredictionLogStats>(
      `/prediction-log/stats?horizon=${horizon}`, undefined, { retries: 0 },
    ),
  predictionLogRecent: (limit = 30) =>
    request<PredictionLogRecent[]>(
      `/prediction-log/recent?limit=${limit}`, undefined, { retries: 0 },
    ),
  activeModel: () => request<ActiveModelInfo>("/model/active", undefined, { retries: 0 }),
  getConfigFlags: () => request<ConfigFlags>("/config/flags", undefined, { retries: 0 }),
  setConfigFlag: (name: string, value: boolean | null) =>
    request<{ ok: boolean; flags: ConfigFlags }>(
      "/config/flags",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, value }),
      },
      { retries: 0 },
    ),
  crisisRiskHistory: (days = 365 * 30, limit?: number) => {
    const q = new URLSearchParams();
    q.set("days", String(days));
    if (limit) q.set("limit", String(limit));
    return request<CrisisHistory>(`/regime/crisis-risk/history?${q.toString()}`, undefined, { retries: 0 });
  },
  insiderActivity: (params: { days?: number; maxFilingsPerDay?: number; endDate?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.days) q.set("days", String(params.days));
    if (params.maxFilingsPerDay) q.set("max_filings_per_day", String(params.maxFilingsPerDay));
    if (params.endDate) q.set("end_date", params.endDate);
    const qs = q.toString();
    return request<InsiderActivity>(`/alt-data/insider-activity${qs ? `?${qs}` : ""}`, undefined, { retries: 0 });
  },
  backtestRun: (params: { startYear?: number; endYear?: number; topN?: number; threshold?: number; costBps?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.startYear) q.set("start_year", String(params.startYear));
    if (params.endYear) q.set("end_year", String(params.endYear));
    if (params.topN) q.set("top_n", String(params.topN));
    if (params.threshold !== undefined) q.set("score_threshold", String(params.threshold));
    if (params.costBps !== undefined) q.set("cost_bps", String(params.costBps));
    q.set("include_dedollar", String(getDedollarBonusFlag()));
    return request<BacktestResult>(`/backtest/run?${q.toString()}`, undefined, { retries: 0 });
  },
  backtestLeadTime: (threshold = 0.35, lookbackMonths = 12) =>
    request<LeadTimeReport>(`/backtest/lead-time?threshold=${threshold}&lookback_months=${lookbackMonths}`, undefined, { retries: 0 }),
  monteCarloForecast: (
    params: {
      nPaths?: number;
      nSteps?: number;
      horizonDays?: number;
      initialSource?: "rule_based" | "ensemble";
    } = {},
  ) => {
    const q = new URLSearchParams();
    if (params.nPaths) q.set("n_paths", String(params.nPaths));
    if (params.nSteps) q.set("n_steps", String(params.nSteps));
    if (params.horizonDays) q.set("horizon_days", String(params.horizonDays));
    if (params.initialSource) q.set("initial_source", params.initialSource);
    q.set("include_dedollar", String(getDedollarBonusFlag()));
    return request<MonteCarloForecast>(`/regime/forecast/monte-carlo?${q.toString()}`, undefined, { retries: 0 });
  },
  scenariosList: () => request<ScenarioPreset[]>("/scenarios/list"),
  scenarioRun: (key: string) =>
    request<ScenarioResult>(
      withDedollar(`/scenarios/run?scenario_key=${encodeURIComponent(key)}`),
      undefined, { retries: 0 },
    ),
  dedollarComparison: () => request<DedollarComparison>("/scoreboard/dedollar-comparison"),
  factorRegimeMapping: (threshold = 0.40) =>
    request<FactorRegimeReport>(`/factors/regime-mapping?threshold=${threshold}`),
  termPremiumReport: (threshold = 0.40, days = 365 * 30) =>
    request<TermPremiumReport>(`/indicators/term-premium?threshold=${threshold}&days=${days}`),
  fomcReport: (limit = 6, forceRefresh = false) =>
    request<FOMCReport>(`/fomc/report?limit=${limit}&force_refresh=${forceRefresh}`, undefined, { retries: 0 }),
  smoothedIndicatorsList: () => request<SmoothableIndicator[]>("/indicators/smoothed/list"),
  smoothedIndicator: (seriesName: string, lambda = 10.0, days = 365 * 5) =>
    request<SmoothedIndicator>(
      `/indicators/smoothed?series_name=${encodeURIComponent(seriesName)}&lambda=${lambda}&days=${days}`,
    ),
  assetCalibration: () => request<CalibrationPayload>("/asset-calibration", undefined, { retries: 0 }),
  runAssetCalibration: () =>
    request<{ status: string; n_classifications: number }>(
      "/asset-calibration/run",
      { method: "POST" },
      { retries: 0 },
    ),
  news: () => request<NewsItem[]>("/news"),
  refresh: () => request<{ status: string }>("/refresh", { method: "POST" }, { retries: 0 }),
  generateDedollarExplanation: () =>
    request<{ explanation: string; date: string }>(
      "/dedollarization/explanation",
      { method: "POST" },
      { retries: 0 },
    ),
};
