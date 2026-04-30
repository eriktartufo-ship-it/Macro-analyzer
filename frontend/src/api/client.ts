import { getDedollarBonusFlag } from "../hooks/useDedollarBonus";
import type {
  BacktestResult,
  CalibrationPayload,
  CurrentRegime,
  DataSnapshot,
  DedollarComparison,
  DedollarHistoryItem,
  Dedollarization,
  EnsembleResult,
  FactorRegimeReport,
  FOMCReport,
  HMMPrediction,
  LeadTimeReport,
  MonteCarloForecast,
  ScenarioPreset,
  ScenarioResult,
  SmoothableIndicator,
  SmoothedIndicator,
  TermPremiumReport,
  MacroIndicatorsHistoryItem,
  NewsItem,
  PlayerHistoryItem,
  RegimeExplain,
  RegimeHistoryItem,
  Scoreboard,
  SignalsHistoryItem,
  TransitionMatrix,
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
  monteCarloForecast: (params: { nPaths?: number; nSteps?: number; horizonDays?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.nPaths) q.set("n_paths", String(params.nPaths));
    if (params.nSteps) q.set("n_steps", String(params.nSteps));
    if (params.horizonDays) q.set("horizon_days", String(params.horizonDays));
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
