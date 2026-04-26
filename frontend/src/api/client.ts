import type {
  BacktestResult,
  CalibrationPayload,
  CurrentRegime,
  DataSnapshot,
  DedollarHistoryItem,
  Dedollarization,
  EnsembleResult,
  HMMPrediction,
  LeadTimeReport,
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

export const api = {
  currentRegime: () => request<CurrentRegime>("/regime/current"),
  regimeHistory: (days = 180) => request<RegimeHistoryItem[]>(`/regime/history?days=${days}`),
  regimeExplain: () => request<RegimeExplain>("/regime/explain"),
  scoreboard: () => request<Scoreboard>("/scoreboard"),
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
    return request<BacktestResult>(`/backtest/run${q.toString() ? "?" + q.toString() : ""}`, undefined, { retries: 0 });
  },
  backtestLeadTime: (threshold = 0.35, lookbackMonths = 12) =>
    request<LeadTimeReport>(`/backtest/lead-time?threshold=${threshold}&lookback_months=${lookbackMonths}`, undefined, { retries: 0 }),
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
