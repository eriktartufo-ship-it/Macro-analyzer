import type {
  CurrentRegime,
  DedollarHistoryItem,
  Dedollarization,
  MacroIndicatorsHistoryItem,
  NewsItem,
  PlayerHistoryItem,
  RegimeExplain,
  RegimeHistoryItem,
  Scoreboard,
  SignalsHistoryItem,
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
  news: () => request<NewsItem[]>("/news"),
  refresh: () => request<{ status: string }>("/refresh", { method: "POST" }, { retries: 0 }),
  generateDedollarExplanation: () =>
    request<{ explanation: string; date: string }>(
      "/dedollarization/explanation",
      { method: "POST" },
      { retries: 0 },
    ),
};
