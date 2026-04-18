import type { CurrentRegime, Dedollarization, NewsItem, RegimeExplain, Scoreboard } from "../types";

const BASE = "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  currentRegime: () => request<CurrentRegime>("/regime/current"),
  regimeExplain: () => request<RegimeExplain>("/regime/explain"),
  scoreboard: () => request<Scoreboard>("/scoreboard"),
  dedollarization: () => request<Dedollarization>("/dedollarization"),
  news: () => request<NewsItem[]>("/news"),
  refresh: () => request<{ status: string }>("/refresh", { method: "POST" }),
  generateDedollarExplanation: () =>
    request<{ explanation: string; date: string }>("/dedollarization/explanation", { method: "POST" }),
};
