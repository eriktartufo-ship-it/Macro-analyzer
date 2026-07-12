/** Client API Portfolio Tracking (/api/v1/pt) — self-contained, token in localStorage. */

const BASE = "/api/v1/pt";
const TOKEN_KEY = "pt-token";

export interface PtSession {
  token: string;
  username: string;
  display_name: string;
}

export interface PtPosition {
  asset_key: string;
  label: string;
  asset_class: "metal" | "crypto" | "ticker";
  ticker: string;
  unit: string;
  quantity: number;
  avg_cost_eur: number;
  invested_eur: number;
  realized_pnl_eur: number;
  spot_eur: number | null;
  market_value_eur: number | null;
  pnl_eur: number | null;
  pnl_pct: number | null;
  allocation_pct: number | null;
}

export interface PtPortfolio {
  username: string;
  display_name: string;
  positions: PtPosition[];
  totals: {
    invested_eur: number;
    market_value_eur: number | null;
    realized_pnl_eur: number;
    pnl_eur: number | null;
    pnl_pct: number | null;
  };
}

export interface PtTransaction {
  id: number;
  asset_key: string;
  asset_class: string;
  label: string;
  ticker: string;
  side: "buy" | "sell";
  quantity: number;
  unit: string;
  unit_price_eur: number;
  fee_eur: number;
  trade_date: string;
  note: string | null;
}

export interface PtTransactionIn {
  asset_key: string;
  side: "buy" | "sell";
  quantity: number;
  unit: string;
  unit_price_eur: number;
  fee_eur: number;
  trade_date: string;
  note?: string | null;
  label?: string | null;
}

export interface PtAdviceAsset {
  light: "green" | "yellow" | "red";
  headline: string;
  reasons: string[];
  what_to_watch: string;
}

export interface PtAdvice {
  id: number;
  generated_at: string;
  trigger: string;
  provider: string;
  macro_summary: string;
  regime?: string;
  regime_confidence?: number;
  assets: Partial<Record<"gold" | "silver" | "bitcoin", PtAdviceAsset>>;
}

export interface PtStrategyTarget {
  label: string;
  target_pct: number;
  asset_keys?: string[];
}

export interface PtStrategy {
  title?: string;
  comment?: string;
  targets?: PtStrategyTarget[];
  next_moves?: { label: string; amount_eur?: number; note?: string }[];
  check_note?: string;
}

export class PtApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    // localStorage non disponibile
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // corpo non-JSON
    }
    if (res.status === 401) setToken(null);
    throw new PtApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const ptApi = {
  login: (username: string, password: string) =>
    request<PtSession>("/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  me: () => request<{ username: string; display_name: string }>("/me"),
  refreshToken: () => request<{ token: string }>("/refresh-token", { method: "POST" }),
  portfolio: () => request<PtPortfolio>("/portfolio"),
  transactions: () => request<{ transactions: PtTransaction[] }>("/transactions"),
  addTransaction: (tx: PtTransactionIn) =>
    request<PtTransaction>("/transactions", { method: "POST", body: JSON.stringify(tx) }),
  deleteTransaction: (id: number) =>
    request<{ deleted: number }>(`/transactions/${id}`, { method: "DELETE" }),
  latestAdvice: () => request<{ advice: PtAdvice | null }>("/advice/latest"),
  refreshAdvice: () => request<{ advice: PtAdvice | null }>("/advice/refresh", { method: "POST" }),
  strategy: () =>
    request<{ strategy: PtStrategy | null; updated_at?: string }>("/strategy"),
};
