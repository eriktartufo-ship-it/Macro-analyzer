/**
 * Nomi italiani e famiglie dei 24 asset (S54, ristrutturazione del sito).
 * Prima ogni pannello aveva il suo formatAsset ("sector energy", "Sett. Energia",
 * "Sector Energy"): le pagine nuove leggono da qui, così lo stesso asset si
 * chiama allo stesso modo ovunque.
 */
import type { PositioningStance, Regime } from "../types";

export type AssetFamily =
  | "azioni_usa"
  | "fattori"
  | "settori"
  | "internazionali"
  | "obbligazioni"
  | "materie_prime"
  | "immobiliare"
  | "cripto";

export const FAMILY_LABEL: Record<AssetFamily, string> = {
  azioni_usa: "Azioni USA",
  fattori: "Fattori",
  settori: "Settori USA",
  internazionali: "Azioni internazionali",
  obbligazioni: "Obbligazioni e liquidità",
  materie_prime: "Materie prime",
  immobiliare: "Immobiliare",
  cripto: "Cripto",
};

const ASSETS: Record<string, { name: string; family: AssetFamily }> = {
  us_equities_growth: { name: "Azioni USA growth", family: "azioni_usa" },
  us_equities_value: { name: "Azioni USA value", family: "azioni_usa" },
  us_equities_sp500: { name: "S&P 500", family: "azioni_usa" },
  factor_momentum: { name: "Fattore momentum", family: "fattori" },
  factor_value: { name: "Fattore value", family: "fattori" },
  factor_quality: { name: "Fattore qualità", family: "fattori" },
  sector_energy: { name: "Settore energia", family: "settori" },
  sector_healthcare: { name: "Settore sanità", family: "settori" },
  sector_staples: { name: "Beni primari", family: "settori" },
  sector_financials: { name: "Settore finanziario", family: "settori" },
  sector_utilities: { name: "Utility", family: "settori" },
  sector_technology: { name: "Tecnologia", family: "settori" },
  sector_semiconductors: { name: "Semiconduttori", family: "settori" },
  international_dm_equities: { name: "Sviluppati ex-USA", family: "internazionali" },
  em_equities: { name: "Azioni emergenti", family: "internazionali" },
  us_bonds_long: { name: "Bond USA lunghi", family: "obbligazioni" },
  us_bonds_short: { name: "Bond USA brevi", family: "obbligazioni" },
  tips_inflation_bonds: { name: "TIPS (bond indicizzati)", family: "obbligazioni" },
  cash_money_market: { name: "Liquidità", family: "obbligazioni" },
  energy: { name: "Energia (materia prima)", family: "materie_prime" },
  gold: { name: "Oro", family: "materie_prime" },
  silver: { name: "Argento", family: "materie_prime" },
  broad_commodities: { name: "Materie prime (paniere)", family: "materie_prime" },
  real_estate_reits: { name: "Immobiliare (REIT)", family: "immobiliare" },
  bitcoin: { name: "Bitcoin", family: "cripto" },
  crypto_broad: { name: "Cripto (paniere)", family: "cripto" },
};

export function assetName(key: string): string {
  return ASSETS[key]?.name ?? key.replace(/_/g, " ");
}

export function assetFamily(key: string): string {
  const f = ASSETS[key]?.family;
  return f ? FAMILY_LABEL[f] : "";
}

export function isKnownAsset(key: string): boolean {
  return key in ASSETS;
}

/**
 * Soglie di lettura dello score (0-100, 50 = neutro). Le usano sia il colore del
 * numero sia il confronto col modello Crisi: un solo posto, una sola regola.
 */
export const SCORE_FAVORED = 55;
export const SCORE_UNFAVORED = 45;

export type ScoreTone = "up" | "down" | "flat";

export function scoreTone(score: number): ScoreTone {
  if (score >= SCORE_FAVORED) return "up";
  if (score < SCORE_UNFAVORED) return "down";
  return "flat";
}

/**
 * Il modello Crisi dice il contrario dello score del regime: sovrappesare un asset
 * che lo score mette fra gli sfavoriti, o sottopesare uno fra i favoriti.
 * Prima i due pareri stavano in due pannelli diversi senza spiegazione.
 */
export function crisisConflict(score: number | undefined, stance: PositioningStance | undefined): boolean {
  if (score == null || !stance) return false;
  if (stance === "overweight") return score < SCORE_UNFAVORED;
  if (stance === "underweight") return score >= SCORE_FAVORED;
  return false;
}

export const REGIME_LABEL: Record<Regime, string> = {
  reflation: "Reflazione",
  stagflation: "Stagflazione",
  deflation: "Deflazione",
  goldilocks: "Goldilocks",
};

export const REGIME_DESCRIPTION: Record<Regime, string> = {
  reflation: "Crescita forte e inflazione in salita.",
  stagflation: "Crescita debole e inflazione alta.",
  deflation: "Crescita in frenata e inflazione in calo.",
  goldilocks: "Crescita solida e inflazione bassa.",
};

export function regimeLabel(r: string | null | undefined): string {
  if (!r) return "—";
  return REGIME_LABEL[r as Regime] ?? r;
}

/** Numeri all'italiana: virgola decimale, punto delle migliaia. */
export function fmt(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toLocaleString("it-IT", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtSigned(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const s = fmt(Math.abs(v), digits);
  if (v > 0) return `+${s}`;
  if (v < 0) return `−${s}`;
  return s;
}

/** Frazione (0.437) → "43,7%". */
export function fmtPct(v: number | null | undefined, digits = 0): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${fmt(v * 100, digits)}%`;
}

/** Frazione con segno (0.0782) → "+7,82%". */
export function fmtPctSigned(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${fmtSigned(v * 100, digits)}%`;
}
