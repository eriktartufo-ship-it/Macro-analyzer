/**
 * Navigazione per DOMANDA (S54, bozza A approvata da Erik il 2026-10-22):
 * 9 voci allo stesso livello → 5 sezioni, ognuna con le sue schede.
 * Ogni pagina ha un indirizzo (#/sezione/scheda) così funzionano il tasto
 * indietro e i link diretti; la scheda di un asset è #/mercati/asset/<chiave>.
 */
import type { IconName } from "../components/Icon";

export type Section = "oggi" | "mercati" | "scenari" | "portafogli" | "metodo";

export interface TabDef {
  id: string;
  label: string;
}

export interface SectionDef {
  id: Section;
  label: string;
  sub: string;
  icon: IconName;
  tabs: TabDef[];
}

export const SECTIONS: SectionDef[] = [
  { id: "oggi", label: "Oggi", sub: "Dove siamo e cosa fare", icon: "home", tabs: [] },
  {
    id: "mercati",
    label: "Mercati",
    sub: "Classifica, rotazione, notizie",
    icon: "chart",
    tabs: [
      { id: "classifica", label: "Classifica" },
      { id: "rotazione", label: "Rotazione" },
      { id: "notizie", label: "Notizie" },
    ],
  },
  {
    id: "scenari",
    label: "Scenari",
    sub: "Regime, ciclo, dollaro, crisi",
    icon: "compass",
    tabs: [
      { id: "regime", label: "Regime" },
      { id: "ciclo", label: "Ciclo" },
      { id: "dollaro", label: "Dollaro" },
      { id: "crisi", label: "Crisi" },
      { id: "europa", label: "Europa" },
    ],
  },
  {
    id: "portafogli",
    label: "Portafogli",
    sub: "Bot, libro reale, i miei",
    icon: "wallet",
    tabs: [
      { id: "bot", label: "Bot" },
      { id: "libro", label: "Libro reale" },
      { id: "miei", label: "I miei" },
    ],
  },
  {
    id: "metodo",
    label: "Metodo",
    sub: "Track record, modelli, dati",
    icon: "flask",
    tabs: [
      { id: "track", label: "Track record" },
      { id: "ragiona", label: "Come ragiona" },
      { id: "modelli", label: "Modelli" },
      { id: "dati", label: "Dati grezzi" },
    ],
  },
];

export interface Route {
  section: Section;
  tab: string | null;
  asset: string | null;
}

export const HOME: Route = { section: "oggi", tab: null, asset: null };

function sectionDef(id: string): SectionDef | undefined {
  return SECTIONS.find((s) => s.id === id);
}

export function defaultTab(section: Section): string | null {
  return sectionDef(section)?.tabs[0]?.id ?? null;
}

export function to(section: Section, tab?: string): Route {
  return { section, tab: tab ?? defaultTab(section), asset: null };
}

export function toAsset(asset: string): Route {
  return { section: "mercati", tab: "classifica", asset };
}

export function parseLocation(hash: string, pathname: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  // Deep-link storico: macro.pieranlab.cloud/portfolio apre i Portafogli personali
  if (parts.length === 0 && pathname.startsWith("/portfolio")) return to("portafogli", "miei");
  const def = sectionDef(parts[0] ?? "");
  if (!def) return HOME;
  if (def.id === "mercati" && parts[1] === "asset" && parts[2]) {
    return toAsset(decodeURIComponent(parts[2]));
  }
  const tab = def.tabs.find((t) => t.id === parts[1])?.id ?? defaultTab(def.id);
  return { section: def.id, tab, asset: null };
}

export function formatRoute(r: Route): string {
  if (r.asset) return `#/mercati/asset/${encodeURIComponent(r.asset)}`;
  if (r.section === "oggi") return "#/oggi";
  return `#/${r.section}${r.tab ? `/${r.tab}` : ""}`;
}
