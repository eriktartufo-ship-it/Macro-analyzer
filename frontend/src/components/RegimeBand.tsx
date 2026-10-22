/**
 * Banda del regime dominante giorno per giorno (S54). Sostituisce come lettura
 * principale le 4 linee di probabilità sovrapposte, da cui non si capiva quale
 * regime ci fosse: qui ogni giorno ha un colore solo.
 */
import type { Regime, RegimeHistoryItem } from "../types";
import { regimeLabel } from "../lib/assets";

const MONTHS = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"];

interface Run {
  regime: Regime;
  from: string;
  to: string;
  days: number;
}

function runs(history: RegimeHistoryItem[]): Run[] {
  const sorted = [...history].sort((a, b) => a.date.localeCompare(b.date));
  const out: Run[] = [];
  for (const h of sorted) {
    const last = out[out.length - 1];
    if (last && last.regime === h.regime) {
      last.to = h.date;
      last.days += 1;
    } else {
      out.push({ regime: h.regime, from: h.date, to: h.date, days: 1 });
    }
  }
  return out;
}

/** Da quando è in vigore il regime corrente (primo giorno della serie finale). */
export function regimeSince(history: RegimeHistoryItem[], regime: Regime): string | null {
  const r = runs(history);
  const last = r[r.length - 1];
  return last && last.regime === regime ? last.from : null;
}

function dayMs(iso: string): number {
  return Date.parse(`${iso}T00:00:00Z`);
}

/** Solo gli ultimi `days` giorni (l'App carica 5 anni di storia per il grafico completo). */
function lastDays(history: RegimeHistoryItem[], days: number): RegimeHistoryItem[] {
  if (history.length === 0) return history;
  const latest = history.reduce((m, h) => (h.date > m ? h.date : m), history[0].date);
  const cutoff = new Date(dayMs(latest) - days * 86400000).toISOString().slice(0, 10);
  return history.filter((h) => h.date > cutoff);
}

export function RegimeBand({
  history,
  showTicks = true,
  height = 14,
  days = 365,
}: {
  history: RegimeHistoryItem[];
  showTicks?: boolean;
  height?: number;
  days?: number;
}) {
  const r = runs(lastDays(history, days));
  if (r.length === 0) return null;
  const start = dayMs(r[0].from);
  const end = dayMs(r[r.length - 1].to);
  const span = Math.max(end - start, 1);

  // Tacche ai veri inizi mese, non spaziate a occhio
  const ticks: { label: string; left: number }[] = [];
  const d = new Date(start);
  d.setUTCDate(1);
  d.setUTCMonth(d.getUTCMonth() + 1);
  while (d.getTime() <= end) {
    ticks.push({ label: MONTHS[d.getUTCMonth()], left: ((d.getTime() - start) / span) * 100 });
    d.setUTCMonth(d.getUTCMonth() + 1);
  }

  const summary = r
    .filter((x) => x.days >= 7)
    .map((x) => `${regimeLabel(x.regime)} dal ${x.from} al ${x.to}`)
    .join("; ");

  return (
    <div className="rband">
      <div className="rband-bar" style={{ height }} role="img" aria-label={`Regime dominante giorno per giorno: ${summary}`}>
        {r.map((x) => (
          <span
            key={x.from}
            style={{ flexGrow: x.days, background: `var(--${x.regime})` }}
            title={`${regimeLabel(x.regime)} · ${x.from} → ${x.to}`}
          />
        ))}
      </div>
      {showTicks && (
        <div className="rband-ticks" aria-hidden="true">
          {ticks.map((t) => (
            <span key={`${t.label}-${t.left}`} style={{ left: `${t.left}%` }}>
              {t.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
