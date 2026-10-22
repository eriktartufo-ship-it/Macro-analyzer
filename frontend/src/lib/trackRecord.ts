/**
 * Track record per versione sigillata del modello (S55).
 *
 * Con un nuovo sigillo il track record riparte: la versione in vigore mostra solo le sue
 * previsioni, la precedente resta visibile come storico. Prima di un secondo sigillo
 * `previous` è null e il comportamento è quello di sempre.
 */
import { api } from "../api/client";
import type { ActiveModelInfo, ModelSnapshotInfo, PredictionLogStats } from "../types";

export type Horizon = "1m" | "3m" | "6m";

export interface TrackRecord {
  model: ActiveModelInfo;
  active: PredictionLogStats;
  activeSnapshot: ModelSnapshotInfo | null;
  previous: PredictionLogStats | null;
  previousSnapshot: ModelSnapshotInfo | null;
}

export async function loadTrackRecord(horizon: Horizon): Promise<TrackRecord> {
  const model = await api.activeModel();
  const snaps = [...model.snapshots].sort((a, b) => (b.locked_at ?? "").localeCompare(a.locked_at ?? ""));
  const activeSnapshot = snaps.find((s) => s.version === model.active_version) ?? null;
  const previousSnapshot = snaps.find((s) => s.version !== model.active_version) ?? null;
  const [active, previous] = await Promise.all([
    api.predictionLogStats(horizon, model.active_version),
    previousSnapshot ? api.predictionLogStats(horizon, previousSnapshot.version) : Promise.resolve(null),
  ]);
  return { model, active, activeSnapshot, previous, previousSnapshot };
}

/** "2026-10-22T09:12:00" → "22/10/2026" */
export function sealDate(s: ModelSnapshotInfo | null): string {
  if (!s?.locked_at) return "—";
  const [y, m, d] = s.locked_at.slice(0, 10).split("-");
  return `${d}/${m}/${y}`;
}
