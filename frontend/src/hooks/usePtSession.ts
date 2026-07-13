/** Sessione Portafogli condivisa a livello app.
 *
 * Il modulo pt (login per-utente) è l'unica "area riservata" di Macro. Questo
 * store rende l'utente loggato visibile ovunque (sidebar/impostazioni), non
 * solo dentro PortfolioPage. Fonte di verità = token in localStorage;
 * useSyncExternalStore tiene tutti i componenti allineati (login/logout
 * riflessi ovunque, come RGV/CognitiveOS).
 */
import { useSyncExternalStore } from "react";
import { getToken, ptApi, setToken } from "../api/pt";

export interface PtSessionUser {
  username: string;
  display_name: string;
}

let user: PtSessionUser | null = null;
let loaded = false; // /me già tentato (evita loop di fetch)
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

function getSnapshot(): PtSessionUser | null {
  return user;
}

/** Chiamato dopo un login riuscito (PortfolioPage). */
export function ptSetSession(u: PtSessionUser, token: string) {
  setToken(token);
  user = u;
  loaded = true;
  emit();
}

/** Logout globale: cancella token + stato (sidebar o PortfolioPage). */
export function ptClearSession() {
  setToken(null);
  user = null;
  loaded = true;
  emit();
}

/** All'avvio: se c'è un token, verifica /me una volta e popola l'utente. */
export async function ptEnsureLoaded() {
  if (loaded) return;
  loaded = true;
  if (!getToken()) return;
  try {
    const me = await ptApi.me();
    user = me;
    emit();
    // sessione rolling: rinnova il token
    ptApi.refreshToken().then((r) => setToken(r.token)).catch(() => undefined);
  } catch {
    setToken(null);
    user = null;
    emit();
  }
}

export function usePtSession(): PtSessionUser | null {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
