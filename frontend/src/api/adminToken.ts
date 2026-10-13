// S51 (2026-10-13): token admin del sito. Serve per gli endpoint che CAMBIANO qualcosa
// (flag, impostazioni LLM, refresh, reset del portafoglio AI, rigenerazioni LLM).
// Vive solo nel browser di chi lo inserisce (localStorage); le letture restano pubbliche.

const KEY = "macro_admin_token";
const EVENT = "macro-admin-token";

export function getAdminToken(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function setAdminToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(KEY, token);
    else localStorage.removeItem(KEY);
  } catch {
    // localStorage non disponibile (navigazione privata): il token vale solo per la pagina
  }
  window.dispatchEvent(new Event(EVENT));
}

export function onAdminTokenChange(cb: () => void): () => void {
  window.addEventListener(EVENT, cb);
  return () => window.removeEventListener(EVENT, cb);
}
