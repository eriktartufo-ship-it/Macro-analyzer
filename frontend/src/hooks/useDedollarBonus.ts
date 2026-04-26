import { useEffect, useState } from "react";

const STORAGE_KEY = "macro_analyzer.dedollar_bonus_enabled";

/** Notifica tutti i listener nello stesso tab quando lo storage cambia. */
const EVENT_NAME = "dedollar-bonus-changed";

function readStorage(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

/** Hook condiviso per il flag dedollar bonus.
 *
 * Default: false (sistema data-driven puro). Persistenza: localStorage.
 * Cambia istantaneamente in tutti i componenti grazie all'evento custom.
 */
export function useDedollarBonus(): [boolean, (next: boolean) => void] {
  const [enabled, setEnabled] = useState<boolean>(() => readStorage());

  useEffect(() => {
    const handler = () => setEnabled(readStorage());
    window.addEventListener(EVENT_NAME, handler);
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener(EVENT_NAME, handler);
      window.removeEventListener("storage", handler);
    };
  }, []);

  const toggle = (next: boolean) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, next ? "1" : "0");
    } catch {
      /* swallow: persistenza non critica */
    }
    setEnabled(next);
    window.dispatchEvent(new Event(EVENT_NAME));
  };

  return [enabled, toggle];
}

/** Helper per le chiamate API — lette le preferenze al momento dell'invocazione. */
export function getDedollarBonusFlag(): boolean {
  return readStorage();
}
