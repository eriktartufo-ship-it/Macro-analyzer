import { useCallback, useEffect, useState } from "react";
import { formatRoute, parseLocation, type Route } from "../lib/nav";

/** Pagina corrente letta dall'indirizzo (#/sezione/scheda): tasto indietro e link diretti funzionano. */
export function useRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() =>
    parseLocation(window.location.hash, window.location.pathname),
  );

  useEffect(() => {
    const onHash = () => {
      setRoute(parseLocation(window.location.hash, window.location.pathname));
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = useCallback((r: Route) => {
    const next = formatRoute(r);
    if (window.location.hash === next) return;
    window.location.hash = next;
  }, []);

  return [route, navigate];
}
