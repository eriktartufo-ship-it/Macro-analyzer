import { useEffect, useState } from "react";

import { api } from "../api/client";

/** Modello LLM attualmente IMPOSTATO, risolto dal backend (runtime → env → default).
 *
 * Perche' un hook e non una costante: fino al 2026-09-05 ogni pannello scriveva a mano
 * "Gemini 2.5 Flash" nella propria intestazione. Quando il modello di progetto e' passato
 * a `gemini-3.8-flash`, quelle scritte non sono cambiate — e una UI che dichiara con
 * sicurezza il modello sbagliato e' peggio di una che non lo dice, perche' nessuno va a
 * ricontrollare un dato che sembra gia' li'.
 *
 * ⚠️ Questo e' il modello impostato ADESSO, cioe' quello che verrebbe usato per una NUOVA
 * generazione. Per un contenuto gia' prodotto e messo in cache (analisi macro, spiegazione
 * dedollarizzazione, analisi FOMC, consiglio di portafoglio) NON si usa questo: si usa il
 * modello che quel contenuto porta con se' (`provider`, `explanation_model`), perche' puo'
 * essere stato scritto da un altro modello mesi fa.
 *
 * Una sola richiesta per montaggio, senza retry: se fallisce si ritorna `null` e chi lo usa
 * mostra un ripiego neutro invece di inventare un nome.
 */
export function useCurrentLlmModel(): string | null {
  const [model, setModel] = useState<string | null>(null);

  useEffect(() => {
    let vivo = true;
    api
      .llmSettings()
      .then((s) => {
        if (vivo) setModel(s.model || null);
      })
      .catch(() => {
        if (vivo) setModel(null);
      });
    return () => {
      vivo = false;
    };
  }, []);

  return model;
}
