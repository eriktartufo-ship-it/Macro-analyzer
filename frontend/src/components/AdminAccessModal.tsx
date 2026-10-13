import { useEffect, useState } from "react";
import { api } from "../api/client";
import { getAdminToken, setAdminToken } from "../api/adminToken";

// S51 (2026-10-13): il token admin si inserisce qui, una volta per dispositivo.
// Le letture del sito restano pubbliche; il token serve solo per modificare.

interface Props {
  open: boolean;
  onClose: () => void;
}

type Stato = "nessuno" | "verifico" | "valido" | "errore";

const RED = "rgba(255,69,58,.85)"; // .btn-danger canonico: Annulla = rosso pieno, a sinistra

export function AdminAccessModal({ open, onClose }: Props) {
  const [input, setInput] = useState("");
  const [show, setShow] = useState(false);
  const [stato, setStato] = useState<Stato>("nessuno");
  const [msg, setMsg] = useState<string | null>(null);
  const salvato = Boolean(getAdminToken());

  useEffect(() => {
    if (!open) return;
    setInput("");
    setMsg(null);
    if (!getAdminToken()) { setStato("nessuno"); return; }
    setStato("verifico");
    api.adminCheck()
      .then(() => setStato("valido"))
      .catch((e) => { setStato("errore"); setMsg(e instanceof Error ? e.message : String(e)); });
  }, [open]);

  if (!open) return null;

  const salva = async () => {
    const tok = input.trim();
    if (!tok) return;
    setAdminToken(tok);
    setStato("verifico");
    setMsg(null);
    try {
      await api.adminCheck();
      setStato("valido");
      setInput("");
    } catch (e) {
      setAdminToken(null); // un token rifiutato non resta salvato
      setStato("errore");
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };

  const esci = () => {
    setAdminToken(null);
    setStato("nessuno");
    setMsg(null);
  };

  const label = { nessuno: "non inserito", verifico: "verifico…", valido: "valido ✓", errore: "non valido" }[stato];
  const colore = stato === "valido" ? "var(--reflation)" : stato === "errore" ? "var(--deflation)" : "var(--muted)";

  return (
    <>
      <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)", zIndex: 100 }} />
      <div
        role="dialog"
        aria-label="Accesso admin"
        style={{
          position: "fixed", top: "10vh", left: "50%", transform: "translateX(-50%)",
          width: "min(460px, 92vw)", background: "var(--bg)", border: "1px solid var(--stroke)",
          borderRadius: 12, padding: 18, zIndex: 101, backdropFilter: "blur(20px)",
          boxShadow: "0 12px 48px rgba(0,0,0,0.3)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
          <h2 style={{ margin: 0 }}>🔑 Accesso admin</h2>
          <button onClick={onClose} aria-label="Chiudi"
            style={{ background: "transparent", border: "none", fontSize: 22, color: "var(--muted)", cursor: "pointer", lineHeight: 1 }}>
            ×
          </button>
        </div>
        <p style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.5, margin: "0 0 12px" }}>
          Il sito si consulta senza accesso. Per <strong>modificare</strong> (flag del modello, impostazioni LLM,
          refresh dei dati, portafoglio AI, rigenerazione delle analisi) serve il token admin. Resta salvato
          solo su questo dispositivo.
        </p>
        <div style={{ fontSize: 13, marginBottom: 10 }}>
          Stato: <strong style={{ color: colore }}>{label}</strong>
        </div>
        {msg && <div style={{ color: "var(--deflation)", fontSize: 12.5, marginBottom: 10 }}>{msg}</div>}

        {/* Un form vero: 1Password (o il gestore del browser) riconosce la credenziale e la compila. */}
        <form onSubmit={(e) => { e.preventDefault(); void salva(); }}>
        <input type="text" name="username" autoComplete="username" value="macro-admin" readOnly hidden />
        {stato !== "valido" && (
          <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
            <input
              type={show ? "text" : "password"}
              name="password"
              placeholder="Incolla il token admin"
              value={input}
              autoComplete="current-password"
              onChange={(e) => setInput(e.target.value)}
              style={{
                flex: 1, minWidth: 0, padding: "10px 12px", background: "var(--surface-sunk)",
                border: "1px solid var(--stroke)", borderRadius: 8, color: "var(--text)",
                fontSize: 14, fontFamily: "monospace",
              }}
            />
            <button type="button" onClick={() => setShow((v) => !v)} title={show ? "Nascondi" : "Mostra"}
              style={{ background: "var(--surface-sunk)", border: "1px solid var(--stroke)", borderRadius: 8,
                padding: "0 12px", color: "var(--muted)", cursor: "pointer", fontSize: 13, minHeight: 44 }}>
              {show ? "🙈" : "👁"}
            </button>
          </div>
        )}

        <div style={{ display: "flex", gap: 8, justifyContent: "space-between", flexWrap: "wrap" }}>
          <button type="button" onClick={onClose}
            style={{ background: RED, border: "none", borderRadius: 8, padding: "10px 16px", color: "#fff",
              cursor: "pointer", fontWeight: 600, minHeight: 44 }}>
            Annulla
          </button>
          {stato === "valido" || (salvato && stato === "verifico") ? (
            <button type="button" onClick={esci}
              style={{ background: "var(--surface-sunk)", border: "1px solid var(--stroke)", borderRadius: 8,
                padding: "10px 16px", color: "var(--text)", cursor: "pointer", fontWeight: 600, minHeight: 44 }}>
              Esci da questo dispositivo
            </button>
          ) : (
            <button type="submit" disabled={!input.trim()}
              style={{ background: "var(--accent)", border: "none", borderRadius: 8, padding: "10px 16px",
                color: "#fff", cursor: input.trim() ? "pointer" : "default", opacity: input.trim() ? 1 : 0.5,
                fontWeight: 600, minHeight: 44 }}>
              Salva e verifica
            </button>
          )}
        </div>
        </form>
      </div>
    </>
  );
}
