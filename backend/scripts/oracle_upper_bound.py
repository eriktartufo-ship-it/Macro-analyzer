"""S42b — IL TETTO MASSIMO: quanto alpha vale un classifier PERFETTO?

LA DOMANDA DI ERIK: *"il modello funziona veramente, se diventa più preciso?"*

Invece di rispondere con un'opinione, si misura il **tetto**: si danno ai trader le
etichette di regime VERE (dagli episodi storici, note solo ex-post — un oracolo che
conosce il futuro) e si guarda l'alpha. Nessun lavoro sul classifier, oggi o mai,
potrà superare questo numero.

  Se oracolo ≈ classifier attuale → il classifier NON è il collo di bottiglia, e
     ogni round speso a renderlo più preciso è sprecato. Risposta definitiva.
  Se oracolo >> classifier attuale → il margine esiste ed è quantificato.

È il [[feedback-chiedi-il-controfattuale-prima-di-costruire]] applicato al classifier:
"e se fosse perfetto, quella risorsa che fine farebbe?".

⚠️ IL CONFONDENTE (isolato apposta): il sizer usa `confidence` per decidere quante
tranche aprire (`dynamic_n_tranches`). Un oracolo con confidence 1.0 alzerebbe il
DEPLOYMENT, non solo la precisione → l'alpha migliorerebbe per la ragione sbagliata e
misureremmo il deployment credendo di misurare il regime. Quindi due bracci:

  ORACLE-REGIME : regime vero, `confidence` LASCIATA quella del classifier reale
                  → isola l'effetto della SOLA precisione del regime.
  ORACLE-FULL   : regime vero + confidence 1.0
                  → il tetto vero (precisione + convinzione).

La differenza fra i due bracci = quanto dell'effetto è deployment e non precisione.

Uso:
    python scripts/oracle_upper_bound.py [--seeds 42,7,99] [--n-sims 15]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Allinea i flag al live PRIMA di importare (come fa horserace_backtest)
os.environ.setdefault("USE_MOMENTUM_PILLARS", "1")
os.environ.setdefault("USE_AI_PORTFOLIO_DEPLOY_BOOST", "0")

from app.services.backtest.walk_forward import WalkForwardCache  # noqa: E402
from app.services.regime.historical_episodes import HISTORICAL_EPISODES  # noqa: E402

_REGIMES = ("reflation", "stagflation", "deflation", "goldilocks")


def oracle_regime(day: date) -> str | None:
    """Il regime VERO per quella data, dagli episodi etichettati a mano.

    Gli episodi si sovrappongono (es. "2020-21 post-COVID reflation" contiene il
    "2020 Q1 deflation"): vince il più SPECIFICO = quello con la finestra più corta.
    None se l'anno non è coperto da nessun episodio (l'oracolo non sa → si lascia
    parlare il classifier reale, così il confronto resta onesto).
    """
    best, best_span = None, 10**6
    for ep in HISTORICAL_EPISODES:
        if ep.start_year <= day.year <= ep.end_year:
            span = ep.end_year - ep.start_year
            if span < best_span:
                best, best_span = ep.true_regime, span
    return best


def install_oracle(full: bool) -> dict:
    """Monkey-patch di WalkForwardCache.get: sostituisce il regime con la verità.

    Gli `indicators` restano quelli veri (il resto della pipeline li usa): si cambia
    SOLO il verdetto di regime, che è la variabile sotto test.
    """
    original = WalkForwardCache.get
    stats = {"hit": 0, "miss": 0}

    def patched(self, as_of, flag_key=None):
        cm = original(self, as_of, flag_key)
        true_r = oracle_regime(as_of)
        if cm is None or true_r is None:
            stats["miss"] += 1
            return cm
        stats["hit"] += 1
        # distribuzione "sicura ma non degenere" sul regime vero
        probs = {r: 0.02 for r in _REGIMES}
        probs[true_r] = 1.0 - 0.02 * (len(_REGIMES) - 1)
        out = dict(cm)
        out["probs"] = probs
        if full:
            out["confidence"] = 1.0  # ORACLE-FULL: anche la convinzione è perfetta
        # ORACLE-REGIME: confidence lasciata identica a quella del classifier reale
        return out

    WalkForwardCache.get = patched
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,99")
    ap.add_argument("--n-sims", type=int, default=15)
    ap.add_argument("--mode", choices=["regime", "full"], default="regime")
    args = ap.parse_args()

    stats = install_oracle(full=(args.mode == "full"))
    print(f"ORACOLO installato · modo = {args.mode.upper()}")
    print(f"  (regime vero dagli episodi etichettati; "
          f"{'confidence 1.0' if args.mode == 'full' else 'confidence del classifier REALE'})")

    import horserace_backtest as hb

    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        series = hb.preload_walk_forward_series(date(1994, 1, 1), date(2024, 12, 31)) \
            if hasattr(hb, "preload_walk_forward_series") else None
        sys.argv = ["horserace_backtest", "--n-sims", str(args.n_sims), "--seed", str(seed)]
        print(f"\n{'='*70}\nSEED {seed} · oracolo {args.mode}\n{'='*70}")
        hb.main()
        results.append(seed)

    print(f"\nOracolo: {stats['hit']} classificazioni sostituite, "
          f"{stats['miss']} lasciate al classifier reale (anno non coperto).")
    print("\nConfronta il 'mean alpha' con la baseline del classifier attuale.")


if __name__ == "__main__":
    main()
