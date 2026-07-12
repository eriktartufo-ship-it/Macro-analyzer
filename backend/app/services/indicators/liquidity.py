"""P1 2026-07-12 — Net liquidity Fed: WALCL − RRP − TGA.

Il beta dei risk-asset segue la liquidità netta effettivamente disponibile
al sistema: il balance sheet Fed (WALCL) al netto di ciò che è parcheggiato
in reverse repo (RRP) e sul conto del Tesoro (TGA). WALCL e RRP erano già
registrati (usati dal modulo dedollar); mancava la TGA — terzo pezzo.

⚠️ UNITÀ FRED DIVERSE (verificate live 2026-07-12):
- WALCL   = $ MILIONI  (6,735,609 = $6.74T)
- RRPONTSYD = $ MILIARDI (0.545 = $0.5B — facility quasi drenata nel 2026)
- WTREGEN = $ MILIONI  (774,062 = $774B)

Output normalizzato in $ MILIARDI.
"""
from __future__ import annotations


def compute_net_liquidity(
    walcl_musd: float | None,
    rrp_busd: float | None,
    tga_musd: float | None,
) -> float | None:
    """Net liquidity in $B = WALCL($M)/1000 − RRP($B) − TGA($M)/1000.

    None se manca uno qualunque dei tre pezzi (mai stimare silenziosamente).
    """
    if walcl_musd is None or rrp_busd is None or tga_musd is None:
        return None
    try:
        walcl_bn = float(walcl_musd) / 1000.0
        rrp_bn = float(rrp_busd)
        tga_bn = float(tga_musd) / 1000.0
    except (TypeError, ValueError):
        return None
    # Sanity: WALCL storicamente 0.8-9T in $B
    if not (800.0 <= walcl_bn <= 12000.0):
        return None
    return walcl_bn - rrp_bn - tga_bn
