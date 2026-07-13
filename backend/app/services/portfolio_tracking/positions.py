"""Logica posizioni — funzioni PURE (niente DB/rete) per testabilità.

Metodo del costo medio (average cost):
- buy:  cost_basis += qty*prezzo + fee; qty += q
- sell: realized += q*(prezzo - PMC) - fee; cost_basis -= q*PMC; qty -= q

Unità metalli: le quantity possono arrivare in grammi o once; canonicalizziamo
tutto in ONCE TROY (lo spot GC=F/SI=F è USD/oz).
"""
from __future__ import annotations

from typing import Any

GRAMS_PER_TROY_OZ = 31.1034768

# Metalli preziosi: la purezza (millesimi) si applica al contenuto fine.
# Titoli comuni per metallo → l'UI ne fa un picker (default = 999 fine).
METAL_PURITIES: dict[str, list[dict[str, Any]]] = {
    "gold": [
        {"label": "999.9 (lingotto)", "value": 0.9999},
        {"label": "999 (fine)", "value": 0.999},
        {"label": "916.7 (22k · Krugerrand)", "value": 0.9167},
        {"label": "900 (Marengo · 20$)", "value": 0.900},
        {"label": "750 (18k)", "value": 0.750},
    ],
    "silver": [
        {"label": "999 (fine)", "value": 0.999},
        {"label": "925 (sterling)", "value": 0.925},
        {"label": "900 (monete)", "value": 0.900},
        {"label": "835", "value": 0.835},
        {"label": "800", "value": 0.800},
    ],
    "platinum": [
        {"label": "999.5 (fine)", "value": 0.9995},
        {"label": "950 (gioielleria)", "value": 0.950},
        {"label": "900", "value": 0.900},
    ],
    "palladium": [
        {"label": "999.5 (fine)", "value": 0.9995},
        {"label": "950", "value": 0.950},
    ],
}

# Asset preset (per l'UI picker e il pricing). Ticker Yahoo:
# - GC=F / SI=F / PL=F / PA=F: futures front-month, proxy spot USD/oz fine
# - BTC-EUR: quotazione diretta in EUR
PRESET_ASSETS: dict[str, dict[str, Any]] = {
    "gold": {
        "label": "Oro",
        "asset_class": "metal",
        "ticker": "GC=F",
        "canonical_unit": "oz",
        "units": ["oz", "g"],
    },
    "silver": {
        "label": "Argento",
        "asset_class": "metal",
        "ticker": "SI=F",
        "canonical_unit": "oz",
        "units": ["oz", "g"],
    },
    "platinum": {
        "label": "Platino",
        "asset_class": "metal",
        "ticker": "PL=F",
        "canonical_unit": "oz",
        "units": ["oz", "g"],
    },
    "palladium": {
        "label": "Palladio",
        "asset_class": "metal",
        "ticker": "PA=F",
        "canonical_unit": "oz",
        "units": ["oz", "g"],
    },
    "bitcoin": {
        "label": "Bitcoin",
        "asset_class": "crypto",
        "ticker": "BTC-EUR",
        "canonical_unit": "btc",
        "units": ["btc"],
    },
}


def to_canonical(quantity: float, unit_price: float, unit: str) -> tuple[float, float]:
    """Converte (quantity, prezzo unitario) nell'unità canonica.

    g → oz: qty/31.1034768, prezzo €/g → €/oz moltiplicando per 31.1034768.
    Il controvalore qty*prezzo resta identico per costruzione.
    """
    if unit == "g":
        return quantity / GRAMS_PER_TROY_OZ, unit_price * GRAMS_PER_TROY_OZ
    return quantity, unit_price


def compute_position(transactions: list[dict]) -> dict:
    """Riduce i movimenti di UN asset a una posizione (average cost).

    transactions: dict con side, quantity, unit, unit_price_eur, fee_eur,
    purity (frazione 0-1, default 1.0), trade_date (ordinamento, id tie-break).
    Ritorna:
    - quantity: peso FISICO posseduto (unità canonica) — quanto hai in mano
    - fine_quantity: contenuto FINE per la valorizzazione (quantity × purezza);
      per ETF/crypto (purezza 1.0) coincide con quantity
    - avg_cost/cost_basis/realized in EUR (il costo NON dipende dalla purezza:
      è quanto hai pagato per l'oggetto fisico)
    Una sell oltre la quantità posseduta viene clampata (dati manuali: meglio
    degradare che esplodere).
    """
    txs = sorted(transactions, key=lambda t: (t.get("trade_date"), t.get("id", 0)))
    qty = 0.0
    fine = 0.0
    cost_basis = 0.0
    realized = 0.0
    for t in txs:
        q, price = to_canonical(
            float(t["quantity"]), float(t["unit_price_eur"]), t.get("unit", "unit"),
        )
        purity = float(t.get("purity") if t.get("purity") is not None else 1.0)
        fee = float(t.get("fee_eur") or 0.0)
        if t.get("side", "buy") == "buy":
            cost_basis += q * price + fee
            qty += q
            fine += q * purity
        else:
            q = min(q, qty)  # clamp: non si vende più di quanto si ha
            avg = cost_basis / qty if qty > 0 else 0.0
            # riduzione fine proporzionale alla purezza media del posseduto
            fine_ratio = fine / qty if qty > 1e-12 else 0.0
            realized += q * (price - avg) - fee
            cost_basis -= q * avg
            fine -= q * fine_ratio
            qty -= q
    avg_cost = cost_basis / qty if qty > 1e-12 else 0.0
    return {
        "quantity": qty,
        "fine_quantity": fine,
        "avg_cost_eur": avg_cost,
        "cost_basis_eur": cost_basis,
        "realized_pnl_eur": realized,
    }


def build_portfolio(
    transactions: list[dict],
    spot_eur_by_key: dict[str, float | None],
) -> dict:
    """Aggrega TUTTI i movimenti di un utente in posizioni + totali.

    spot_eur_by_key: prezzo EUR per unità canonica per asset_key (None = spot
    non disponibile → market value null, il resto del portafoglio non si rompe).
    """
    by_key: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    for t in transactions:
        key = t["asset_key"]
        by_key.setdefault(key, []).append(t)
        meta[key] = {
            "label": t.get("label", key),
            "asset_class": t.get("asset_class", "ticker"),
            "ticker": t.get("ticker", key),
        }

    positions = []
    for key, txs in by_key.items():
        pos = compute_position(txs)
        if pos["quantity"] <= 1e-12 and abs(pos["realized_pnl_eur"]) < 0.005:
            continue  # asset mai posseduto davvero (o dati nulli)
        spot = spot_eur_by_key.get(key)
        # valorizzazione sul contenuto FINE (per i metalli impuri < peso fisico)
        market_value = pos["fine_quantity"] * spot if spot is not None else None
        pnl = (market_value - pos["cost_basis_eur"]) if market_value is not None else None
        pnl_pct = (
            pnl / pos["cost_basis_eur"] * 100
            if pnl is not None and pos["cost_basis_eur"] > 1e-9
            else None
        )
        preset = PRESET_ASSETS.get(key, {})
        avg_purity = (
            pos["fine_quantity"] / pos["quantity"] if pos["quantity"] > 1e-12 else 1.0
        )
        positions.append({
            "asset_key": key,
            "label": meta[key]["label"],
            "asset_class": meta[key]["asset_class"],
            "ticker": meta[key]["ticker"],
            "unit": preset.get("canonical_unit", "unit"),
            "quantity": round(pos["quantity"], 8),
            "fine_quantity": round(pos["fine_quantity"], 8),
            # purezza media mostrata solo se il metallo non è fine (≈1.0)
            "avg_purity": round(avg_purity, 4) if avg_purity < 0.9995 else None,
            "avg_cost_eur": round(pos["avg_cost_eur"], 4),
            "invested_eur": round(pos["cost_basis_eur"], 2),
            "realized_pnl_eur": round(pos["realized_pnl_eur"], 2),
            "spot_eur": round(spot, 4) if spot is not None else None,
            "market_value_eur": round(market_value, 2) if market_value is not None else None,
            "pnl_eur": round(pnl, 2) if pnl is not None else None,
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        })

    valued = [p for p in positions if p["market_value_eur"] is not None]
    total_value = sum(p["market_value_eur"] for p in valued)
    total_invested = sum(p["invested_eur"] for p in positions)
    total_realized = sum(p["realized_pnl_eur"] for p in positions)
    for p in positions:
        p["allocation_pct"] = (
            round(p["market_value_eur"] / total_value * 100, 2)
            if p["market_value_eur"] is not None and total_value > 1e-9
            else None
        )
    positions.sort(key=lambda p: -(p["market_value_eur"] or 0))

    invested_valued = sum(p["invested_eur"] for p in valued)
    total_pnl = total_value - invested_valued if valued else None
    return {
        "positions": positions,
        "totals": {
            "invested_eur": round(total_invested, 2),
            "market_value_eur": round(total_value, 2) if valued else None,
            "realized_pnl_eur": round(total_realized, 2),
            "pnl_eur": round(total_pnl, 2) if total_pnl is not None else None,
            "pnl_pct": (
                round(total_pnl / invested_valued * 100, 2)
                if total_pnl is not None and invested_valued > 1e-9
                else None
            ),
        },
    }
