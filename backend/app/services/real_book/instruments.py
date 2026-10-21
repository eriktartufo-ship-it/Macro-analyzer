"""Registro strumenti del libro reale: per ogni asset class il gemello UCITS/ETC in euro.

Il SEGNALE resta calcolato sull'ETF americano (`prices/asset_universe.py`: storia lunga).
L'ESECUZIONE avviene su uno strumento che un investitore italiano può comprare davvero
(Scalable: Xetra/gettex; alcuni solo su Borsa Italiana). Tutte le quotazioni scelte sono
in EUR, quindi il cambio EUR/USD entra nel risultato da solo, come per un investitore vero.

Verifica fatta il 2026-10-21 (nessun ISIN è scritto a memoria senza controllo):
- nome, borsa e valuta letti da Yahoo (`Ticker.info`) per ogni simbolo;
- ISIN confermato da due fonti indipendenti: la ricerca Yahoo per ISIN restituisce
  un'altra quotazione DELLO STESSO fondo (es. IE00B579F325 → SGLD.L) e, dove serviva,
  una scheda web (Deutsche Börse, justETF/finanzen, SSGA).
- Una memoria sbagliata scoperta così: IE00BD1F4L37 è il fattore QUALITY, non momentum.
- Due coppie Xetra/Milano che NON sono lo stesso prodotto (prezzi diversi): IQQU.DE vs
  IUSP.MI, BCFE.DE vs CMOD.MI ⇒ per REIT e materie prime si usa la riga di Milano.

`half_spread` è una STIMA dichiarata (metà della forbice denaro/lettera), non una misura:
0,05% sui fondi molto scambiati, 0,15% su settori/fattori poco scambiati, 0,20% sui
crypto ETP. Si cambia qui, in un posto solo.
Il TER non c'è di proposito: è già dentro il prezzo dell'ETF, e un numero scritto a
memoria qui sarebbe stato l'unico dato non verificato del registro.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    key: str            # chiave interna stabile
    name: str           # nome ufficiale (privato: non va sul sito pubblico)
    ticker: str         # simbolo di borsa locale
    yahoo: str          # simbolo Yahoo per i prezzi (OHLC in EUR)
    isin: str
    exchange: str       # Xetra | Borsa Italiana
    kind: str           # ETF | ETC | ETP
    half_spread: float  # stima, frazione (0.0005 = 0,05%)
    distributes: bool   # paga cedole (accreditate in cassa alla data di stacco)

    @property
    def tax_rate(self) -> float:
        """Aliquota sulle plusvalenze e sulle cedole: 12,5% sui fondi di titoli di Stato USA
        (white list, ~100% del patrimonio), 26% sul resto."""
        return 0.125 if self.key in _GOVT_BOND_FUNDS else 0.26

    @property
    def tax_class(self) -> str:
        """`oicr` = fondo/ETF: la plusvalenza è reddito di CAPITALE, non si compensa con le
        minusvalenze; la minusvalenza invece entra nello zainetto. `diversi` = ETC/ETN/ETP: la
        plusvalenza è reddito DIVERSO e si compensa con lo zainetto."""
        return "diversi" if self.kind in ("ETC", "ETP") else "oicr"


_I = Instrument

INSTRUMENTS: dict[str, Instrument] = {i.key: i for i in (
    _I("EQQQ", "Invesco EQQQ NASDAQ-100 UCITS ETF (Dist)", "EQQQ", "EQQQ.DE", "IE0032077012", "Xetra", "ETF", 0.0005, True),
    _I("ZPRU", "SPDR MSCI USA Value UCITS ETF (Acc)", "ZPRU", "ZPRU.DE", "IE00BSPLC520", "Xetra", "ETF", 0.0015, False),
    _I("IXUA", "iShares MSCI World ex-USA UCITS ETF (Acc)", "IXUA", "IXUA.DE", "IE000R4ZNTN3", "Xetra", "ETF", 0.0005, False),
    _I("EIMI", "iShares Core MSCI EM IMI UCITS ETF (Acc)", "IS3N", "IS3N.DE", "IE00BKM4GZ66", "Xetra", "ETF", 0.0005, False),
    _I("IBTS", "iShares $ Treasury Bond 1-3yr UCITS ETF (Dist)", "IUSU", "IUSU.DE", "IE00B14X4S71", "Xetra", "ETF", 0.0005, True),
    _I("IDTL", "iShares $ Treasury Bond 20+yr UCITS ETF (Dist)", "IS04", "IS04.DE", "IE00BSKRJZ44", "Xetra", "ETF", 0.0005, True),
    _I("ITPS", "iShares $ TIPS UCITS ETF (Acc)", "IUST", "IUST.DE", "IE00B1FZSC47", "Xetra", "ETF", 0.0010, False),
    _I("SGLD", "Invesco Physical Gold ETC", "8PSG", "8PSG.DE", "IE00B579F325", "Xetra", "ETC", 0.0008, False),
    _I("PHAG", "WisdomTree Physical Silver", "PHAG", "PHAG.MI", "JE00B1VS3333", "Borsa Italiana", "ETC", 0.0010, False),
    _I("CMOD", "Invesco Bloomberg Commodity UCITS ETF (Acc)", "CMOD", "CMOD.MI", "IE00BD6FTQ80", "Borsa Italiana", "ETF", 0.0010, False),
    _I("SXLE", "SPDR S&P U.S. Energy Select Sector UCITS ETF (Acc)", "ZPDE", "ZPDE.DE", "IE00BWBXM492", "Xetra", "ETF", 0.0015, False),
    _I("IUSP", "iShares US Property Yield UCITS ETF (Dist)", "IUSP", "IUSP.MI", "IE00B1FZSF77", "Borsa Italiana", "ETF", 0.0010, True),
    _I("XEON", "Xtrackers II EUR Overnight Rate Swap UCITS ETF 1C", "XEON", "XEON.DE", "LU0290358497", "Xetra", "ETF", 0.0005, False),
    _I("IB1T", "iShares Bitcoin ETP", "IB1T", "IB1T.DE", "XS2940466316", "Xetra", "ETP", 0.0020, False),
    _I("ZETH", "Bitwise Physical Ethereum ETP", "ZETH", "ZETH.DE", "DE000A3GMKD7", "Xetra", "ETP", 0.0020, False),
    _I("SXLF", "SPDR S&P U.S. Financials Select Sector UCITS ETF (Acc)", "ZPDF", "ZPDF.DE", "IE00BWBXM500", "Xetra", "ETF", 0.0015, False),
    _I("SXLU", "SPDR S&P U.S. Utilities Select Sector UCITS ETF (Acc)", "ZPDU", "ZPDU.DE", "IE00BWBXMB69", "Xetra", "ETF", 0.0015, False),
    _I("SXLK", "SPDR S&P U.S. Technology Select Sector UCITS ETF (Acc)", "ZPDT", "ZPDT.DE", "IE00BWBXM948", "Xetra", "ETF", 0.0015, False),
    _I("SXLV", "SPDR S&P U.S. Health Care Select Sector UCITS ETF (Acc)", "ZPDH", "ZPDH.DE", "IE00BWBXM617", "Xetra", "ETF", 0.0015, False),
    _I("SXLP", "SPDR S&P U.S. Consumer Staples Select Sector UCITS ETF (Acc)", "ZPDS", "ZPDS.DE", "IE00BWBXM385", "Xetra", "ETF", 0.0015, False),
    _I("SMH", "VanEck Semiconductor UCITS ETF (Acc)", "VVSM", "VVSM.DE", "IE00BMC38736", "Xetra", "ETF", 0.0010, False),
    _I("IUMO", "iShares Edge MSCI USA Momentum Factor UCITS ETF (Acc)", "QDVA", "QDVA.DE", "IE00BD1F4N50", "Xetra", "ETF", 0.0015, False),
    _I("IUQA", "iShares Edge MSCI USA Quality Factor UCITS ETF (Acc)", "QDVB", "QDVB.DE", "IE00BD1F4L37", "Xetra", "ETF", 0.0015, False),
    _I("IUVL", "iShares Edge MSCI USA Value Factor UCITS ETF (Acc)", "QDVI", "QDVI.DE", "IE00BD1F4M44", "Xetra", "ETF", 0.0015, False),
    _I("VUAA", "Vanguard S&P 500 UCITS ETF (Acc)", "VUAA", "VUAA.DE", "IE00BFMXXD54", "Xetra", "ETF", 0.0005, False),
    # 2026-10-21: benchmark MSCI ACWI (Erik: «più simile al mio portafoglio»). ISIN da due fonti:
    # nome Yahoo di IUSQ.DE + ricerca per ISIN che restituisce ISAC.L (stesso fondo a Londra).
    _I("IUSQ", "iShares MSCI ACWI UCITS ETF (Acc)", "IUSQ", "IUSQ.DE", "IE00B6R52259", "Xetra", "ETF", 0.0005, False),
)}

_GOVT_BOND_FUNDS = {"IBTS", "IDTL", "ITPS"}

# Benchmark in euro del libro reale, come li comprerebbe un investitore italiano.
BENCHMARK_KEY = "VUAA"          # S&P 500
BENCHMARK_ACWI_KEY = "IUSQ"     # MSCI ACWI (azionario mondiale, paesi emergenti compresi)
BENCHMARK_BOND_KEY = "IDTL"     # la gamba 40 del 60/40 (stessa definizione dei bot: Treasury 20+ anni)

# asset class (scoring engine) → (strumento, etichetta pubblica in italiano)
# ⚠️ `energy` e `sector_energy` hanno lo STESSO segnale (XLE) ⇒ stesso strumento.
# Il libro li copia fedelmente e la pagina segnala la concentrazione: non è il libro a
# correggere le decisioni dei bot, altrimenti non misurerebbe più loro.
ASSET_MAP: dict[str, tuple[str, str]] = {
    "us_equities_growth": ("EQQQ", "Azioni USA growth (Nasdaq-100)"),
    "us_equities_sp500": ("VUAA", "Azioni USA (S&P 500)"),  # solo bot a rischio gestito
    "us_equities_value": ("ZPRU", "Azioni USA value"),
    "international_dm_equities": ("IXUA", "Azioni paesi sviluppati ex-USA"),
    "em_equities": ("EIMI", "Azioni mercati emergenti"),
    "us_bonds_short": ("IBTS", "Titoli di Stato USA 1-3 anni"),
    "us_bonds_long": ("IDTL", "Titoli di Stato USA 20+ anni"),
    "tips_inflation_bonds": ("ITPS", "Titoli di Stato USA indicizzati all'inflazione"),
    "gold": ("SGLD", "Oro fisico"),
    "silver": ("PHAG", "Argento fisico"),
    "broad_commodities": ("CMOD", "Materie prime (paniere)"),
    "energy": ("SXLE", "Energia USA"),
    "real_estate_reits": ("IUSP", "Immobiliare USA (REIT)"),
    "cash_money_market": ("XEON", "Liquidità (tasso overnight euro)"),
    "bitcoin": ("IB1T", "Bitcoin"),
    "crypto_broad": ("ZETH", "Ethereum"),
    "sector_energy": ("SXLE", "Settore energia USA"),
    "sector_financials": ("SXLF", "Settore finanziario USA"),
    "sector_utilities": ("SXLU", "Settore utility USA"),
    "sector_technology": ("SXLK", "Settore tecnologia USA"),
    "sector_healthcare": ("SXLV", "Settore sanità USA"),
    "sector_staples": ("SXLP", "Beni di prima necessità USA"),
    "sector_semiconductors": ("SMH", "Semiconduttori"),
    "factor_momentum": ("IUMO", "Fattore momentum USA"),
    "factor_quality": ("IUQA", "Fattore qualità USA"),
    "factor_value": ("IUVL", "Fattore value USA"),
}


def instrument_for(asset_class: str) -> Instrument | None:
    entry = ASSET_MAP.get(asset_class)
    return INSTRUMENTS.get(entry[0]) if entry else None


def label_for(asset_class: str) -> str:
    entry = ASSET_MAP.get(asset_class)
    return entry[1] if entry else asset_class.replace("_", " ")


def shared_instruments() -> dict[str, list[str]]:
    """Strumenti su cui puntano più asset class (rischio di concentrazione nascosta)."""
    out: dict[str, list[str]] = {}
    for asset, (key, _) in ASSET_MAP.items():
        out.setdefault(key, []).append(asset)
    return {k: v for k, v in out.items() if len(v) > 1}
