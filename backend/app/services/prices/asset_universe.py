"""Mapping asset class -> ticker Yahoo Finance.

Per ogni asset class del nostro scoring engine, scegliamo l'ETF/proxy piu'
rappresentativo con storia lunga. Quando un proxy moderno ha storia corta
(BTC dal 2014, EM dal 2003), forniamo un fallback (`backfill_proxy`) per
estendere la serie usando un sostituto di buona correlazione.

Note:
- Crypto: BTC-USD start 2014. Pre-2014 backfill non disponibile (asset non esisteva)
- TIPS: TIP ETF dal 2003. Pre-2003 fallback su long bonds (proxy debole)
- REITs: VNQ dal 2004. Pre-2004 fallback su settore immobiliare via DJIA RE
"""

# ticker primario + descrizione + ticker fallback per backfill storico
ASSET_TICKERS: dict[str, dict[str, str | None]] = {
    "us_equities_growth": {
        "ticker": "QQQ",  # Nasdaq 100 (growth proxy), 1999+
        "backfill_proxy": "^IXIC",  # Nasdaq Composite, 1971+
        "description": "Nasdaq-100 ETF / Nasdaq Composite (growth)",
    },
    "us_equities_value": {
        "ticker": "VTV",  # Vanguard Value ETF, 2004+
        "backfill_proxy": "^DJI",  # Dow Jones, 1929+
        "description": "Vanguard Value / Dow Jones (value/dividend)",
    },
    "international_dm_equities": {
        "ticker": "EFA",  # iShares MSCI EAFE, 2001+
        "backfill_proxy": None,
        "description": "MSCI EAFE (Developed ex-US)",
    },
    "em_equities": {
        "ticker": "EEM",  # iShares MSCI Emerging Markets, 2003+
        "backfill_proxy": None,
        "description": "MSCI Emerging Markets",
    },
    "us_bonds_short": {
        "ticker": "SHY",  # 1-3Y Treasury, 2002+
        # NOTA: ^IRX e' un yield (livello), non un total return.
        # Pre-2002 servirebbe ricostruire dai yield (DGS2 da FRED) — non
        # forniamo backfill per evitare numeri falsi.
        "backfill_proxy": None,
        "description": "1-3 Year Treasury (TR, post-2002)",
    },
    "us_bonds_long": {
        "ticker": "TLT",  # 20+Y Treasury, 2002+
        # ^TYX e' yield, non TR. Pre-2002: backfill robusto richiede price ricostruito
        # dal yield via duration formula. Lasciamo None per ora.
        "backfill_proxy": None,
        "description": "20+ Year Treasury (TR, post-2002)",
    },
    "tips_inflation_bonds": {
        "ticker": "TIP",  # iShares TIPS Bond, 2003+
        "backfill_proxy": None,
        "description": "Inflation-Protected Treasuries",
    },
    "gold": {
        "ticker": "GLD",  # SPDR Gold, 2004+
        "backfill_proxy": "GC=F",  # Gold futures, 2000+
        "description": "Gold (SPDR / futures)",
    },
    "silver": {
        "ticker": "SLV",  # iShares Silver, 2006+
        "backfill_proxy": "SI=F",  # Silver futures, 2000+
        "description": "Silver",
    },
    "broad_commodities": {
        "ticker": "DBC",  # Invesco DB Commodity, 2006+
        "backfill_proxy": "^SPGSCI",  # GSCI, 1969+
        "description": "Broad Commodities (DBC / GSCI)",
    },
    "energy": {
        "ticker": "XLE",  # Energy Select SPDR, 1998+
        "backfill_proxy": "CL=F",  # Crude oil futures, 2000+
        "description": "Energy sector / WTI",
    },
    "real_estate_reits": {
        "ticker": "VNQ",  # Vanguard REIT, 2004+
        "backfill_proxy": "IYR",  # iShares US Real Estate, 2000+
        "description": "REITs",
    },
    "cash_money_market": {
        "ticker": "BIL",  # SPDR 1-3M T-bill TR, 2007+
        # ^IRX e' yield level, non total return. Pre-2007: ricostruzione richiesta.
        "backfill_proxy": None,
        "description": "1-3M T-bill TR (post-2007)",
    },
    "bitcoin": {
        "ticker": "BTC-USD",  # 2014+
        "backfill_proxy": None,
        "description": "Bitcoin",
    },
    "crypto_broad": {
        "ticker": "ETH-USD",  # 2017+ (proxy crypto-broad)
        "backfill_proxy": "BTC-USD",
        "description": "Crypto broad (ETH / BTC fallback)",
    },
}


# Indici di riferimento extra (non asset class ma utili come benchmark/feature)
BENCHMARK_TICKERS = {
    "sp500": "^GSPC",  # 1928+
    "nasdaq": "^IXIC",  # 1971+
    "dollar_index": "DX-Y.NYB",  # DXY 1971+
    "vix": "^VIX",  # 1990+ (gia' su FRED come VIXCLS, ridondante)
    "move": "^MOVE",  # bond volatility, 1988+
    "copper_gold_ratio": None,  # calcolato derivato
}
