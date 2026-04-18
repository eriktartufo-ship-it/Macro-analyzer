"""Codici FRED per tutti gli indicatori macro utilizzati dal sistema."""

# Struttura: {nome_interno: {fred_id, description, frequency, transform}}

FRED_SERIES = {
    # --- GDP & Output ---
    "gdp": {
        "fred_id": "GDP",
        "description": "Gross Domestic Product (nominal, quarterly)",
        "frequency": "quarterly",
        "transform": "roc",
    },
    "real_gdp": {
        "fred_id": "GDPC1",
        "description": "Real GDP (chained 2017 dollars, quarterly)",
        "frequency": "quarterly",
        "transform": "roc",
    },
    # --- Inflation ---
    "cpi": {
        "fred_id": "CPIAUCSL",
        "description": "Consumer Price Index (All Urban, SA)",
        "frequency": "monthly",
        "transform": "yoy",
    },
    "core_cpi": {
        "fred_id": "CPILFESL",
        "description": "Core CPI (less food & energy, SA)",
        "frequency": "monthly",
        "transform": "yoy",
    },
    "pce": {
        "fred_id": "PCEPI",
        "description": "PCE Price Index",
        "frequency": "monthly",
        "transform": "yoy",
    },
    "core_pce": {
        "fred_id": "PCEPILFE",
        "description": "Core PCE Price Index (Fed preferred)",
        "frequency": "monthly",
        "transform": "yoy",
    },
    # --- Labor Market ---
    "unrate": {
        "fred_id": "UNRATE",
        "description": "Unemployment Rate",
        "frequency": "monthly",
        "transform": "level",
    },
    "initial_claims": {
        "fred_id": "ICSA",
        "description": "Initial Jobless Claims (weekly, SA)",
        "frequency": "weekly",
        "transform": "roc",
    },
    "nonfarm_payrolls": {
        "fred_id": "PAYEMS",
        "description": "Total Nonfarm Payrolls",
        "frequency": "monthly",
        "transform": "roc",
    },
    # --- Manufacturing & Business ---
    "pmi_manufacturing": {
        "fred_id": "MANEMP",
        "description": "Manufacturing Employment (proxy PMI)",
        "frequency": "monthly",
        "transform": "level",
    },
    "ism_manufacturing": {
        "fred_id": "NAPM",
        "description": "ISM Manufacturing PMI",
        "frequency": "monthly",
        "transform": "level",
    },
    "industrial_production": {
        "fred_id": "INDPRO",
        "description": "Industrial Production Index",
        "frequency": "monthly",
        "transform": "roc",
    },
    # --- Leading Indicators ---
    "lei": {
        "fred_id": "USSLIND",
        "description": "Leading Economic Index (Conference Board)",
        "frequency": "monthly",
        "transform": "roc",
    },
    # --- Interest Rates & Yield Curve ---
    "fed_funds": {
        "fred_id": "FEDFUNDS",
        "description": "Effective Federal Funds Rate",
        "frequency": "monthly",
        "transform": "level",
    },
    "treasury_10y": {
        "fred_id": "DGS10",
        "description": "10-Year Treasury Constant Maturity Rate",
        "frequency": "daily",
        "transform": "level",
    },
    "treasury_2y": {
        "fred_id": "DGS2",
        "description": "2-Year Treasury Constant Maturity Rate",
        "frequency": "daily",
        "transform": "level",
    },
    "yield_curve_10y2y": {
        "fred_id": "T10Y2Y",
        "description": "10Y-2Y Treasury Spread",
        "frequency": "daily",
        "transform": "level",
    },
    "yield_curve_10y3m": {
        "fred_id": "T10Y3M",
        "description": "10Y-3M Treasury Spread",
        "frequency": "daily",
        "transform": "level",
    },
    # --- Credit & Financial Conditions ---
    "baa_spread": {
        "fred_id": "BAA10Y",
        "description": "BAA Corporate Bond Spread over 10Y",
        "frequency": "daily",
        "transform": "level",
    },
    "financial_stress": {
        "fred_id": "STLFSI2",
        "description": "St. Louis Fed Financial Stress Index",
        "frequency": "weekly",
        "transform": "level",
    },
    # --- Market Expectations (forward-looking) ---
    "breakeven_10y": {
        "fred_id": "T10YIE",
        "description": "10-Year Breakeven Inflation Rate (market inflation expectations)",
        "frequency": "daily",
        "transform": "level",
    },
    "vix": {
        "fred_id": "VIXCLS",
        "description": "CBOE Volatility Index (market fear gauge)",
        "frequency": "daily",
        "transform": "level",
    },
    "nfci": {
        "fred_id": "NFCI",
        "description": "Chicago Fed National Financial Conditions Index (>0 = tight)",
        "frequency": "weekly",
        "transform": "level",
    },
    # --- Money Supply ---
    "m2": {
        "fred_id": "M2SL",
        "description": "M2 Money Stock",
        "frequency": "monthly",
        "transform": "roc",
    },
    # --- Housing ---
    "housing_starts": {
        "fred_id": "HOUST",
        "description": "Housing Starts (thousands, SAAR)",
        "frequency": "monthly",
        "transform": "roc",
    },
    # --- Consumer ---
    "retail_sales": {
        "fred_id": "RSAFS",
        "description": "Retail Sales (Advance)",
        "frequency": "monthly",
        "transform": "roc",
    },
    "consumer_sentiment": {
        "fred_id": "UMCSENT",
        "description": "University of Michigan Consumer Sentiment",
        "frequency": "monthly",
        "transform": "level",
    },
    # --- Dedollarization ---
    "dxy_broad": {
        "fred_id": "DTWEXBGS",
        "description": "Nominal Broad U.S. Dollar Index (trade-weighted)",
        "frequency": "daily",
        "transform": "roc",
    },
    "gold_price": {
        "fred_id": "IQ12260",
        "description": "Export Price Index: Nonmonetary Gold (proxy for gold price trend, monthly)",
        "frequency": "monthly",
        "transform": "roc",
    },
    "oil_price": {
        "fred_id": "DCOILWTICO",
        "description": "Crude Oil Prices: WTI (USD/Barrel)",
        "frequency": "daily",
        "transform": "roc",
    },
    "debt_gdp": {
        "fred_id": "GFDEGDQ188S",
        "description": "Federal Debt: Total Public Debt as % of GDP",
        "frequency": "quarterly",
        "transform": "level",
    },
    # --- Dedollarization: Macro Players (Step 1) ---
    # SYSTEM / Trust
    "sp500": {
        "fred_id": "SP500",
        "description": "S&P 500 Index",
        "frequency": "daily",
        "transform": "level",
    },
    "copper_price": {
        "fred_id": "PCOPPUSDM",
        "description": "Global Price of Copper (USD/MT, monthly)",
        "frequency": "monthly",
        "transform": "level",
    },
    # USA / Hegemony
    "real_yield_10y": {
        "fred_id": "DFII10",
        "description": "10Y Treasury Real Yield (TIPS)",
        "frequency": "daily",
        "transform": "level",
    },
    "interest_outlays": {
        "fred_id": "A091RC1Q027SBEA",
        "description": "Federal Government: Interest Payments (quarterly, $B SAAR)",
        "frequency": "quarterly",
        "transform": "level",
    },
    "tax_receipts": {
        "fred_id": "FYFRGDA188S",
        "description": "Federal Receipts as % of GDP (annual)",
        "frequency": "annual",
        "transform": "level",
    },
    "gross_federal_debt": {
        "fred_id": "GFDEBTN",
        "description": "Gross Federal Debt (millions USD, quarterly)",
        "frequency": "quarterly",
        "transform": "level",
    },
    "foreign_treasury_holdings": {
        "fred_id": "FDHBFIN",
        "description": "Foreign & International Holdings of Treasury Securities (millions USD)",
        "frequency": "monthly",
        "transform": "roc",
    },
    # EUROPE / Stability
    "italy_10y": {
        "fred_id": "IRLTLT01ITM156N",
        "description": "Italy 10Y Government Bond Yield",
        "frequency": "monthly",
        "transform": "level",
    },
    "germany_10y": {
        "fred_id": "IRLTLT01DEM156N",
        "description": "Germany 10Y Government Bond Yield",
        "frequency": "monthly",
        "transform": "level",
    },
    "chf_per_usd": {
        "fred_id": "DEXSZUS",
        "description": "Swiss Francs per USD (CHF/USD)",
        "frequency": "daily",
        "transform": "level",
    },
    "usd_per_eur": {
        "fred_id": "DEXUSEU",
        "description": "USD per Euro (USD/EUR)",
        "frequency": "daily",
        "transform": "level",
    },
    # JAPAN / Carry
    "japan_10y": {
        "fred_id": "IRLTLT01JPM156N",
        "description": "Japan 10Y Government Bond Yield",
        "frequency": "monthly",
        "transform": "level",
    },
    "jpy_per_usd": {
        "fred_id": "DEXJPUS",
        "description": "Japanese Yen per USD (JPY/USD)",
        "frequency": "daily",
        "transform": "level",
    },
    # COMMODITY FX
    "cad_per_usd": {
        "fred_id": "DEXCAUS",
        "description": "Canadian Dollars per USD (CAD/USD)",
        "frequency": "daily",
        "transform": "level",
    },
    "usd_per_aud": {
        "fred_id": "DEXUSAL",
        "description": "USD per Australian Dollar (USD/AUD)",
        "frequency": "daily",
        "transform": "level",
    },
    # EM
    "em_hy_oas": {
        "fred_id": "BAMLH0A0HYM2",
        "description": "ICE BofA US High Yield Index OAS (risk-on proxy)",
        "frequency": "daily",
        "transform": "level",
    },
    # DEFENSE
    "defense_spending": {
        "fred_id": "A997RC1Q027SBEA",
        "description": "National Defense Consumption Expenditures (quarterly, annualized)",
        "frequency": "quarterly",
        "transform": "level",
    },
    # --- Dedollarization Step 2: Strutturale + Aspettative + Fed Liquidity + BRICS+ + Metals breadth ---
    # USA: Dollaro reale + Twin deficit + Monetizzazione Fed
    "real_broad_dxy": {
        "fred_id": "RTWEXBGS",
        "description": "Real Trade Weighted U.S. Dollar Index: Broad, Goods & Services (monthly, 2006=100)",
        "frequency": "monthly",
        "transform": "roc",
    },
    "current_account": {
        "fred_id": "IEABC",
        "description": "Balance on Current Account (quarterly, millions USD)",
        "frequency": "quarterly",
        "transform": "level",
    },
    "niip": {
        "fred_id": "IIPUSNETIQ",
        "description": "Net International Investment Position of the US (quarterly, millions USD)",
        "frequency": "quarterly",
        "transform": "level",
    },
    "fed_debt_holdings": {
        "fred_id": "FDHBFRBN",
        "description": "Federal Debt Held by Federal Reserve Banks (quarterly, billions USD)",
        "frequency": "quarterly",
        "transform": "level",
    },
    # EXPECTATIONS: aspettative long-run + term premium
    "breakeven_5y5y": {
        "fred_id": "T5YIFR",
        "description": "5-Year, 5-Year Forward Inflation Expectation Rate (daily, %)",
        "frequency": "daily",
        "transform": "level",
    },
    "term_premium_10y": {
        "fred_id": "THREEFYTP10",
        "description": "ACM 10-Year Treasury Term Premium (monthly, %)",
        "frequency": "monthly",
        "transform": "level",
    },
    # FED LIQUIDITY
    "fed_balance_sheet": {
        "fred_id": "WALCL",
        "description": "Federal Reserve Total Assets (weekly, millions USD)",
        "frequency": "weekly",
        "transform": "roc",
    },
    "reverse_repo": {
        "fred_id": "RRPONTSYD",
        "description": "Overnight Reverse Repurchase Agreements: Treasury Securities (daily, billions USD)",
        "frequency": "daily",
        "transform": "level",
    },
    # BRICS+ (poli alternativi)
    "cny_per_usd": {
        "fred_id": "DEXCHUS",
        "description": "Chinese Yuan Renminbi per USD (daily)",
        "frequency": "daily",
        "transform": "level",
    },
    "india_10y": {
        "fred_id": "INDIRLTLT01STM",
        "description": "India 10Y Government Bond Yield (monthly, %)",
        "frequency": "monthly",
        "transform": "level",
    },
    "brazil_policy_rate": {
        "fred_id": "INTGSTBRM193N",
        "description": "Brazil Short-Term Treasury Bill / Policy Rate (monthly, %)",
        "frequency": "monthly",
        "transform": "level",
    },
    # METALS BREADTH
    "silver_price": {
        "fred_id": "NASDAQQSLVO",
        "description": "Credit Suisse NASDAQ Silver FLOWS106 Price Index (daily)",
        "frequency": "daily",
        "transform": "roc",
    },
    # --- Step 3: EM breadth + Europa ---
    "em_fx_dollar_index": {
        "fred_id": "DTWEXEMEGS",
        "description": "Nominal Broad Dollar vs EM Economies (daily, dal 2006)",
        "frequency": "daily",
        "transform": "roc",
    },
    "ecb_balance_sheet": {
        "fred_id": "ECBASSETSW",
        "description": "Central Bank Assets for Euro Area (weekly, dal 1999)",
        "frequency": "weekly",
        "transform": "roc",
    },
    "france_10y": {
        "fred_id": "IRLTLT01FRM156N",
        "description": "France 10Y Government Bond Yield (%, monthly)",
        "frequency": "monthly",
        "transform": "level",
    },
}
