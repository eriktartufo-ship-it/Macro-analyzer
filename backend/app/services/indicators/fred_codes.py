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
    # CFNAIMA3 (Chicago Fed National Activity Index, 3-month MA) sintetizza 85
    # indicatori macro in un singolo z-score (media 0 = trend growth, +1 = sopra,
    # -1 = sotto). Sostituisce USSLIND (state-level, scale incompatibile con
    # le condizioni del classifier — produceva ROC fino a 30%).
    "lei": {
        "fred_id": "CFNAIMA3",
        "description": "Chicago Fed National Activity Index, 3-month MA (proxy LEI)",
        "frequency": "monthly",
        "transform": "level",
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
    # AUDIT 2026-07-12: rimossa la definizione duplicata di hy_credit_spread
    # (ridefinita più sotto, sezione T17: l'ultima vince nel dict — questa era
    # silenziosamente sovrascritta).
    "ig_credit_spread": {
        "fred_id": "BAMLC0A0CM",
        "description": "ICE BofA Investment Grade OAS (% over Treasuries)",
        "frequency": "daily",
        "transform": "level",
    },
    # --- Nowcasting ---
    "gdp_nowcast": {
        "fred_id": "GDPNOW",
        "description": "Atlanta Fed GDPNow (real-time GDP forecast %)",
        "frequency": "monthly",
        "transform": "level",
    },
    # --- Ground truth recessione (per backtesting) ---
    "nber_recession": {
        "fred_id": "USREC",
        "description": "NBER recession indicator (1=recession, 0=expansion)",
        "frequency": "monthly",
        "transform": "level",
    },
    # --- Term premium decomposition (Adrian-Crump-Moench) ---
    # Decompone yield 10Y in expected path (cosa il mercato prevede per Fed)
    # vs term premium (compensation per rischio duration). Standard NY Fed.
    "acm_term_premium_10y": {
        "fred_id": "THREEFYTP10",
        "description": "ACM 10-Year Term Premium (% over expected short rate path)",
        "frequency": "daily",
        "transform": "level",
    },
    "acm_fitted_yield_10y": {
        "fred_id": "THREEFY10",
        "description": "ACM Fitted 10-Year Yield (model-implied)",
        "frequency": "daily",
        "transform": "level",
    },
    # --- Market Expectations (forward-looking) ---
    "breakeven_10y": {
        "fred_id": "T10YIE",
        "description": "10-Year Breakeven Inflation Rate (market inflation expectations)",
        "frequency": "daily",
        "transform": "level",
    },
    # T13 Council 2026-05-31: forward-looking inflation features to fix
    # classifier "inflation blindness" (livello vs expectation discrepancy)
    "umich_inflation_1y": {
        "fred_id": "MICH",
        "description": "Univ. of Michigan: Inflation Expectation (1y, consumer survey)",
        "frequency": "monthly",
        "transform": "level",
    },
    "sticky_cpi_yoy": {
        "fred_id": "CORESTICKM158SFRBATL",
        "description": "Atlanta Fed Sticky-Price CPI (12m change, sticky inflation)",
        "frequency": "monthly",
        "transform": "level",
    },
    # T17 Council 2026-05-31 sessione 23: classifier accuracy features.
    # Attaccare la radice dell'alpha (council edge sono solo DD reducer).
    "wage_growth_atlanta": {
        "fred_id": "FRBATLWGTUMHWGO",
        "description": "Atlanta Fed Wage Growth Tracker (median 3m MA, overall)",
        "frequency": "monthly",
        "transform": "level",  # already YoY percentile
    },
    "jolts_quits_rate": {
        "fred_id": "JTSQUR",
        "description": "JOLTS Quits Rate (% labor force, labor market tightness)",
        "frequency": "monthly",
        "transform": "level",
    },
    "avg_weekly_hours": {
        "fred_id": "AWHAETP",
        "description": "Avg Weekly Hours (private nonfarm payrolls, early labor signal)",
        "frequency": "monthly",
        "transform": "level",
    },
    "hy_credit_spread": {
        "fred_id": "BAMLH0A0HYM2",
        "description": "ICE BofA US High Yield Index OAS spread (credit conditions)",
        "frequency": "daily",
        "transform": "level",
    },
    # T18 Council 2026-05-31 sessione 25: cyclical leading indicators.
    # Aggressive features predittive che precedono macro shifts (heavy truck = freight
    # demand collapse leading recession, building permits = housing cycle, consumer
    # credit = household balance sheet, durable goods = business investment).
    "heavy_truck_sales": {
        "fred_id": "HTRUCKSSAAR",
        "description": "Heavy Truck Sales SAAR (early cyclical: freight demand)",
        "frequency": "monthly",
        "transform": "yoy",
    },
    "building_permits": {
        "fred_id": "PERMIT",
        "description": "New Private Housing Units Authorized by Building Permits",
        "frequency": "monthly",
        "transform": "yoy",
    },
    "consumer_credit": {
        "fred_id": "TOTALSL",
        "description": "Total Consumer Credit Outstanding (household balance sheet)",
        "frequency": "monthly",
        "transform": "yoy",
    },
    # AUDIT 2026-07-12: rimossa definizione duplicata di durable_goods_orders
    # (ridefinita più sotto con transform "roc": l'ultima vince nel dict).
    # T19 Council 2026-05-31 sessione 26: NOWCAST features high-frequency macro.
    # Real-time GDP/CPI/wages estimates aggiornati daily/weekly. Lead 1-3 mesi su
    # BEA/BLS official prints. Council rated highest expected alpha (+0.8-1.5pp).
    "gdp_nowcast_atlanta": {
        "fred_id": "GDPNOW",
        "description": "Atlanta Fed GDPNow Real GDP Nowcast (current quarter, %SAAR)",
        "frequency": "weekly",
        "transform": "level",
    },
    "real_gdp_quarterly_nowcast": {
        "fred_id": "GDPC1",
        "description": "Real GDP Chained 2017$ (quarterly, fallback for nowcast deviation calc)",
        "frequency": "quarterly",
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
    "m2_yoy": {
        "fred_id": "M2SL",
        "description": "M2 Money Stock YoY (leading inflation indicator)",
        "frequency": "monthly",
        "transform": "yoy",
    },
    # --- T9: Banking & Credit ---
    "delinquency_rate_loans": {
        "fred_id": "DRBLACBS",
        "description": "Delinquency Rate on Business Loans (recession leading indicator)",
        "frequency": "quarterly",
        "transform": "level",
    },
    "bank_credit": {
        "fred_id": "TOTBKCR",
        "description": "Bank Credit, All Commercial Banks YoY",
        "frequency": "weekly",
        "transform": "yoy",
    },
    "commercial_industrial_loans": {
        "fred_id": "BUSLOANS",
        "description": "Commercial & Industrial Loans (corporate borrowing)",
        "frequency": "weekly",
        "transform": "yoy",
    },
    # --- T9: Treasury market positioning ---
    "treasury_volatility_move": {
        "fred_id": "VIXCLS",  # Fallback to VIX since MOVE not freely available on FRED
        "description": "VIX as bond volatility proxy (MOVE index not on FRED free)",
        "frequency": "daily",
        "transform": "level",
    },
    # --- T9: Wage growth (stagflation leading) ---
    "avg_hourly_earnings_yoy": {
        "fred_id": "CES0500000003",
        "description": "Average Hourly Earnings YoY (private nonfarm, wage inflation)",
        "frequency": "monthly",
        "transform": "yoy",
    },
    # --- T9: Valuation (Shiller CAPE proxy via S&P 500 + earnings) ---
    # AUDIT 2026-07-12: rimossa definizione duplicata di sp500 (ridefinita più
    # sotto, identica: l'ultima vince nel dict).
    # --- T9: Fed dot plot / forward guidance proxy ---
    "fed_funds_futures_1y": {
        "fred_id": "DGS1",
        "description": "1Y Treasury yield (forward expectation Fed funds in 1y, proxy)",
        "frequency": "daily",
        "transform": "level",
    },
    # --- T9: Global liquidity / global financial conditions ---
    "global_real_economic_activity_index": {
        "fred_id": "IGREA",
        "description": "Kilian Global Real Economic Activity Index (shipping rates proxy)",
        "frequency": "monthly",
        "transform": "level",
    },
    # --- T9: Yield curve detail (key recession lead) ---
    "yield_curve_3m10y": {
        "fred_id": "T10Y3M",
        "description": "10Y minus 3M Treasury spread (NY Fed recession model best fit)",
        "frequency": "daily",
        "transform": "level",
    },
    # --- T9-AUDIT P1: 5 feeds aggiunti post-council ---
    # AUDIT 2026-07-12: rimosse definizioni duplicate di fed_balance_sheet e
    # reverse_repo (ridefinite più sotto, identiche: l'ultima vince nel dict).
    "hy_oas_baml": {
        "fred_id": "BAMLH0A0HYM2",
        "description": "BAML US HY Master OAS — tail credit stress, complement to BAA",
        "frequency": "daily",
        "transform": "level",
    },
    # AUDIT 2026-07-12: rimosso anche wilshire_5000 — WILL5000IND è
    # DISCONTINUED su FRED (Bad Request a ogni fetch = log-noise quotidiano)
    # e la serie non era consumata da nessun modulo.
    "nfci_leverage": {
        "fred_id": "NFCILEVERAGE",
        "description": "Chicago Fed NFCI Leverage subindex — separates leverage from volatility",
        "frequency": "weekly",
        "transform": "level",
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
    # ========================================================================
    # Tier 3.6 roadmap Bridgewater — Alt data ingest (BLS, ECB, BoE, BoJ).
    # Estende coverage cross-country + indicatori coincidenti del Big Four
    # Conference Board (personal income aggiunto a industrial/payrolls/retail).
    # ========================================================================
    "job_openings": {
        "fred_id": "JTSJOL",
        "description": "JOLTS Job Openings (thousands, monthly, BLS) — labor demand",
        "frequency": "monthly",
        "transform": "level",
    },
    "personal_income": {
        "fred_id": "PI",
        "description": "Personal Income ($ billions, monthly) — Big Four Conf Board coincident",
        "frequency": "monthly",
        "transform": "roc",
    },
    "durable_goods_orders": {
        "fred_id": "DGORDER",
        "description": "Manufacturers New Orders Durable Goods ($ mil, monthly) — leading capex",
        "frequency": "monthly",
        "transform": "roc",
    },
    "ecb_main_refi_rate": {
        "fred_id": "ECBMRRFR",
        "description": "ECB Main Refinancing Operations Rate (%, daily) — eurozone policy",
        "frequency": "daily",
        "transform": "level",
    },
    "boe_bank_rate": {
        "fred_id": "IR3TIB01GBM156N",
        "description": "UK 3M Interbank Rate (%, monthly, OECD) — proxy BoE policy",
        "frequency": "monthly",
        "transform": "level",
    },
    "boj_policy_rate": {
        "fred_id": "IRSTCI01JPM156N",
        "description": "BoJ Immediate Rate / Call Rate (%, monthly) — Japan policy",
        "frequency": "monthly",
        "transform": "level",
    },
    # --- P1 2026-07-12 (gap analysis audit): liquidità completa + survey
    #     leading + estero. Tutte verificate live su FRED prima dell'aggiunta.
    "treasury_general_account": {
        "fred_id": "WTREGEN",
        "description": "Treasury General Account presso la Fed ($M, weekly) — 3° pezzo net liquidity = WALCL − RRP − TGA",
        "frequency": "weekly",
        "transform": "level",
    },
    "empire_fed_activity": {
        "fred_id": "GACDISA066MSFRBNY",
        "description": "Empire State Mfg Survey general activity (diffusion, SA) — leading, esce PRIMA dell'ISM",
        "frequency": "monthly",
        "transform": "level",
    },
    "philly_fed_activity": {
        "fred_id": "GACDFSA066MSFRBPHI",
        "description": "Philadelphia Fed Mfg Business Outlook general activity (diffusion, SA) — leading",
        "frequency": "monthly",
        "transform": "level",
    },
    "dallas_fed_activity": {
        "fred_id": "BACTSAMFRBDAL",
        "description": "Dallas Fed Mfg Outlook general business activity (diffusion, SA) — leading",
        "frequency": "monthly",
        "transform": "level",
    },
    "korea_exports": {
        "fred_id": "XTEXVA01KRM667S",
        "description": "Korea exports valore USD (monthly, OECD) — canary del trade mondiale, lead su global growth",
        "frequency": "monthly",
        "transform": "yoy",
    },
    # --- T20 P2 2026-07-12 (council): livello-vs-accelerazione ---
    "unit_labor_cost": {
        "fred_id": "ULCNFB",
        "description": "Unit Labor Cost nonfarm business (Q, index, dal 1947) — YoY = pressione salariale netta di produttività",
        "frequency": "quarterly",
        "transform": "yoy",
    },
    "productivity_nfb": {
        "fred_id": "OPHNFB",
        "description": "Nonfarm business output per hour (Q, index, dal 1947) — YoY = produttività (goldilocks quality)",
        "frequency": "quarterly",
        "transform": "yoy",
    },
}
