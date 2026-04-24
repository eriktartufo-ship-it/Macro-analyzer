"""Job scheduler per refresh giornaliero dei dati macro."""

from datetime import date

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from app.config import settings

scheduler = BackgroundScheduler()


def daily_refresh():
    """Job giornaliero: fetch dati FRED, classifica regime, calcola scores."""
    from app.services.dedollarization.scorer import (
        calculate_dedollarization,
        calculate_secular_bonus,
    )
    from app.services.indicators.fetcher import FredFetcher
    from app.services.regime.classifier import classify_regime
    from app.services.scoring.engine import calculate_final_scores

    logger.info("Avvio refresh giornaliero...")

    try:
        # 1. Fetch ultimi dati da FRED
        fetcher = FredFetcher()
        latest = fetcher.fetch_all_latest()
        logger.info(f"Fetched {len(latest)} indicatori da FRED")

        # 2. Prepara indicatori per il classifier
        indicators = _prepare_indicators(latest, fetcher)

        # 3. Classifica regime
        regime_result = classify_regime(indicators)
        logger.info(
            f"Regime: {regime_result['regime']} "
            f"(confidence: {regime_result['confidence']:.2f})"
        )

        # 4. Calcola dedollarizzazione e secular bonus
        dedollar_indicators, player_history = _prepare_dedollarization_indicators(latest, fetcher, indicators)
        dedollar_result = calculate_dedollarization(dedollar_indicators, player_history=player_history)
        secular_bonus = calculate_secular_bonus(dedollar_result["combined_score"])
        logger.info(
            f"Dedollarization: cyclical={dedollar_result['score']:.3f} "
            f"structural={dedollar_result['structural_score']:.3f} "
            f"accel={dedollar_result['acceleration']:+.3f} "
            f"combined={dedollar_result['combined_score']:.3f}"
        )

        # 4b. Explanation AI NON viene più generata durante il refresh.
        # L'utente la richiede on-demand via POST /api/v1/dedollarization/explanation.

        # 5. News scoring via Groq
        news_signals: dict[str, float] = {}
        scored_news: list[dict] = []
        headlines: list[dict] = []
        try:
            from app.services.news import fetch_macro_news, score_news_batch, aggregate_signals
            headlines = fetch_macro_news(max_age_days=3)
            if headlines:
                scored_news = score_news_batch(headlines)
                news_signals = aggregate_signals(scored_news)
                active_signals = {k: v for k, v in news_signals.items() if v != 0.0}
                logger.info(f"News scoring: {len(headlines)} headlines → {len(active_signals)} asset signals")
        except Exception as e:
            logger.warning(f"News scoring fallito (non bloccante): {e}")

        # 6. Calcola trajectory (dove stiamo andando)
        from app.services.regime.trajectory import calculate_trajectory

        avg_news_sentiment = 0.0
        if scored_news:
            relevant = [n for n in scored_news if n.get("relevance", 0) > 0.3]
            if relevant:
                avg_news_sentiment = sum(n["sentiment"] * n["relevance"] for n in relevant) / sum(n["relevance"] for n in relevant)

        trajectory = calculate_trajectory(
            current_probabilities=regime_result["probabilities"],
            indicators=indicators,
            news_sentiment=avg_news_sentiment,
            dedollar_score=dedollar_result.get("combined_score", 0.0),
            current_fit_scores=regime_result.get("fit_scores"),
        )
        logger.info(
            f"Trajectory: {trajectory['current_regime']} → {trajectory['projected_regime']} "
            f"(transition risk: {trajectory['transition_risk']:.0%})"
        )

        # 7. Calcola scores per tutti gli asset
        scores = calculate_final_scores(
            regime_result["probabilities"],
            secular_bonus=secular_bonus,
            news_signals=news_signals or None,
        )

        # 7b. Scores proiettati usando le probabilità del regime futuro
        projected_scores = calculate_final_scores(
            trajectory["projected_probabilities"],
            secular_bonus=secular_bonus,
            news_signals=news_signals or None,
        )
        trajectory["projected_scores"] = projected_scores
        logger.info(f"Scores calcolati per {len(scores)} asset classes")

        # 8. Salva risultati nel database
        _save_results(
            regime_result, scores, dedollar_result, secular_bonus,
            raw_indicators=indicators,
            raw_dedollar=dedollar_indicators,
            news_signals=news_signals,
            scored_news=scored_news,
            headlines=headlines,
            trajectory=trajectory,
        )

        # 9. Rolling window: mantieni solo gli ultimi 365 giorni
        try:
            from app.services.backfill import prune_old_records
            prune_old_records(days_to_keep=365)
        except Exception as e:
            logger.warning(f"Prune rolling window fallito (non bloccante): {e}")

        logger.info("Refresh giornaliero completato con successo")

    except Exception as e:
        logger.error(f"Errore durante il refresh giornaliero: {e}")
        raise


def _prepare_indicators(latest: dict[str, float], fetcher) -> dict[str, float]:
    """Prepara gli indicatori dal formato FRED al formato classifier.

    Calcola ROC e trasformazioni necessarie.
    """
    indicators = {}

    # GDP ROC (trimestrale, ultimo disponibile)
    if "real_gdp" in latest:
        try:
            gdp_data = fetcher.fetch_and_transform("real_gdp")
            roc = gdp_data.get("roc_3m")
            if roc is not None and not roc.empty:
                indicators["gdp_roc"] = float(roc.dropna().iloc[-1])
        except Exception:
            pass

    # PMI
    if "ism_manufacturing" in latest:
        indicators["pmi"] = latest["ism_manufacturing"]

    # CPI YoY
    if "cpi" in latest:
        try:
            cpi_data = fetcher.fetch_and_transform("cpi")
            roc_12m = cpi_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                indicators["cpi_yoy"] = float(roc_12m.dropna().iloc[-1])
        except Exception:
            pass

    # Unemployment
    if "unrate" in latest:
        indicators["unrate"] = latest["unrate"]
        try:
            unrate_data = fetcher.fetch_and_transform("unrate")
            roc = unrate_data.get("roc_3m")
            if roc is not None and not roc.empty:
                indicators["unrate_roc"] = float(roc.dropna().iloc[-1])
        except Exception:
            pass

    # Yield curve
    if "yield_curve_10y2y" in latest:
        indicators["yield_curve_10y2y"] = latest["yield_curve_10y2y"]

    # Initial claims ROC
    if "initial_claims" in latest:
        try:
            claims_data = fetcher.fetch_and_transform("initial_claims")
            roc = claims_data.get("roc_3m")
            if roc is not None and not roc.empty:
                indicators["initial_claims_roc"] = float(roc.dropna().iloc[-1])
        except Exception:
            pass

    # LEI ROC
    if "lei" in latest:
        try:
            lei_data = fetcher.fetch_and_transform("lei")
            roc = lei_data.get("roc_6m")
            if roc is not None and not roc.empty:
                indicators["lei_roc"] = float(roc.dropna().iloc[-1])
        except Exception:
            pass

    # Fed Funds
    if "fed_funds" in latest:
        indicators["fed_funds_rate"] = latest["fed_funds"]

    # --- Nuovi indicatori classifier (Core PCE, Payrolls, IP, BAA, Sentiment) ---
    if "core_pce" in latest:
        try:
            pce_data = fetcher.fetch_and_transform("core_pce")
            roc_12m = pce_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                indicators["core_pce_yoy"] = float(roc_12m.dropna().iloc[-1])
        except Exception:
            pass

    if "nonfarm_payrolls" in latest:
        try:
            pay_data = fetcher.fetch_and_transform("nonfarm_payrolls")
            roc_12m = pay_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                indicators["payrolls_roc_12m"] = float(roc_12m.dropna().iloc[-1])
        except Exception:
            pass

    if "industrial_production" in latest:
        try:
            ip_data = fetcher.fetch_and_transform("industrial_production")
            roc_12m = ip_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                indicators["indpro_roc_12m"] = float(roc_12m.dropna().iloc[-1])
        except Exception:
            pass

    if "baa_spread" in latest:
        indicators["baa_spread"] = latest["baa_spread"]

    if "consumer_sentiment" in latest:
        indicators["consumer_sentiment"] = latest["consumer_sentiment"]

    # Yield curve 10Y-3M: esposto come indicatore informativo (non ancora in REGIME_CONDITIONS
    # perché altamente correlato a 10y2y — evitiamo il doppio conteggio)
    if "yield_curve_10y3m" in latest:
        indicators["yield_curve_10y3m"] = latest["yield_curve_10y3m"]

    # --- Forward-looking market indicators ---
    # 10Y Breakeven inflation (expectation)
    if "breakeven_10y" in latest:
        indicators["breakeven_10y"] = latest["breakeven_10y"]
        try:
            be_data = fetcher.fetch_series("breakeven_10y")
            if be_data is not None and len(be_data) > 90:
                # Change vs 3 mesi fa (daily series: ~63 business days)
                change_3m = float(be_data.iloc[-1]) - float(be_data.iloc[-63])
                indicators["breakeven_10y_change_3m"] = change_3m
        except Exception:
            pass

    # VIX: livello + spike detection (level / 3-month mean)
    if "vix" in latest:
        indicators["vix"] = latest["vix"]
        try:
            vix_data = fetcher.fetch_series("vix")
            if vix_data is not None and len(vix_data) > 63:
                ma_3m = float(vix_data.iloc[-63:].mean())
                if ma_3m > 0:
                    indicators["vix_ma_ratio"] = float(vix_data.iloc[-1]) / ma_3m
        except Exception:
            pass

    # NFCI: livello (>0 tight, <0 loose) + trend
    if "nfci" in latest:
        indicators["nfci"] = latest["nfci"]
        try:
            nfci_data = fetcher.fetch_series("nfci")
            if nfci_data is not None and len(nfci_data) > 12:
                # Change vs 3 mesi (weekly: ~13 periods)
                indicators["nfci_change_3m"] = float(nfci_data.iloc[-1]) - float(nfci_data.iloc[-13])
        except Exception:
            pass

    # Housing starts YoY (monthly, 12 periods)
    try:
        hs_data = fetcher.fetch_and_transform("housing_starts")
        roc_12m = hs_data.get("roc_12m") if hs_data else None
        if roc_12m is not None and not roc_12m.empty:
            indicators["housing_starts_roc_12m"] = float(roc_12m.dropna().iloc[-1])
    except Exception:
        pass

    return indicators


def _prepare_dedollarization_indicators(
    latest: dict[str, float],
    fetcher,
    macro_indicators: dict[str, float],
) -> dict[str, float]:
    """Prepara gli indicatori per il calcolo della dedollarizzazione.

    Calcola sia dati ciclici (12m) che strutturali (5Y) e accelerazione.
    """
    dedollar: dict[str, float] = {}

    # === DXY (Dollar Index) ===
    if "dxy_broad" in latest:
        try:
            dxy_data = fetcher.fetch_and_transform("dxy_broad")
            raw = dxy_data.get("raw")
            roc_12m = dxy_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                dedollar["dxy_roc_12m"] = float(roc_12m.dropna().iloc[-1])
                # ROC 12m di 2 anni fa (per accelerazione)
                if len(roc_12m.dropna()) > 504:
                    dedollar["dxy_roc_12m_2y_ago"] = float(roc_12m.dropna().iloc[-504])
            if raw is not None:
                # ROC 5Y annualizzato (daily: ~1260 trading days)
                if len(raw) > 1260:
                    total_roc = (float(raw.iloc[-1]) / float(raw.iloc[-1260]) - 1) * 100
                    dedollar["dxy_roc_5y"] = total_roc / 5.0
                # ROC 10Y annualizzato (daily: ~2520 trading days)
                if len(raw) > 2520:
                    total_roc = (float(raw.iloc[-1]) / float(raw.iloc[-2520]) - 1) * 100
                    dedollar["dxy_roc_10y"] = total_roc / 10.0
        except Exception:
            pass

    # === Gold ===
    # Gold usa IQ12260 (Export Price Index, monthly) dopo dismissione LBMA da FRED
    if "gold_price" in latest:
        try:
            from app.services.indicators.fred_codes import FRED_SERIES
            freq = FRED_SERIES["gold_price"]["frequency"]
            gold_data = fetcher.fetch_and_transform("gold_price")
            raw = gold_data.get("raw")
            roc_12m = gold_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                dedollar["gold_roc_12m"] = float(roc_12m.dropna().iloc[-1])
                # 2 anni fa: 24 mesi (monthly) o 504 (daily)
                two_y_back = 24 if freq == "monthly" else 504
                if len(roc_12m.dropna()) > two_y_back:
                    dedollar["gold_roc_12m_2y_ago"] = float(roc_12m.dropna().iloc[-two_y_back])
            if raw is not None:
                # Indici periodici per 5Y/10Y/20Y in base alla frequenza
                idx_5y = 60 if freq == "monthly" else 1260
                idx_10y = 120 if freq == "monthly" else 2520
                idx_20y = 240 if freq == "monthly" else 5040
                if len(raw) > idx_5y:
                    total_roc = (float(raw.iloc[-1]) / float(raw.iloc[-idx_5y]) - 1) * 100
                    dedollar["gold_roc_5y"] = total_roc / 5.0
                if len(raw) > idx_10y:
                    total_roc = (float(raw.iloc[-1]) / float(raw.iloc[-idx_10y]) - 1) * 100
                    dedollar["gold_roc_10y"] = total_roc / 10.0
                if len(raw) > idx_20y:
                    total_roc = (float(raw.iloc[-1]) / float(raw.iloc[-idx_20y]) - 1) * 100
                    dedollar["gold_roc_20y"] = total_roc / 20.0
        except Exception:
            pass

    # Gold/Oil ratio
    if "gold_price" in latest and "oil_price" in latest:
        oil = latest["oil_price"]
        if oil > 0:
            dedollar["gold_oil_ratio"] = latest["gold_price"] / oil

    # === Debt/GDP (trimestrale) ===
    if "debt_gdp" in latest:
        dedollar["debt_gdp"] = latest["debt_gdp"]
        try:
            debt_data = fetcher.fetch_series("debt_gdp")
            if debt_data is not None:
                if len(debt_data) > 20:
                    dedollar["debt_gdp_5y_ago"] = float(debt_data.iloc[-20])
                if len(debt_data) > 40:
                    dedollar["debt_gdp_10y_ago"] = float(debt_data.iloc[-40])
        except Exception:
            pass

    # Real rate = Fed Funds - CPI YoY
    fed_funds = macro_indicators.get("fed_funds_rate")
    cpi_yoy = macro_indicators.get("cpi_yoy")
    if fed_funds is not None and cpi_yoy is not None:
        dedollar["real_rate"] = fed_funds - cpi_yoy

    # === M2 (mensile) ===
    if "m2" in latest:
        try:
            m2_data = fetcher.fetch_and_transform("m2")
            raw = m2_data.get("raw")
            roc_12m = m2_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                dedollar["m2_roc_12m"] = float(roc_12m.dropna().iloc[-1])
            if raw is not None:
                # ROC 5Y annualizzato (mensile: 60 mesi)
                if len(raw) > 60:
                    total_roc = (float(raw.iloc[-1]) / float(raw.iloc[-60]) - 1) * 100
                    dedollar["m2_roc_5y"] = total_roc / 5.0
                # ROC 10Y annualizzato (mensile: 120 mesi)
                if len(raw) > 120:
                    total_roc = (float(raw.iloc[-1]) / float(raw.iloc[-120]) - 1) * 100
                    dedollar["m2_roc_10y"] = total_roc / 10.0
                # ROC 20Y annualizzato (mensile: 240 mesi)
                if len(raw) > 240:
                    total_roc = (float(raw.iloc[-1]) / float(raw.iloc[-240]) - 1) * 100
                    dedollar["m2_roc_20y"] = total_roc / 20.0
        except Exception:
            pass

    # === 20Y DXY / Gold / Debt/GDP ===
    if "dxy_broad" in latest:
        try:
            raw = fetcher.fetch_series("dxy_broad")
            if raw is not None and len(raw) > 5040:  # 20 anni trading days
                total_roc = (float(raw.iloc[-1]) / float(raw.iloc[-5040]) - 1) * 100
                dedollar["dxy_roc_20y"] = total_roc / 20.0
        except Exception:
            pass
    if "debt_gdp" in latest:
        try:
            raw = fetcher.fetch_series("debt_gdp")
            if raw is not None and len(raw) > 80:  # 20 anni trimestrale
                dedollar["debt_gdp_20y_ago"] = float(raw.iloc[-80])
        except Exception:
            pass

    # === REAL BROAD DXY (inflation-adjusted USD, monthly) ===
    if "real_broad_dxy" in latest:
        try:
            rbd_data = fetcher.fetch_and_transform("real_broad_dxy")
            raw = rbd_data.get("raw")
            roc_12m = rbd_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                dedollar["real_broad_dxy_roc_12m"] = float(roc_12m.dropna().iloc[-1])
            if raw is not None:
                if len(raw) > 60:
                    total = (float(raw.iloc[-1]) / float(raw.iloc[-60]) - 1) * 100
                    dedollar["real_broad_dxy_roc_5y"] = total / 5.0
                if len(raw) > 120:
                    total = (float(raw.iloc[-1]) / float(raw.iloc[-120]) - 1) * 100
                    dedollar["real_broad_dxy_roc_10y"] = total / 10.0
                if len(raw) > 240:
                    total = (float(raw.iloc[-1]) / float(raw.iloc[-240]) - 1) * 100
                    dedollar["real_broad_dxy_roc_20y"] = total / 20.0
        except Exception:
            pass

    # === SILVER (NASDAQQSLVO, daily, price index) ===
    if "silver_price" in latest:
        try:
            sil_data = fetcher.fetch_and_transform("silver_price")
            raw = sil_data.get("raw")
            roc_12m = sil_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                dedollar["silver_roc_12m"] = float(roc_12m.dropna().iloc[-1])
            if raw is not None:
                # daily: 5Y=1260, 10Y=2520, 20Y=5040 trading days
                if len(raw) > 1260:
                    total = (float(raw.iloc[-1]) / float(raw.iloc[-1260]) - 1) * 100
                    dedollar["silver_roc_5y"] = total / 5.0
                if len(raw) > 2520:
                    total = (float(raw.iloc[-1]) / float(raw.iloc[-2520]) - 1) * 100
                    dedollar["silver_roc_10y"] = total / 10.0
                if len(raw) > 5040:
                    total = (float(raw.iloc[-1]) / float(raw.iloc[-5040]) - 1) * 100
                    dedollar["silver_roc_20y"] = total / 20.0
        except Exception:
            pass

    # === PLAYER SIGNALS ===
    _prepare_player_signals(latest, fetcher, dedollar)

    return dedollar, _compute_player_history(fetcher, latest)


def _prepare_player_signals(
    latest: dict[str, float],
    fetcher,
    dedollar: dict[str, float],
) -> None:
    """Calcola i segnali per-player e li inserisce nel dict dedollar."""
    # --- SYSTEM: Gold/SP500 ratio (gold proxy IQ12260 è index, SP500 è prezzo) ---
    # Normalizziamo entrambi come indici = 100 al primo valore comune, così il ratio
    # riflette la variazione relativa nel tempo (proxy del gold/sp500 classico).
    try:
        gold_raw = fetcher.fetch_series("gold_price") if "gold_price" in latest else None
        sp500_raw = fetcher.fetch_series("sp500") if "sp500" in latest else None
        if gold_raw is not None and sp500_raw is not None and len(gold_raw) > 0 and len(sp500_raw) > 0:
            # Resample entrambi a monthly end (SP500 è daily) per allineare
            gold_m = gold_raw.resample("ME").last() if hasattr(gold_raw, "resample") else gold_raw
            sp_m = sp500_raw.resample("ME").last() if hasattr(sp500_raw, "resample") else sp500_raw
            joined = gold_m.to_frame("gold").join(sp_m.to_frame("sp"), how="inner").dropna()
            if len(joined) > 0:
                # Normalizza a 100 al primo valore per avere ratio confrontabile nel tempo
                gold_norm = joined["gold"] / joined["gold"].iloc[0] * 100
                sp_norm = joined["sp"] / joined["sp"].iloc[0] * 100
                ratio = gold_norm / sp_norm
                dedollar["gold_sp500_ratio"] = float(ratio.iloc[-1])
                if len(ratio) > 12:
                    dedollar["gold_sp500_ratio_roc_12m"] = (float(ratio.iloc[-1]) / float(ratio.iloc[-12]) - 1) * 100
    except Exception:
        pass

    # --- SYSTEM: Copper/Gold ratio (entrambi monthly) ---
    # Copper è USD/MT (~9000), gold_price è un indice → normalizziamo.
    # Usiamo ratio normalizzato: rapporto delle variazioni relative dal 2000.
    try:
        if "copper_price" in latest and "gold_price" in latest:
            copper_raw = fetcher.fetch_series("copper_price")
            gold_raw = fetcher.fetch_series("gold_price")
            if copper_raw is not None and gold_raw is not None:
                joined = copper_raw.to_frame("cu").join(gold_raw.to_frame("au"), how="inner").dropna()
                if len(joined) > 0:
                    cu_norm = joined["cu"] / joined["cu"].iloc[0] * 100
                    au_norm = joined["au"] / joined["au"].iloc[0] * 100
                    ratio = cu_norm / au_norm * 2500  # riscalato al range "classico" copper/gold (~2500)
                    dedollar["copper_gold_ratio"] = float(ratio.iloc[-1])
    except Exception:
        pass

    # --- USA: 10Y Real Yield ---
    if "real_yield_10y" in latest:
        dedollar["real_yield_10y"] = latest["real_yield_10y"]

    # --- USA: Yield Curve ---
    if "yield_curve_10y2y" in latest:
        dedollar["yield_curve_10y2y"] = latest["yield_curve_10y2y"]

    # --- USA: Interest/Tax ratio ---
    try:
        if "interest_outlays" in latest:
            interest = latest["interest_outlays"]  # $B SAAR trimestrale
            if "tax_receipts" in latest and "gdp" in latest:
                tax_pct_gdp = latest["tax_receipts"]  # es. 17.0 (% di GDP)
                gdp_nominal = latest["gdp"]  # $B
                tax_dollars = tax_pct_gdp / 100.0 * gdp_nominal  # $B
                if tax_dollars > 0:
                    dedollar["interest_tax_ratio"] = (interest / tax_dollars) * 100  # in %
    except Exception:
        pass

    # --- USA: Foreign Treasury Holdings ROC 12m ---
    if "foreign_treasury_holdings" in latest:
        try:
            fth_data = fetcher.fetch_and_transform("foreign_treasury_holdings")
            roc_12m = fth_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                dedollar["foreign_treasury_roc_12m"] = float(roc_12m.dropna().iloc[-1])
        except Exception:
            pass

    # --- EUROPE: BTP-Bund Spread ---
    if "italy_10y" in latest and "germany_10y" in latest:
        dedollar["btp_bund_spread"] = latest["italy_10y"] - latest["germany_10y"]

    # --- EUROPE: EUR/CHF ---
    try:
        if "chf_per_usd" in latest and "usd_per_eur" in latest:
            # EUR/CHF = USD/EUR * CHF/USD → no, devo invertire
            # CHF/USD = DEXSZUS; USD/EUR = DEXUSEU
            # EUR/CHF (quanti CHF per 1 EUR) = (USD per EUR) * (CHF per USD)
            eur_chf = latest["usd_per_eur"] * latest["chf_per_usd"]
            dedollar["eur_chf"] = eur_chf
    except Exception:
        pass

    # --- JAPAN: 10Y JGB ---
    if "japan_10y" in latest:
        dedollar["japan_10y"] = latest["japan_10y"]

    # --- JAPAN: JPY appreciation (3m) ---
    if "jpy_per_usd" in latest:
        try:
            jpy_raw = fetcher.fetch_series("jpy_per_usd")
            if jpy_raw is not None and len(jpy_raw) > 63:
                # Se DEXJPUS scende → yen si rafforza. ROC negativo = yen forte.
                roc_3m = (float(jpy_raw.iloc[-1]) / float(jpy_raw.iloc[-63]) - 1) * 100
                dedollar["jpy_usd_roc_3m"] = roc_3m
        except Exception:
            pass

    # --- COMMODITY FX: CAD & AUD strength vs USD ---
    try:
        if "cad_per_usd" in latest:
            cad_raw = fetcher.fetch_series("cad_per_usd")
            if cad_raw is not None and len(cad_raw) > 252:
                # CAD/USD sale → CAD debole. Inverti: ROC negativo = CAD forte
                roc_12m = -(float(cad_raw.iloc[-1]) / float(cad_raw.iloc[-252]) - 1) * 100
                dedollar["cad_strength_roc_12m"] = roc_12m
        if "usd_per_aud" in latest:
            aud_raw = fetcher.fetch_series("usd_per_aud")
            if aud_raw is not None and len(aud_raw) > 252:
                # USD/AUD sale → AUD forte (USD per AUD = quanti dollari per 1 AUD)
                roc_12m = (float(aud_raw.iloc[-1]) / float(aud_raw.iloc[-252]) - 1) * 100
                dedollar["aud_strength_roc_12m"] = roc_12m
        cad = dedollar.get("cad_strength_roc_12m")
        aud = dedollar.get("aud_strength_roc_12m")
        vals = [v for v in (cad, aud) if v is not None]
        if vals:
            dedollar["commodity_fx_strength"] = sum(vals) / len(vals)
    except Exception:
        pass

    # --- EM: HY OAS ---
    if "em_hy_oas" in latest:
        dedollar["em_hy_oas"] = latest["em_hy_oas"]

    # --- DEFENSE: Defense / GDP ---
    try:
        if "defense_spending" in latest and "gdp" in latest:
            defense = latest["defense_spending"]  # $B annualized
            gdp_nominal = latest["gdp"]  # $B
            if gdp_nominal > 0:
                dedollar["defense_gdp_pct"] = (defense / gdp_nominal) * 100
    except Exception:
        pass

    # --- USA (estensione): Current Account % GDP ---
    # IEABC = millions USD quarterly; GDP = billions USD SAAR → CA/GDP_q_actual × 100
    try:
        if "current_account" in latest and "gdp" in latest and latest["gdp"] > 0:
            dedollar["current_account_gdp_pct"] = (
                float(latest["current_account"]) * 4.0 / (float(latest["gdp"]) * 1000.0) * 100
            )
    except Exception:
        pass

    # --- USA (estensione): NIIP % GDP (stock su GDP SAAR) ---
    try:
        if "niip" in latest and "gdp" in latest and latest["gdp"] > 0:
            dedollar["niip_gdp_pct"] = (
                float(latest["niip"]) / (float(latest["gdp"]) * 1000.0) * 100
            )
    except Exception:
        pass

    # --- FED LIQUIDITY: Fed holdings / GDP (% monetizzazione) ---
    # FDHBFRBN e GDP sono entrambi in $B → nessuna conversione.
    try:
        if "fed_debt_holdings" in latest and "gdp" in latest and latest["gdp"] > 0:
            dedollar["fed_debt_pct_gdp"] = (
                float(latest["fed_debt_holdings"]) / float(latest["gdp"]) * 100
            )
    except Exception:
        pass

    # --- FED LIQUIDITY: Fed balance sheet ROC 12m ---
    if "fed_balance_sheet" in latest:
        try:
            fb_data = fetcher.fetch_and_transform("fed_balance_sheet")
            roc_12m = fb_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                dedollar["fed_balance_roc_12m"] = float(roc_12m.dropna().iloc[-1])
        except Exception:
            pass

    # --- FED LIQUIDITY: Reverse repo (billions USD) ---
    if "reverse_repo" in latest:
        try:
            dedollar["reverse_repo_level_bn"] = float(latest["reverse_repo"])
        except Exception:
            pass

    # --- EXPECTATIONS: 5Y5Y Forward Breakeven (%) ---
    if "breakeven_5y5y" in latest:
        dedollar["breakeven_5y5y"] = float(latest["breakeven_5y5y"])

    # --- EXPECTATIONS: 10Y Term Premium ACM (%) ---
    if "term_premium_10y" in latest:
        dedollar["term_premium_10y"] = float(latest["term_premium_10y"])

    # --- BRICS+: CNY strength (inverse ROC 12m of CNY per USD) ---
    if "cny_per_usd" in latest:
        try:
            cny_raw = fetcher.fetch_series("cny_per_usd")
            if cny_raw is not None and len(cny_raw) > 252:
                # DEXCHUS in salita → CNY debole. Inverti per avere "strength"
                roc_12m = -(float(cny_raw.iloc[-1]) / float(cny_raw.iloc[-252]) - 1) * 100
                dedollar["cny_strength_roc_12m"] = roc_12m
        except Exception:
            pass

    # --- BRICS+: yields di India / Policy Rate Brasile ---
    if "india_10y" in latest:
        dedollar["india_10y"] = float(latest["india_10y"])
    if "brazil_policy_rate" in latest:
        dedollar["brazil_policy_rate"] = float(latest["brazil_policy_rate"])

    # --- METALS BREADTH: Gold / Silver performance differential (12m) ---
    # Non abbiamo spot prices (FRED non li espone); usiamo il differenziale dei ROC 12m
    # tra gold (IQ12260) e silver (NASDAQQSLVO). Positivo = gold outperforma (paura /
    # flight-to-quality); negativo = silver outperforma (ciclo industriale / mania).
    try:
        if "gold_price" in latest and "silver_price" in latest:
            gold_raw = fetcher.fetch_series("gold_price")
            sil_raw = fetcher.fetch_series("silver_price")
            if gold_raw is not None and sil_raw is not None and len(gold_raw) > 0 and len(sil_raw) > 0:
                gold_m = gold_raw.resample("ME").last()
                sil_m = sil_raw.resample("ME").last()
                joined = gold_m.to_frame("au").join(
                    sil_m.to_frame("ag"), how="inner"
                ).dropna()
                if len(joined) > 12:
                    au_roc = (float(joined["au"].iloc[-1]) / float(joined["au"].iloc[-13]) - 1) * 100
                    ag_roc = (float(joined["ag"].iloc[-1]) / float(joined["ag"].iloc[-13]) - 1) * 100
                    dedollar["gold_silver_perf_diff"] = au_roc - ag_roc
    except Exception:
        pass

    # --- EM: Dollar Index vs EM currencies ROC 12m (daily, dal 2006) ---
    # DTWEXEMEGS sale → USD forte su EM → stress → NO dedollar verso EM.
    # Lo score dedollar dovrà invertire: ROC negativo = EM forti = dedollar.
    if "em_fx_dollar_index" in latest:
        try:
            em_raw = fetcher.fetch_series("em_fx_dollar_index")
            if em_raw is not None and len(em_raw) > 252:
                roc_12m = (float(em_raw.iloc[-1]) / float(em_raw.iloc[-252]) - 1) * 100
                dedollar["em_fx_dollar_roc_12m"] = roc_12m
        except Exception:
            pass

    # --- EUROPE: ECB balance sheet ROC 12m (weekly) ---
    if "ecb_balance_sheet" in latest:
        try:
            ecb_data = fetcher.fetch_and_transform("ecb_balance_sheet")
            roc_12m = ecb_data.get("roc_12m")
            if roc_12m is not None and not roc_12m.empty:
                dedollar["ecb_balance_roc_12m"] = float(roc_12m.dropna().iloc[-1])
        except Exception:
            pass

    # --- EUROPE: EUR/USD ROC 12m (da DEXUSEU = usd_per_eur, daily) ---
    if "usd_per_eur" in latest:
        try:
            eur_raw = fetcher.fetch_series("usd_per_eur")
            if eur_raw is not None and len(eur_raw) > 252:
                clean = eur_raw.dropna()
                if len(clean) > 252:
                    roc_12m = (float(clean.iloc[-1]) / float(clean.iloc[-252]) - 1) * 100
                    dedollar["eur_usd_roc_12m"] = roc_12m
        except Exception:
            pass

    # --- EUROPE: OAT-Bund spread (Francia 10Y - Germania 10Y) ---
    if "france_10y" in latest and "germany_10y" in latest:
        try:
            dedollar["oat_bund_spread"] = float(latest["france_10y"]) - float(latest["germany_10y"])
        except Exception:
            pass


def _compute_player_history(fetcher, latest: dict[str, float]) -> dict[str, dict[str, float]]:
    """Calcola i segnali player a lookback storici (1Y, 5Y, 10Y, 20Y).

    Ogni serie viene fetchata una sola volta (cache nel fetcher) e poi
    si leggono i valori a indici passati.

    Returns:
        {horizon: {signal_key: value}} — es. {"1y": {"gold_sp500_ratio": 0.42, ...}}
    """
    LOOKBACKS = {
        "1y": {"daily": 252, "monthly": 12, "quarterly": 4},
        "5y": {"daily": 1260, "monthly": 60, "quarterly": 20},
        "10y": {"daily": 2520, "monthly": 120, "quarterly": 40},
        "20y": {"daily": 5040, "monthly": 240, "quarterly": 80},
    }
    history: dict[str, dict[str, float]] = {h: {} for h in LOOKBACKS}

    # --- Gold/SP500 ratio (monthly aligned, normalized) ---
    try:
        gold_raw = fetcher.fetch_series("gold_price") if "gold_price" in latest else None
        sp500_raw = fetcher.fetch_series("sp500") if "sp500" in latest else None
        if gold_raw is not None and sp500_raw is not None and len(gold_raw) > 0 and len(sp500_raw) > 0:
            gold_m = gold_raw.resample("ME").last()
            sp_m = sp500_raw.resample("ME").last()
            joined = gold_m.to_frame("gold").join(sp_m.to_frame("sp"), how="inner").dropna()
            if len(joined) > 0:
                gold_norm = joined["gold"] / joined["gold"].iloc[0] * 100
                sp_norm = joined["sp"] / joined["sp"].iloc[0] * 100
                ratio = gold_norm / sp_norm
                for h, lb in LOOKBACKS.items():
                    if len(ratio) > lb["monthly"]:
                        history[h]["gold_sp500_ratio"] = float(ratio.iloc[-lb["monthly"]])
    except Exception:
        pass

    # --- Copper/Gold ratio (monthly, normalized) ---
    try:
        copper_raw = fetcher.fetch_series("copper_price") if "copper_price" in latest else None
        gold_raw2 = fetcher.fetch_series("gold_price") if "gold_price" in latest else None
        if copper_raw is not None and gold_raw2 is not None:
            joined = copper_raw.to_frame("cu").join(gold_raw2.to_frame("au"), how="inner").dropna()
            if len(joined) > 0:
                cu_norm = joined["cu"] / joined["cu"].iloc[0] * 100
                au_norm = joined["au"] / joined["au"].iloc[0] * 100
                ratio = cu_norm / au_norm * 2500
                for h, lb in LOOKBACKS.items():
                    if len(ratio) > lb["monthly"]:
                        history[h]["copper_gold_ratio"] = float(ratio.iloc[-lb["monthly"]])
    except Exception:
        pass

    # --- M2 ROC 12m (monthly) ---
    try:
        m2_data = fetcher.fetch_and_transform("m2")
        roc_12m = m2_data.get("roc_12m")
        if roc_12m is not None:
            clean = roc_12m.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["monthly"]:
                    history[h]["m2_roc_12m"] = float(clean.iloc[-lb["monthly"]])
    except Exception:
        pass

    # --- Yield curve 10Y-2Y (daily level) ---
    try:
        yc_raw = fetcher.fetch_series("yield_curve_10y2y")
        if yc_raw is not None:
            clean = yc_raw.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["daily"]:
                    history[h]["yield_curve_10y2y"] = float(clean.iloc[-lb["daily"]])
    except Exception:
        pass

    # --- Real yield 10Y (daily level) ---
    try:
        ry_raw = fetcher.fetch_series("real_yield_10y")
        if ry_raw is not None:
            clean = ry_raw.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["daily"]:
                    history[h]["real_yield_10y"] = float(clean.iloc[-lb["daily"]])
    except Exception:
        pass

    # --- Interest / Tax ratio (quarterly) ---
    try:
        int_raw = fetcher.fetch_series("interest_outlays")
        tax_raw = fetcher.fetch_series("tax_receipts")
        gdp_raw = fetcher.fetch_series("gdp")
        if int_raw is not None and tax_raw is not None and gdp_raw is not None:
            tax_q = tax_raw.resample("QE").ffill()
            joined = int_raw.to_frame("interest").join(
                tax_q.to_frame("tax_pct"), how="inner"
            ).join(gdp_raw.to_frame("gdp"), how="inner").dropna()
            for h, lb in LOOKBACKS.items():
                if len(joined) > lb["quarterly"]:
                    interest = float(joined["interest"].iloc[-lb["quarterly"]])
                    tax_pct = float(joined["tax_pct"].iloc[-lb["quarterly"]])
                    gdp = float(joined["gdp"].iloc[-lb["quarterly"]])
                    tax_dollars = tax_pct / 100.0 * gdp
                    if tax_dollars > 0:
                        history[h]["interest_tax_ratio"] = (interest / tax_dollars) * 100
    except Exception:
        pass

    # --- Foreign Treasury Holdings ROC 12m (monthly) ---
    try:
        fth_data = fetcher.fetch_and_transform("foreign_treasury_holdings")
        roc_12m = fth_data.get("roc_12m")
        if roc_12m is not None:
            clean = roc_12m.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["monthly"]:
                    history[h]["foreign_treasury_roc_12m"] = float(clean.iloc[-lb["monthly"]])
    except Exception:
        pass

    # --- BTP-Bund spread (monthly) ---
    try:
        it_raw = fetcher.fetch_series("italy_10y")
        de_raw = fetcher.fetch_series("germany_10y")
        if it_raw is not None and de_raw is not None:
            joined = it_raw.to_frame("it").join(de_raw.to_frame("de"), how="inner").dropna()
            for h, lb in LOOKBACKS.items():
                if len(joined) > lb["monthly"]:
                    history[h]["btp_bund_spread"] = float(
                        joined["it"].iloc[-lb["monthly"]] - joined["de"].iloc[-lb["monthly"]]
                    )
    except Exception:
        pass

    # --- EUR/CHF (daily) ---
    try:
        eur_raw = fetcher.fetch_series("usd_per_eur")
        chf_raw = fetcher.fetch_series("chf_per_usd")
        if eur_raw is not None and chf_raw is not None:
            joined = eur_raw.to_frame("eur").join(chf_raw.to_frame("chf"), how="inner").dropna()
            for h, lb in LOOKBACKS.items():
                if len(joined) > lb["daily"]:
                    history[h]["eur_chf"] = float(
                        joined["eur"].iloc[-lb["daily"]] * joined["chf"].iloc[-lb["daily"]]
                    )
    except Exception:
        pass

    # --- Japan 10Y (monthly level) ---
    try:
        jp_raw = fetcher.fetch_series("japan_10y")
        if jp_raw is not None:
            clean = jp_raw.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["monthly"]:
                    history[h]["japan_10y"] = float(clean.iloc[-lb["monthly"]])
    except Exception:
        pass

    # --- JPY/USD ROC 3m (daily) ---
    try:
        jpy_raw = fetcher.fetch_series("jpy_per_usd")
        if jpy_raw is not None:
            clean = jpy_raw.dropna()
            for h, lb in LOOKBACKS.items():
                idx = lb["daily"]
                if len(clean) > idx + 63:
                    v_end = float(clean.iloc[-idx])
                    v_start = float(clean.iloc[-(idx + 63)])
                    if v_start > 0:
                        history[h]["jpy_usd_roc_3m"] = (v_end / v_start - 1) * 100
    except Exception:
        pass

    # --- Commodity FX strength: CAD + AUD 12m ROC (daily) ---
    try:
        cad_raw = fetcher.fetch_series("cad_per_usd") if "cad_per_usd" in latest else None
        aud_raw = fetcher.fetch_series("usd_per_aud") if "usd_per_aud" in latest else None
        for h, lb in LOOKBACKS.items():
            vals = []
            idx = lb["daily"]
            if cad_raw is not None and len(cad_raw) > idx + 252:
                cad_val = -(float(cad_raw.iloc[-idx]) / float(cad_raw.iloc[-(idx + 252)]) - 1) * 100
                vals.append(cad_val)
            if aud_raw is not None and len(aud_raw) > idx + 252:
                aud_val = (float(aud_raw.iloc[-idx]) / float(aud_raw.iloc[-(idx + 252)]) - 1) * 100
                vals.append(aud_val)
            if vals:
                history[h]["commodity_fx_strength"] = sum(vals) / len(vals)
    except Exception:
        pass

    # --- EM HY OAS (daily level) ---
    try:
        oas_raw = fetcher.fetch_series("em_hy_oas")
        if oas_raw is not None:
            clean = oas_raw.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["daily"]:
                    history[h]["em_hy_oas"] = float(clean.iloc[-lb["daily"]])
    except Exception:
        pass

    # --- Defense / GDP (quarterly) ---
    try:
        def_raw = fetcher.fetch_series("defense_spending")
        gdp_raw = fetcher.fetch_series("gdp")
        if def_raw is not None and gdp_raw is not None:
            joined = def_raw.to_frame("defense").join(gdp_raw.to_frame("gdp"), how="inner").dropna()
            for h, lb in LOOKBACKS.items():
                if len(joined) > lb["quarterly"]:
                    d = float(joined["defense"].iloc[-lb["quarterly"]])
                    g = float(joined["gdp"].iloc[-lb["quarterly"]])
                    if g > 0:
                        history[h]["defense_gdp_pct"] = (d / g) * 100
    except Exception:
        pass

    # --- Current Account / GDP (quarterly): IEABC millions vs GDP $B SAAR ---
    try:
        ca_raw = fetcher.fetch_series("current_account")
        gdp_raw = fetcher.fetch_series("gdp")
        if ca_raw is not None and gdp_raw is not None:
            joined = ca_raw.to_frame("ca").join(gdp_raw.to_frame("gdp"), how="inner").dropna()
            for h, lb in LOOKBACKS.items():
                if len(joined) > lb["quarterly"]:
                    ca = float(joined["ca"].iloc[-lb["quarterly"]])
                    g = float(joined["gdp"].iloc[-lb["quarterly"]])
                    if g > 0:
                        history[h]["current_account_gdp_pct"] = ca * 4.0 / (g * 1000.0) * 100
    except Exception:
        pass

    # --- NIIP / GDP (quarterly) ---
    try:
        niip_raw = fetcher.fetch_series("niip")
        gdp_raw = fetcher.fetch_series("gdp")
        if niip_raw is not None and gdp_raw is not None:
            joined = niip_raw.to_frame("niip").join(gdp_raw.to_frame("gdp"), how="inner").dropna()
            for h, lb in LOOKBACKS.items():
                if len(joined) > lb["quarterly"]:
                    n = float(joined["niip"].iloc[-lb["quarterly"]])
                    g = float(joined["gdp"].iloc[-lb["quarterly"]])
                    if g > 0:
                        history[h]["niip_gdp_pct"] = n / (g * 1000.0) * 100
    except Exception:
        pass

    # --- Fed holdings / GDP (quarterly) — entrambi in $B, no conversion ---
    try:
        fdh_raw = fetcher.fetch_series("fed_debt_holdings")
        gdp_raw = fetcher.fetch_series("gdp")
        if fdh_raw is not None and gdp_raw is not None:
            joined = fdh_raw.to_frame("f").join(gdp_raw.to_frame("gdp"), how="inner").dropna()
            for h, lb in LOOKBACKS.items():
                if len(joined) > lb["quarterly"]:
                    f = float(joined["f"].iloc[-lb["quarterly"]])
                    g = float(joined["gdp"].iloc[-lb["quarterly"]])
                    if g > 0:
                        history[h]["fed_debt_pct_gdp"] = f / g * 100
    except Exception:
        pass

    # --- Fed balance sheet ROC 12m (weekly series) ---
    try:
        fb_data = fetcher.fetch_and_transform("fed_balance_sheet")
        roc_12m = fb_data.get("roc_12m")
        if roc_12m is not None:
            clean = roc_12m.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["daily"]:
                    history[h]["fed_balance_roc_12m"] = float(clean.iloc[-lb["daily"]])
    except Exception:
        pass

    # --- Reverse Repo level ($bn, daily) — disponibile solo da ~2013 ---
    try:
        rr_raw = fetcher.fetch_series("reverse_repo")
        if rr_raw is not None:
            clean = rr_raw.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["daily"]:
                    history[h]["reverse_repo_level_bn"] = float(clean.iloc[-lb["daily"]])
    except Exception:
        pass

    # --- 5Y5Y Breakeven (daily, da 2003) ---
    try:
        be_raw = fetcher.fetch_series("breakeven_5y5y")
        if be_raw is not None:
            clean = be_raw.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["daily"]:
                    history[h]["breakeven_5y5y"] = float(clean.iloc[-lb["daily"]])
    except Exception:
        pass

    # --- 10Y Term Premium ACM (monthly) ---
    try:
        tp_raw = fetcher.fetch_series("term_premium_10y")
        if tp_raw is not None:
            clean = tp_raw.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["monthly"]:
                    history[h]["term_premium_10y"] = float(clean.iloc[-lb["monthly"]])
    except Exception:
        pass

    # --- CNY strength ROC 12m (inverted DEXCHUS, daily) ---
    try:
        cny_raw = fetcher.fetch_series("cny_per_usd")
        if cny_raw is not None:
            clean = cny_raw.dropna()
            for h, lb in LOOKBACKS.items():
                idx = lb["daily"]
                if len(clean) > idx + 252:
                    v_end = float(clean.iloc[-idx])
                    v_start = float(clean.iloc[-(idx + 252)])
                    if v_start > 0:
                        history[h]["cny_strength_roc_12m"] = -(v_end / v_start - 1) * 100
    except Exception:
        pass

    # --- India 10Y / Brazil Policy Rate (monthly level) ---
    for sig_name in ("india_10y", "brazil_policy_rate"):
        try:
            raw = fetcher.fetch_series(sig_name)
            if raw is not None:
                clean = raw.dropna()
                for h, lb in LOOKBACKS.items():
                    if len(clean) > lb["monthly"]:
                        history[h][sig_name] = float(clean.iloc[-lb["monthly"]])
        except Exception:
            continue

    # --- Silver ROC 12m (daily: NASDAQQSLVO) ---
    try:
        sil_data = fetcher.fetch_and_transform("silver_price")
        roc_12m = sil_data.get("roc_12m")
        if roc_12m is not None:
            clean = roc_12m.dropna()
            for h, lb in LOOKBACKS.items():
                if len(clean) > lb["daily"]:
                    history[h]["silver_roc_12m"] = float(clean.iloc[-lb["daily"]])
    except Exception:
        pass

    # --- Gold/Silver performance diff storico (12m rolling) ---
    # Allineiamo su ME, calcoliamo roc_12m separatamente, poi differenza.
    try:
        gold_raw = fetcher.fetch_series("gold_price") if "gold_price" in latest else None
        sil_raw = fetcher.fetch_series("silver_price") if "silver_price" in latest else None
        if gold_raw is not None and sil_raw is not None and len(gold_raw) > 0 and len(sil_raw) > 0:
            gold_m = gold_raw.resample("ME").last()
            sil_m = sil_raw.resample("ME").last()
            joined = gold_m.to_frame("au").join(sil_m.to_frame("ag"), how="inner").dropna()
            if len(joined) > 12:
                au_roc = (joined["au"] / joined["au"].shift(12) - 1) * 100
                ag_roc = (joined["ag"] / joined["ag"].shift(12) - 1) * 100
                diff = (au_roc - ag_roc).dropna()
                for h, lb in LOOKBACKS.items():
                    if len(diff) > lb["monthly"]:
                        history[h]["gold_silver_perf_diff"] = float(diff.iloc[-lb["monthly"]])
    except Exception:
        pass

    # --- EM FX dollar index ROC 12m (daily, dal 2006) ---
    try:
        em_raw = fetcher.fetch_series("em_fx_dollar_index")
        if em_raw is not None:
            clean = em_raw.dropna()
            for h, lb in LOOKBACKS.items():
                idx = lb["daily"]
                if len(clean) > idx + 252:
                    v_end = float(clean.iloc[-idx])
                    v_start = float(clean.iloc[-(idx + 252)])
                    if v_start > 0:
                        history[h]["em_fx_dollar_roc_12m"] = (v_end / v_start - 1) * 100
    except Exception:
        pass

    # --- ECB balance sheet ROC 12m (weekly) ---
    try:
        ecb_data = fetcher.fetch_and_transform("ecb_balance_sheet")
        roc_12m = ecb_data.get("roc_12m")
        if roc_12m is not None:
            clean = roc_12m.dropna()
            # weekly → ~52 periodi/anno
            for h, lb in LOOKBACKS.items():
                idx = lb["monthly"] * 52 // 12  # mesi → settimane approssimate
                if len(clean) > idx:
                    history[h]["ecb_balance_roc_12m"] = float(clean.iloc[-idx])
    except Exception:
        pass

    # --- EUR/USD ROC 12m (daily) ---
    try:
        eur_raw = fetcher.fetch_series("usd_per_eur")
        if eur_raw is not None:
            clean = eur_raw.dropna()
            for h, lb in LOOKBACKS.items():
                idx = lb["daily"]
                if len(clean) > idx + 252:
                    v_end = float(clean.iloc[-idx])
                    v_start = float(clean.iloc[-(idx + 252)])
                    if v_start > 0:
                        history[h]["eur_usd_roc_12m"] = (v_end / v_start - 1) * 100
    except Exception:
        pass

    # --- OAT-Bund spread (France 10Y - Germany 10Y, monthly) ---
    try:
        fr_raw = fetcher.fetch_series("france_10y")
        de_raw = fetcher.fetch_series("germany_10y")
        if fr_raw is not None and de_raw is not None:
            joined = fr_raw.to_frame("fr").join(de_raw.to_frame("de"), how="inner").dropna()
            for h, lb in LOOKBACKS.items():
                if len(joined) > lb["monthly"]:
                    history[h]["oat_bund_spread"] = float(
                        joined["fr"].iloc[-lb["monthly"]] - joined["de"].iloc[-lb["monthly"]]
                    )
    except Exception:
        pass

    # Rimuovi orizzonti vuoti
    return {h: signals for h, signals in history.items() if signals}


def _save_results(
    regime_result: dict,
    scores: dict[str, float],
    dedollar_result: dict | None = None,
    secular_bonus: dict[str, float] | None = None,
    raw_indicators: dict[str, float] | None = None,
    raw_dedollar: dict[str, float] | None = None,
    news_signals: dict[str, float] | None = None,
    scored_news: list[dict] | None = None,
    headlines: list[dict] | None = None,
    trajectory: dict | None = None,
):
    """Salva regime classification, daily signals e secular trend nel database."""
    import json

    from sqlalchemy.orm import Session

    from app.database import engine
    from app.models import DailySignal, RegimeClassification
    from app.models.secular_trends import SecularTrend

    today = date.today()

    # Aggregato sentiment news (weighted by relevance) per persistenza giornaliera:
    # oggi è solo informativo, in futuro potrà entrare nel classifier una volta
    # accumulato storico sufficiente.
    avg_news_sentiment = 0.0
    if scored_news:
        relevant = [n for n in scored_news if n.get("relevance", 0) > 0.3]
        if relevant:
            total_rel = sum(n["relevance"] for n in relevant)
            if total_rel > 0:
                avg_news_sentiment = sum(
                    n["sentiment"] * n["relevance"] for n in relevant
                ) / total_rel

    # Assembla il JSON di dettaglio con condizioni + indicatori + trajectory
    conditions_payload = {
        "conditions": regime_result["conditions_detail"],
        "indicators": raw_indicators or {},
        "dedollar_indicators": raw_dedollar or {},
        "trajectory": trajectory or {},
        "news_sentiment": avg_news_sentiment,
        "fit_scores": regime_result.get("fit_scores", {}),
    }

    with Session(engine) as session:
        # Upsert regime classification
        session.query(RegimeClassification).filter_by(date=today).delete()

        regime_record = RegimeClassification(
            date=today,
            regime=regime_result["regime"],
            probability_reflation=regime_result["probabilities"]["reflation"],
            probability_stagflation=regime_result["probabilities"]["stagflation"],
            probability_deflation=regime_result["probabilities"]["deflation"],
            probability_goldilocks=regime_result["probabilities"]["goldilocks"],
            confidence=regime_result["confidence"],
            conditions_met=json.dumps(conditions_payload),
        )
        session.add(regime_record)

        # Upsert daily signals
        session.query(DailySignal).filter_by(date=today).delete()
        _secular_bonus = secular_bonus or {}
        _news_signals = news_signals or {}

        for asset, score in scores.items():
            signal = DailySignal(
                date=today,
                asset_class=asset,
                final_score=score,
                regime_component=score - _secular_bonus.get(asset, 0.0) - _news_signals.get(asset, 0.0),
                secular_trend_bonus=_secular_bonus.get(asset, 0.0),
                news_signal=_news_signals.get(asset, 0.0),
                momentum_penalty=0.0,
                breakdown=json.dumps({
                    "regime": regime_result["regime"],
                    "probabilities": regime_result["probabilities"],
                }),
            )
            session.add(signal)

        # Salva notizie scored
        _headlines = headlines or []
        if scored_news:
            from app.models.news_signals import NewsSignal
            session.query(NewsSignal).filter_by(date=today).delete()
            for i, news in enumerate(scored_news):
                if news.get("relevance", 0) < 0.2:
                    continue
                source = _headlines[i]["source"] if i < len(_headlines) else "unknown"
                title = _headlines[i]["title"] if i < len(_headlines) else ""
                session.add(NewsSignal(
                    date=today,
                    source=source,
                    title=title,
                    content_summary=news.get("summary", ""),
                    sentiment_score=news.get("sentiment", 0.0),
                    relevance_score=news.get("relevance", 0.0),
                    affected_assets=json.dumps(news.get("affected_assets", {})),
                ))

        # Upsert secular trend (dedollarization)
        if dedollar_result:
            # Preserva la explanation esistente: il refresh non la rigenera, quindi
            # non deve cancellarla. Viene sostituita solo quando l'utente clicca
            # "Rigenera analisi AI" nell'apposita scheda.
            preserved_explanation: str | None = None
            existing = (
                session.query(SecularTrend)
                .filter_by(date=today, trend_name="dedollarization")
                .first()
            )
            if existing and existing.components:
                try:
                    existing_payload = json.loads(existing.components)
                    preserved_explanation = existing_payload.get("explanation")
                except Exception:
                    preserved_explanation = None
            # Se oggi non esiste ancora un record, prova a ereditare dall'ultimo disponibile
            if not preserved_explanation:
                prev = (
                    session.query(SecularTrend)
                    .filter(SecularTrend.trend_name == "dedollarization")
                    .filter(SecularTrend.date < today)
                    .order_by(SecularTrend.date.desc())
                    .first()
                )
                if prev and prev.components:
                    try:
                        prev_payload = json.loads(prev.components)
                        preserved_explanation = prev_payload.get("explanation")
                    except Exception:
                        preserved_explanation = None

            session.query(SecularTrend).filter_by(
                date=today, trend_name="dedollarization"
            ).delete()
            session.add(SecularTrend(
                date=today,
                trend_name="dedollarization",
                score=dedollar_result["combined_score"],
                components=json.dumps({
                    "components": dedollar_result["components"],
                    "structural": dedollar_result["structural"],
                    "decade": dedollar_result["decade"],
                    "twenty_year": dedollar_result.get("twenty_year", {}),
                    "by_player": dedollar_result.get("by_player", {}),
                    "player_history": dedollar_result.get("player_history", {}),
                    "player_acceleration": dedollar_result.get("player_acceleration", {}),
                    "structural_score": dedollar_result["structural_score"],
                    "decade_score": dedollar_result["decade_score"],
                    "twenty_year_score": dedollar_result.get("twenty_year_score"),
                    "acceleration": dedollar_result["acceleration"],
                    "combined_score": dedollar_result["combined_score"],
                    "geopolitical_score": dedollar_result.get("geopolitical_score", 0.0),
                    # Explanation: conserva quella esistente (o dell'ultimo record) se il
                    # refresh non ne ha generata una nuova.
                    "explanation": dedollar_result.get("explanation") or preserved_explanation,
                    # Raw indicators snapshot — serve per rigenerare la explanation on-demand
                    "raw_dedollar_indicators": raw_dedollar or {},
                }),
            ))

        session.commit()


def start_scheduler():
    """Avvia lo scheduler con job giornaliero."""
    scheduler.add_job(
        daily_refresh,
        "cron",
        hour=settings.scheduler_hour,
        minute=settings.scheduler_minute,
        id="daily_macro_refresh",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        f"Scheduler avviato: refresh giornaliero alle "
        f"{settings.scheduler_hour:02d}:{settings.scheduler_minute:02d} UTC"
    )


def stop_scheduler():
    """Ferma lo scheduler."""
    scheduler.shutdown()
    logger.info("Scheduler fermato")
