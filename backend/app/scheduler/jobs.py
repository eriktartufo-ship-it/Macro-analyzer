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

        # 3. Classifica regime. T6.1: estrai adaptive_recipes (popolato in
        # _prepare_indicators se flag USE_ADAPTIVE_THRESHOLDS=1) e passalo.
        adaptive_recipes = indicators.pop("__adaptive_recipes__", None)
        regime_result = classify_regime(indicators, adaptive_recipes=adaptive_recipes)
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

        # 4c. Tier 5.1 LOOP CLOSURE — Crisis modulation post-classifier.
        # Opt-in via USE_CRISIS_MODULATION. Se attivo, modula probs in base
        # a crisis_type + risk_score (deflation_crash → +defl, stagflation
        # _debasement → +stag, bubble_goldilocks → +defl/+stag soft).
        # Applicato DOPO dedollar (perché modulation legge dedollar_combined).
        try:
            from app.services.regime.crisis_modulation import apply_crisis_modulation

            modulated = apply_crisis_modulation(
                regime_result["probabilities"],
                indicators,
                dedollar_combined=dedollar_result.get("combined_score"),
            )
            if modulated != regime_result["probabilities"]:
                regime_result["probabilities"] = modulated
                # Aggiorna anche dominant regime se shift ha cambiato il top
                regime_result["regime"] = max(modulated, key=modulated.get)
                logger.info(
                    f"Crisis modulation applied: regime={regime_result['regime']} "
                    f"probs={ {k: round(v, 3) for k, v in modulated.items()} }"
                )
        except Exception as e:
            logger.warning(f"Crisis modulation skipped (non bloccante): {e}")

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

        # 7. Calcola scores per tutti gli asset.
        # Tier 5.4 LOOP CLOSURE: passa gdp_yoy_dfm per opt-in DFM asset bonus.
        dfm_value = indicators.get("gdp_yoy_dfm")
        scores = calculate_final_scores(
            regime_result["probabilities"],
            secular_bonus=secular_bonus,
            news_signals=news_signals or None,
            gdp_yoy_dfm=dfm_value,
        )

        # 7b. Scores proiettati usando le probabilità del regime futuro
        projected_scores = calculate_final_scores(
            trajectory["projected_probabilities"],
            secular_bonus=secular_bonus,
            news_signals=news_signals or None,
            gdp_yoy_dfm=dfm_value,
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

        # 10. CRITICAL #1 council 2026-05-27: log prediction + evaluate pending.
        # Senza questo step, il modello non sta vivendo. PredictionLog daily
        # accumula track record live; evaluate_log calcola realized returns
        # sulle predictions di 1/3/6 mesi fa.
        try:
            _log_and_evaluate_predictions(regime_result, scores, trajectory)
        except Exception as e:
            logger.warning(f"Prediction log + evaluate fallito (non bloccante): {e}")

        logger.info("Refresh giornaliero completato con successo")

    except Exception as e:
        logger.error(f"Errore durante il refresh giornaliero: {e}")
        raise


def _log_and_evaluate_predictions(
    regime_result: dict,
    scores: dict[str, float],
    trajectory: dict,
) -> None:
    """CRITICAL #1 council 2026-05-27: scheduler integration.

    Step daily:
    1. Log current prediction (top-5 weighted by score) -> PredictionLog
    2. Evaluate pending logs (1m/3m/6m horizons) -> realized returns vs benchmark

    Senza questo, il modello non sta accumulando track record live. Ogni
    prediction è automatically attribuita allo snapshot freezato corrente
    via `log_prediction()` (vedi prediction_logger.py).
    """
    from app.database import SessionLocal
    from app.services.validation.prediction_logger import (
        evaluate_log, log_prediction,
    )
    from app.services.backtest.real_returns_matrix import (
        build_real_returns_matrix, cumulative_returns_provider,
    )
    from app.services.scoring.engine import ASSET_REGIME_DATA

    # Paper trading TEST F validation 2026-05-27 (council 5/5):
    # TOP-7 EQUAL WEIGHT optimal config (Sharpe 1.28, α 60/40 +4.09pp CI [+2.08, +5.94]).
    # Equal cattura full edge dei picks classifier (risk_parity dilute alpha).
    # Top-7 sweet spot vs top-5 (+1.61pp alpha) e top-10 (-1.47pp).
    confidence = regime_result.get("confidence", 0.0)
    try:
        from app.services.config_flags import use_uncertainty_gate
        if use_uncertainty_gate() and confidence < 0.30:
            top5_assets = [
                {"asset": "us_equities_growth", "score": 60.0, "weight": 0.60},
                {"asset": "us_bonds_long", "score": 40.0, "weight": 0.40},
            ]
            allocation_mode = "uncertainty_fallback_60_40"
        else:
            # Top-7 equal weight (paper trading validated)
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:7]
            n = len(sorted_scores)
            top5_assets = [
                {"asset": a, "score": round(s, 2), "weight": round(1.0 / n, 4)}
                for a, s in sorted_scores
            ]
            allocation_mode = "top7_equal_weight"
    except Exception:
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])[:7]
        n = len(sorted_scores)
        top5_assets = [
            {"asset": a, "score": round(s, 2), "weight": round(1.0 / n, 4)}
            for a, s in sorted_scores
        ]
        allocation_mode = "top7_equal_weight"

    classify_output = {
        "regime": regime_result["regime"],
        "probabilities": regime_result["probabilities"],
        "confidence": regime_result.get("confidence", 0.0),
    }

    with SessionLocal() as db:
        try:
            logged = log_prediction(db, classify_output, top5_assets, allocation_mode=allocation_mode)
            logger.info(
                f"PredictionLog #{logged.id}: {logged.regime} ({logged.top5_count} assets, "
                f"mode={allocation_mode})"
            )
        except Exception as e:
            logger.warning(f"log_prediction failed: {e}")

        # Evaluate pending: serve real returns matrix per (start, end) -> cumulative
        try:
            asset_keys = list(ASSET_REGIME_DATA.keys())
            matrix = build_real_returns_matrix(asset_keys, use_cache=True)
            if matrix.empty:
                logger.warning("Real returns matrix empty, skip evaluate_log")
                return
            provider = cumulative_returns_provider(matrix)
            updated = evaluate_log(db, provider)
            if updated:
                logger.info(f"PredictionLog evaluated {updated} records (realized returns 1m/3m/6m)")
        except Exception as e:
            logger.warning(f"evaluate_log failed: {e}")


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

    # LEI level (CFNAIMA3): la serie e' gia' un z-score (range circa +/-3) che
    # rappresenta deviation dal trend di crescita. NON applicare ROC: lo z-score
    # oscillante attorno a 0 produce ROC sballati (es. da -0.2 a 0.1 -> +150%).
    # Il classifier usa la chiave `lei_roc` ma vuole il LIVELLO CFNAI come tale
    # (vedi backfill._build_indicators_as_of con last_before("lei")).
    if "lei" in latest:
        try:
            indicators["lei_roc"] = float(latest["lei"])
        except (TypeError, ValueError):
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

    # ACM Term Premium 10Y (Tier 1.3 roadmap Bridgewater): pillar regime.
    # TP > 0.5% = mercato pricing duration risk → +stagflation/+deflation.
    # TP < 0 = risk-on accettato → +reflation.
    if "term_premium_10y" in latest:
        indicators["term_premium_10y"] = float(latest["term_premium_10y"])

    # GDP YoY DFM nowcast (Tier 2.5 roadmap Bridgewater): risolve lag trimestrale
    # del classico gdp_roc QoQ. DFM su 5 indicatori coincidenti (industrial
    # production, payrolls, retail, sentiment, housing) → factor latente +
    # OLS mapping a GDP YoY. Aggiornato mensilmente, complementare a gdp_roc.
    # Best-effort: se fetch/fit fallisce, classifier usa default neutro 2.0%.
    try:
        from app.services.indicators.dfm_nowcast import compute_gdp_nowcast

        dfm_result = compute_gdp_nowcast(fetcher=fetcher, n_factors=1)
        if dfm_result is not None and dfm_result.latest_gdp_yoy_implied is not None:
            indicators["gdp_yoy_dfm"] = float(dfm_result.latest_gdp_yoy_implied)
    except Exception as e:
        logger.warning(f"DFM nowcast skipped in scheduler: {e}")

    # Tier 3.6 roadmap — Alt data ingest. Esposti come informativi nel
    # data-snapshot (non ancora pillar classifier, fase A solo dati).
    # JOLTS, personal income, durable goods, ECB/BoE/BoJ policy rates.
    for name in (
        "job_openings",          # BLS labor demand
        "personal_income",       # Big Four DFM coincident
        "durable_goods_orders",  # leading capex
        "ecb_main_refi_rate",    # eurozone policy
        "boe_bank_rate",         # UK policy (proxy 3M interbank)
        "boj_policy_rate",       # Japan policy
    ):
        try:
            raw = fetcher.fetch_series(name)
            if raw is not None and not raw.empty:
                indicators[name] = float(raw.iloc[-1])
        except Exception as e:
            logger.warning(f"Tier 3.6 alt data {name} fetch failed: {e}")

    # Tier 3.6 Fase B — SEC Form 4 insider score. Letto dall'ultimo record
    # persistito da `daily_insider_refresh` in SecularTrend. Default 0.0
    # (neutral) se non disponibile → classifier pillar è soft (peso 0.02-0.03).
    try:
        from app.database import SessionLocal
        from app.models import SecularTrend

        with SessionLocal() as session:
            latest = (
                session.query(SecularTrend)
                .filter(SecularTrend.trend_name == "insider_activity")
                .order_by(SecularTrend.date.desc())
                .first()
            )
            if latest is not None and latest.score is not None:
                indicators["insider_score"] = float(latest.score)
    except Exception as e:
        logger.warning(f"Insider score load skipped: {e}")

    # Tier 6.3 CROSS-ASSET SIGNALS — Compute 3 ratio se opt-in.
    # T9-AUDIT-BUG-FIX HIGH-4 (council): cross-asset ratios SEMPRE popolati
    # (era condizionale a USE_CROSS_ASSET_PILLARS, ma indicator JSON ne ha
    # bisogno anche quando flag OFF — copper_gold/gold_oil sono signal
    # informativi generali, non solo pillar classifier).
    # Pre-fix: gold_oil_ratio usava `gold_price` FRED (index) ÷ `oil_price`
    # → valore 1.71 sbagliato. copper_gold_ratio scalato ×2500 arbitrariamente.
    # Post-fix: usa Yahoo Finance futures GC=F/HG=F/CL=F sempre.
    try:
        from app.services.indicators.cross_asset import (
            compute_copper_gold_ratio,
            compute_gold_oil_ratio,
            compute_hy_ig_spread_ratio,
        )
        from app.services.prices.yahoo_fetcher import YahooFetcher

        yh = YahooFetcher()
        try:
            copper_s = yh.fetch("HG=F")  # $/lb
            gold_s = yh.fetch("GC=F")    # $/oz
            oil_s = yh.fetch("CL=F")     # $/barrel
            copper_p = float(copper_s.iloc[-1]) if copper_s is not None and not copper_s.empty else None
            gold_p = float(gold_s.iloc[-1]) if gold_s is not None and not gold_s.empty else None
            oil_p = float(oil_s.iloc[-1]) if oil_s is not None and not oil_s.empty else None
        except Exception:
            copper_p = gold_p = oil_p = None

        cg = compute_copper_gold_ratio(copper_p, gold_p)
        if cg is not None:
            indicators["copper_gold_ratio"] = cg
        go = compute_gold_oil_ratio(gold_p, oil_p)
        if go is not None:
            indicators["gold_oil_ratio"] = go
        # HY/IG spread ratio dai dati FRED già fetched
        hy = latest.get("hy_credit_spread")
        ig = latest.get("ig_credit_spread")
        hy_ig = compute_hy_ig_spread_ratio(hy, ig)
        if hy_ig is not None:
            indicators["hy_ig_spread_ratio"] = hy_ig

        # T9-AUDIT MED-5 (bughunter): hy_oas + energy_yoy dead paths.
        # hy_oas = High Yield OAS (BAML index BAMLH0A0HYM2 via FRED).
        # Stesso valore di hy_credit_spread se popolato; popola anche con
        # nome canonico atteso da financial_stress_veto.py.
        if hy is not None:
            indicators["hy_oas"] = hy

        # energy_yoy: variazione 12-mesi dell'energy index Yahoo (XLE o WTI).
        try:
            energy_s = yh.fetch("CL=F")  # WTI crude
            if energy_s is not None and len(energy_s) > 252:  # ~12m daily
                p_now = float(energy_s.iloc[-1])
                p_year_ago = float(energy_s.iloc[-252])
                if p_year_ago > 0:
                    indicators["energy_yoy"] = (p_now / p_year_ago - 1.0) * 100.0
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Cross-asset ratios skipped: {e}")

    # Tier 6.1 ADAPTIVE THRESHOLDS — Precompute recipes per il classifier
    # se flag USE_ADAPTIVE_THRESHOLDS=1. Pattern: carica serie storiche
    # dalle RegimeClassification + calcola quantili rolling 10y.
    # Output salvato in `indicators["__adaptive_recipes__"]` (key speciale
    # consumata dal scheduler, NON dal classifier direttamente).
    try:
        from app.services.config_flags import use_adaptive_thresholds
        if use_adaptive_thresholds():
            from app.database import SessionLocal
            from app.services.indicators.adaptive_thresholds import (
                _CONDITION_RECIPES,
                get_recipe_for_condition,
                list_adaptive_conditions,
                load_series_from_classifications,
            )

            with SessionLocal() as session:
                # Carica serie per gli indicator unici target
                target_keys = {v[0] for v in _CONDITION_RECIPES.values()}
                series_by_key = {
                    k: load_series_from_classifications(session, k)
                    for k in target_keys
                }
                recipes = {}
                for cond in list_adaptive_conditions():
                    ind_key, _ttype = _CONDITION_RECIPES[cond]
                    series = series_by_key.get(ind_key)
                    if series is None or series.empty:
                        continue
                    r = get_recipe_for_condition(cond, series, window_years=10)
                    if r is not None:
                        recipes[cond] = r
                indicators["__adaptive_recipes__"] = recipes
    except Exception as e:
        logger.warning(f"T6.1 adaptive recipes precompute skipped: {e}")

    # Tier 5.3 LOOP CLOSURE — News avg sentiment come pillar classifier
    # (opt-in via USE_NEWS_PILLAR). Letto dagli ultimi NewsSignal in DB,
    # pesato per relevance × decay_weight. Default 0.0 (neutral) se assente.
    try:
        from datetime import date, timedelta
        from app.database import SessionLocal
        from app.models import NewsSignal

        with SessionLocal() as session:
            cutoff = date.today() - timedelta(days=7)
            news = (
                session.query(NewsSignal)
                .filter(NewsSignal.date >= cutoff)
                .filter(NewsSignal.decay_weight >= 0.1)
                .all()
            )
            if news:
                # Weighted avg: sentiment × (relevance × decay)
                total_w = sum(n.relevance_score * n.decay_weight for n in news)
                if total_w > 0:
                    weighted = sum(
                        n.sentiment_score * n.relevance_score * n.decay_weight
                        for n in news
                    )
                    indicators["news_sentiment"] = float(weighted / total_w)
    except Exception as e:
        logger.warning(f"News sentiment pillar load skipped: {e}")

    # Tier 5.2 LOOP CLOSURE — Dedollar combined come pillar classifier
    # (opt-in via USE_DEDOLLAR_PILLAR). Letto dall'ultimo record SecularTrend
    # persistito dal daily_refresh principale. Default 0.5 (neutral) se assente.
    try:
        import json as _json
        from app.database import SessionLocal
        from app.models import SecularTrend

        with SessionLocal() as session:
            latest_ded = (
                session.query(SecularTrend)
                .filter(SecularTrend.trend_name == "dedollarization")
                .order_by(SecularTrend.date.desc())
                .first()
            )
            if latest_ded is not None:
                ded_combined = None
                if latest_ded.components:
                    try:
                        payload = _json.loads(latest_ded.components)
                        ded_combined = payload.get("combined_score")
                    except (ValueError, TypeError):
                        ded_combined = None
                if ded_combined is None and latest_ded.score is not None:
                    ded_combined = float(latest_ded.score)
                if ded_combined is not None:
                    indicators["dedollar_combined"] = float(ded_combined)
    except Exception as e:
        logger.warning(f"Dedollar pillar load skipped: {e}")

    # T11 (2026-05-27): momentum derivatives per pillar acceleration-based.
    # Riusa la helper di backfill per consistency (cpi_yoy_change_6m etc.).
    try:
        import pandas as _pd
        from app.services.regime.backfill import _enrich_with_momentum_derivatives

        series_for_momentum: dict[str, _pd.Series] = {}
        for name in ("cpi", "core_pce", "real_gdp"):
            try:
                s = fetcher.fetch_series(name)
                if s is not None and not s.empty:
                    series_for_momentum[name] = s
            except Exception:
                pass
        if series_for_momentum:
            cutoff = _pd.Timestamp.now()
            _enrich_with_momentum_derivatives(indicators, series_for_momentum, cutoff)
    except Exception as e:
        logger.warning(f"Momentum derivatives skipped: {e}")

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

    # Gold/Oil ratio via Yahoo futures
    # T9-AUDIT-BUG-FIX HIGH-4 (council): pre-fix usava FRED gold_price (index)
    # ÷ oil_price → valore 1.71 sbagliato. Ora usa Yahoo GC=F/CL=F.
    # Storicamente: ~17-20 in goldilocks, ~30+ in deflation (gold strong),
    # ~10-15 in stagflation (oil strong).
    try:
        from app.services.prices.yahoo_fetcher import YahooFetcher
        _yh = YahooFetcher()
        _au = _yh.fetch("GC=F")
        _oil = _yh.fetch("CL=F")
        if _au is not None and not _au.empty and _oil is not None and not _oil.empty:
            _au_p = float(_au.iloc[-1])
            _oil_p = float(_oil.iloc[-1])
            if _oil_p > 0:
                dedollar["gold_oil_ratio"] = _au_p / _oil_p
    except Exception:
        pass

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

    # --- SYSTEM: Copper/Gold ratio via Yahoo futures ---
    # T9-AUDIT-BUG-FIX HIGH-4 (council): rimosso scaling ×2500 arbitrario
    # su FRED gold_price (index, non $/oz). Ora usa Yahoo HG=F (copper $/lb)
    # e GC=F (gold $/oz) → ratio raw 0.001-0.002 (storicamente ~0.0017 in
    # contesto normale).
    try:
        from app.services.prices.yahoo_fetcher import YahooFetcher
        _yh = YahooFetcher()
        _cu = _yh.fetch("HG=F")
        _au = _yh.fetch("GC=F")
        if _cu is not None and not _cu.empty and _au is not None and not _au.empty:
            _cu_p = float(_cu.iloc[-1])
            _au_p = float(_au.iloc[-1])
            if _au_p > 0:
                dedollar["copper_gold_ratio"] = _cu_p / _au_p
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


def daily_insider_refresh():
    """Job notturno: fetch SEC Form 4 + persisti `insider_score` in SecularTrend.

    Chiama `compute_insider_activity(days=2)` per coprire ieri + 1 giorno
    cuscinetto (SEC pubblica indexes a fine giornata business). Cache disco
    è permanente: chiamate successive sui giorni cached sono no-op.

    Sample limitato a `max_filings_per_day=30` per limitare latenza (SEC
    rate-limit 5 req/s × 30 filings × 2 days = ~12s + parsing). Score
    aggregato resta significativo anche su sample piccolo.
    """
    from datetime import date, timedelta
    import json as _json
    from app.database import SessionLocal
    from app.models import SecularTrend
    from app.services.alt_data.sec_form4 import compute_insider_activity

    logger.info("Avvio daily_insider_refresh...")
    try:
        # End_date = ieri (oggi spesso ancora non pubblicato a fine giornata)
        yesterday = date.today() - timedelta(days=1)
        report = compute_insider_activity(
            days=2,
            end_date=yesterday,
            max_filings_per_day=30,
        )
        if report.n_transactions == 0:
            logger.warning("Insider refresh: 0 transactions fetched (likely cache empty + holiday)")
            return

        components = {
            "n_transactions": report.n_transactions,
            "n_buy_filings": report.n_buy_filings,
            "n_sell_filings": report.n_sell_filings,
            "total_buy_usd": report.total_buy_usd,
            "total_sell_usd": report.total_sell_usd,
            "buy_sell_ratio": report.buy_sell_ratio,
            "net_flow_usd": report.net_flow_usd,
            "insider_score": report.insider_score,
            "top_buy_tickers": [{"ticker": t, "usd": v} for t, v in report.top_buy_tickers],
            "top_sell_tickers": [{"ticker": t, "usd": v} for t, v in report.top_sell_tickers],
            "period_start": str(report.period_start) if report.period_start else None,
            "period_end": str(report.period_end) if report.period_end else None,
        }

        with SessionLocal() as session:
            # Upsert by (trend_name, date) — sostituisce same-day record
            existing = (
                session.query(SecularTrend)
                .filter(
                    SecularTrend.trend_name == "insider_activity",
                    SecularTrend.date == date.today(),
                )
                .first()
            )
            if existing is not None:
                existing.score = report.insider_score
                existing.components = _json.dumps(components)
            else:
                session.add(SecularTrend(
                    trend_name="insider_activity",
                    date=date.today(),
                    score=report.insider_score,
                    components=_json.dumps(components),
                ))
            session.commit()

        logger.info(
            f"Insider refresh OK: score={report.insider_score:+.3f} "
            f"buy=${report.total_buy_usd:,.0f} sell=${report.total_sell_usd:,.0f} "
            f"({report.n_transactions} txs)"
        )
    except Exception as e:
        logger.error(f"daily_insider_refresh failed: {e}")


def monthly_calibration_refresh():
    """Tier 6.6 — Job mensile: ricalcola live calibration `asset_regime_data`.

    Chiama `live_calibration.calibrate_and_persist(db)` che riusa la
    pipeline Bayesian shrinkage esistente + persiste snapshot versionato
    in DB. Best-effort: se calibration fallisce, no-op (la pipeline scoring
    cade su prior hardcoded automaticamente).
    """
    from app.database import SessionLocal
    from app.services.scoring.live_calibration import calibrate_and_persist

    logger.info("Avvio monthly_calibration_refresh...")
    try:
        with SessionLocal() as session:
            result = calibrate_and_persist(session)
            if result:
                logger.info(
                    f"Calibration refresh OK: version={result.version} "
                    f"n_classifications={result.n_classifications} "
                    f"n_assets={result.n_assets}"
                )
            else:
                logger.warning("Calibration refresh: no result (data insufficient?)")
    except Exception as e:
        logger.error(f"monthly_calibration_refresh failed: {e}")


def ai_portfolio_daily_scan():
    """Daily AI Portfolio scan + performance snapshot.

    Runs dopo daily_macro_refresh (30 minuti dopo) per usare gli ultimi
    DailySignal + RegimeClassification.
    """
    from app.database import SessionLocal
    from app.services.ai_portfolio import (
        strategist,
        data_strategist,
        performance as perf,
        llm_reasoner,
    )

    logger.info("AI portfolio HORSERACE daily scan starting...")
    with SessionLocal() as db:
        # 2 strategie indipendenti — capitale $100k cadauno
        for strategist_mod, label in (
            (strategist, "model_driven"),
            (data_strategist, "data_driven"),
        ):
            try:
                summary = strategist_mod.daily_scan(db)
                logger.info(f"AI portfolio scan {label} complete: {summary}")
                snap = perf.snapshot(db, strategy_type=label)
                if snap:
                    logger.info(
                        f"AI portfolio NAV {label}: ${snap.nav:.2f}, "
                        f"cum_ret: {snap.cumulative_return_pct*100:.2f}%, "
                        f"DD: {snap.drawdown_pct*100:.2f}%, n_pos: {snap.n_positions}"
                    )
                try:
                    reasoning = llm_reasoner.generate_reasoning(
                        db, strategy_type=label,
                    )
                    if reasoning:
                        logger.info(
                            f"AI portfolio LLM {label}: ai_agrees={reasoning.ai_agrees}, "
                            f"alt_regime={reasoning.ai_alternative_regime}"
                        )
                except Exception as e:
                    logger.warning(f"LLM reasoning {label} failed (non-fatal): {e}")
            except Exception as e:
                logger.exception(f"AI portfolio scan {label} failed: {e}")


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
    # AI Portfolio: 30 minuti dopo daily_refresh, usa i dati appena aggiornati
    scheduler.add_job(
        ai_portfolio_daily_scan,
        "cron",
        hour=settings.scheduler_hour,
        minute=(settings.scheduler_minute + 30) % 60,
        id="ai_portfolio_daily_scan",
        replace_existing=True,
    )
    # Insider refresh: 03:00 UTC, separato dal macro refresh per non
    # bloccarlo in caso di lentezza SEC. Cache disco è incrementale.
    scheduler.add_job(
        daily_insider_refresh,
        "cron",
        hour=3,
        minute=0,
        id="daily_insider_refresh",
        replace_existing=True,
    )
    # T6.6: live calibration refresh mensile (primo del mese, 04:00 UTC).
    scheduler.add_job(
        monthly_calibration_refresh,
        "cron",
        day=1,
        hour=4,
        minute=0,
        id="monthly_calibration_refresh",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        f"Scheduler avviato: macro refresh alle "
        f"{settings.scheduler_hour:02d}:{settings.scheduler_minute:02d} UTC, "
        f"insider refresh alle 03:00 UTC"
    )


def stop_scheduler():
    """Ferma lo scheduler."""
    scheduler.shutdown()
    logger.info("Scheduler fermato")
