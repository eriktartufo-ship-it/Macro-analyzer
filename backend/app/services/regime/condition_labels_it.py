"""Etichette italiane delle condizioni del classificatore, per il sito.

Stanno FUORI da `REGIME_CONDITIONS` di proposito: `compute_classifier_hash()` (model_lock)
fa l'md5 di quel dizionario descrizioni comprese, quindi tradurle lì cambierebbe
l'impronta del modello senza che il modello sia cambiato.

Chiave = descrizione originale (la stessa condizione ha soglie diverse in regimi diversi:
la descrizione le distingue, il nome no). Se una descrizione cambia, l'etichetta torna
all'originale e `tests/test_condition_labels_it.py` diventa rosso.
"""
from __future__ import annotations

CONDITION_LABEL_IT: dict[str, str] = {
    # Reflazione
    "GDP ROC > 2%": "PIL in crescita oltre il 2%",
    "PMI > 50": "Indice dei direttori acquisti (PMI) sopra 50",
    "CPI YoY > 2.5%": "Inflazione a 12 mesi sopra il 2,5%",
    "Unemployment < 5% o ROC < 0": "Disoccupazione sotto il 5% o in calo",
    "Initial claims ROC < 0": "Richieste di sussidio in calo",
    "LEI ROC > 0": "Indicatori anticipatori (LEI) in salita",
    "10Y-2Y spread > 0.5": "Curva dei tassi 10a-2a sopra 0,5 punti",
    "Fed funds < 3%": "Tasso Fed sotto il 3%",
    "Nonfarm payrolls YoY > 1.5%": "Occupati in crescita oltre l'1,5% annuo",
    "Industrial production YoY > 2%": "Produzione industriale in crescita oltre il 2% annuo",
    "BAA-10Y spread < 2%": "Spread delle obbligazioni societarie (BAA) sotto il 2%",
    "Housing starts YoY > 3%": "Nuovi cantieri residenziali in crescita oltre il 3% annuo",
    "NFCI < -0.1 (credit easy)": "Condizioni finanziarie (NFCI) sotto −0,1: credito facile",
    "VIX < 18 (risk-on)": "Volatilità (VIX) sotto 18: mercati propensi al rischio",
    "ACM Term Premium 10Y < 0 (risk-on accepted)": "Premio a termine sul 10 anni sotto zero: rischio accettato",
    "DFM nowcast GDP YoY > 2.5% (real-time growth signal)": "Stima in tempo reale del PIL sopra il 2,5%",
    "SEC Form 4 insider score > 0.3 (smart money bullish)": "Insider che comprano le proprie azioni (punteggio sopra 0,3)",
    "Dedollar combined < 0.4 (dollar strong, risk-on global)": "De-dollarizzazione sotto 0,4: dollaro forte, rischio globale accettato",
    "News avg sentiment > 0.3 (positive narrative)": "Tono medio delle notizie sopra 0,3: racconto positivo",
    "Copper/gold ratio > 0.0028 ~p75 (cyclical strength)": "Rapporto rame/oro alto (sopra 0,0028): ciclo economico forte",
    "HY/IG spread ratio < 3 (credit risk-on)": "Spread high yield / investment grade sotto 3: credito propenso al rischio",
    # Stagflazione
    "CPI YoY > 4%": "Inflazione a 12 mesi sopra il 4%",
    "GDP ROC < 1.5%": "PIL in crescita sotto l'1,5%",
    "PMI < 50": "Indice dei direttori acquisti (PMI) sotto 50",
    "Unemployment ROC > 0.2": "Disoccupazione in salita (oltre 0,2)",
    "Fed funds > 4%": "Tasso Fed sopra il 4%",
    "LEI ROC < 0": "Indicatori anticipatori (LEI) in calo",
    "Initial claims ROC > 5%": "Richieste di sussidio in salita oltre il 5%",
    "10Y-2Y spread < 0.5": "Curva dei tassi 10a-2a sotto 0,5 punti",
    "Core PCE YoY > 3.5% (Fed-preferred)": "Inflazione PCE di fondo sopra il 3,5% (la misura preferita dalla Fed)",
    "BAA-10Y spread > 2.3%": "Spread delle obbligazioni societarie (BAA) sopra il 2,3%",
    "Consumer sentiment < 70": "Fiducia dei consumatori sotto 70",
    "Breakeven 10Y > 2.5% (inflation expect)": "Inflazione attesa a 10 anni sopra il 2,5%",
    "VIX > 25 (risk-off)": "Volatilità (VIX) sopra 25: fuga dal rischio",
    "Housing starts YoY < 0": "Nuovi cantieri residenziali in calo",
    "ACM Term Premium 10Y > 0.4% (duration risk priced)": "Premio a termine sul 10 anni sopra lo 0,4%: il rischio dei bond lunghi si paga",
    "DFM nowcast GDP YoY < 1.5% (real-time slowdown signal)": "Stima in tempo reale del PIL sotto l'1,5%: rallentamento",
    "SEC Form 4 insider score < -0.3 (smart money bearish)": "Insider che vendono le proprie azioni (punteggio sotto −0,3)",
    "Dedollar combined > 0.6 (USD debasement, real assets bid)": "De-dollarizzazione sopra 0,6: dollaro che si svaluta, beni reali richiesti",
    "Gold/oil ratio < 11.5 ~p25 (oil-driven inflation, stagflation 1973-style)": "Rapporto oro/petrolio sotto 11,5: inflazione da petrolio, come nel 1973",
    # Deflazione
    "GDP ROC < 1%": "PIL in crescita sotto l'1%",
    "CPI YoY < 2%": "Inflazione a 12 mesi sotto il 2%",
    "Initial claims ROC > 0": "Richieste di sussidio in salita",
    "Unemployment ROC > 0": "Disoccupazione in salita",
    "Claims spike + yield inversion": "Sussidi in impennata e curva dei tassi invertita",
    "Industrial production YoY < 0": "Produzione industriale in calo",
    "Nonfarm payrolls YoY < 0.8%": "Occupati in crescita sotto lo 0,8% annuo",
    "BAA-10Y spread > 2.5%": "Spread delle obbligazioni societarie (BAA) sopra il 2,5%",
    "VIX > 30 (panic)": "Volatilità (VIX) sopra 30: panico",
    "NFCI > 0.3 (credit stress)": "Condizioni finanziarie (NFCI) sopra 0,3: credito sotto stress",
    "Breakeven 10Y < 1.5%": "Inflazione attesa a 10 anni sotto l'1,5%",
    "News avg sentiment < -0.3 (panic narrative)": "Tono medio delle notizie sotto −0,3: racconto di panico",
    "Copper/gold ratio < 0.0020 ~p25 (cyclical collapse, deflation)": "Rapporto rame/oro basso (sotto 0,0020): ciclo economico che crolla",
    "Gold/oil ratio > 40 ~p90 (gold strong vs oil collapse, deflation)": "Rapporto oro/petrolio sopra 40: oro forte e petrolio crollato",
    "HY/IG spread ratio > 5 (credit stress 2008-style)": "Spread high yield / investment grade sopra 5: credito sotto stress come nel 2008",
    # Goldilocks
    "GDP ROC 1.5-3%": "PIL in crescita fra l'1,5% e il 3%",
    "CPI YoY < 2.5%": "Inflazione a 12 mesi sotto il 2,5%",
    "Unemployment < 4%": "Disoccupazione sotto il 4%",
    "PMI 52-57": "Indice dei direttori acquisti (PMI) fra 52 e 57",
    "10Y-2Y spread 0.5-2.0": "Curva dei tassi 10a-2a fra 0,5 e 2 punti",
    "Initial claims ROC < -3%": "Richieste di sussidio in calo oltre il 3%",
    "Fed funds 1-3%": "Tasso Fed fra l'1% e il 3%",
    "Core PCE YoY < 2.3%": "Inflazione PCE di fondo sotto il 2,3%",
    "BAA-10Y spread < 1.8%": "Spread delle obbligazioni societarie (BAA) sotto l'1,8%",
    "Consumer sentiment > 82": "Fiducia dei consumatori sopra 82",
    "VIX < 16 (tranquility)": "Volatilità (VIX) sotto 16: mercati tranquilli",
    "NFCI < -0.3": "Condizioni finanziarie (NFCI) sotto −0,3",
    "Breakeven 10Y 1.7-2.3%": "Inflazione attesa a 10 anni fra l'1,7% e il 2,3%",
    # Pilastri aggiuntivi (attivati da flag)
    "CPI YoY change 6m > 1%": "Inflazione salita di oltre 1 punto in 6 mesi",
    "Core PCE YoY change 6m > 0.8%": "Inflazione PCE di fondo salita di oltre 0,8 punti in 6 mesi",
    "GDP ROC change 6m < -0.5%": "Crescita del PIL scesa di oltre 0,5 punti in 6 mesi",
    "min(CPI YoY last 6m) > 2.5%": "Inflazione sempre sopra il 2,5% negli ultimi 6 mesi",
    "DFM nowcast change 3m < -0.4%": "Stima in tempo reale del PIL scesa di oltre 0,4 punti in 3 mesi",
    "UMich 1y inflation expectation > 3.5%": "Inflazione attesa dai consumatori a 1 anno sopra il 3,5%",
    "Atlanta Fed sticky CPI > 4% (hardest to break)": "Inflazione dei prezzi rigidi (Fed di Atlanta) sopra il 4%: la più difficile da piegare",
    "5y5y forward inflation > 2.5% (Fed anchor break)": "Inflazione attesa fra 5 e 10 anni sopra il 2,5%: l'ancora della Fed cede",
    "UMich 1y < 2% (consumer expects disinflation)": "Inflazione attesa dai consumatori a 1 anno sotto il 2%",
    "UMich 1y in 2.0-2.5% (anchored at target)": "Inflazione attesa dai consumatori a 1 anno fra il 2% e il 2,5%: ancorata all'obiettivo",
    "Sticky CPI in 2.0-2.7% (anchored sticky)": "Inflazione dei prezzi rigidi fra il 2% e il 2,7%: ancorata",
    "Atlanta wage tracker > 4.5% (wage-price spiral)": "Salari (Fed di Atlanta) in crescita oltre il 4,5%: rischio spirale salari-prezzi",
    "JOLTS quits > 2.5% (labor tight, wage power)": "Dimissioni volontarie sopra il 2,5%: lavoro scarso, lavoratori forti sui salari",
    "HY OAS < 3% (risk-on credit conditions)": "Spread high yield sotto il 3%: credito propenso al rischio",
    "Avg weekly hours YoY < 0 (early labor weakening)": "Ore lavorate a settimana in calo: primo segno di lavoro in frenata",
    "HY OAS > 5% (credit conditions crisis)": "Spread high yield sopra il 5%: credito in crisi",
    "JOLTS quits > 2.5% (healthy job market)": "Dimissioni volontarie sopra il 2,5%: mercato del lavoro sano",
    "Heavy truck sales YoY > +5% (freight demand robust)": "Vendite di camion pesanti in crescita oltre il 5%: trasporto merci robusto",
    "Building permits YoY > +5% (housing cycle expansion)": "Permessi di costruzione in crescita oltre il 5%: edilizia in espansione",
    "Durable goods orders YoY > +5% (business investment)": "Ordini di beni durevoli in crescita oltre il 5%: le imprese investono",
    "Consumer credit YoY > +5% (household leverage)": "Credito al consumo in crescita oltre il 5%: famiglie che si indebitano",
    "Heavy truck sales YoY < -10% (freight collapse, recession leading)": "Vendite di camion pesanti in calo oltre il 10%: merci ferme, anticipa la recessione",
    "Building permits YoY < -20% (housing crash)": "Permessi di costruzione in calo oltre il 20%: crollo dell'edilizia",
    "Durable goods YoY < -5% (capex collapse)": "Ordini di beni durevoli in calo oltre il 5%: investimenti che crollano",
    "Building permits YoY > +5% (healthy housing cycle)": "Permessi di costruzione in crescita oltre il 5%: edilizia sana",
    "Consumer credit YoY > +5% (healthy household demand)": "Credito al consumo in crescita oltre il 5%: domanda delle famiglie sana",
    "Atlanta GDPNow > 3% (real-time growth strong)": "Stima del PIL della Fed di Atlanta sopra il 3%: crescita forte",
    "Atlanta GDPNow < 1% (real-time growth weak)": "Stima del PIL della Fed di Atlanta sotto l'1%: crescita debole",
    "Atlanta GDPNow 1.5-3% (healthy moderate growth)": "Stima del PIL della Fed di Atlanta fra l'1,5% e il 3%: crescita sana",
    "Payrolls YoY < 0 (contrazione occupazione, non rallentamento)": "Occupati in calo: l'occupazione si contrae, non rallenta soltanto",
    "CPI YoY in CADUTA (chg 6m < p25) E claims in salita (composito)": "Inflazione che scende in fretta e sussidi in salita insieme",
    "Payrolls YoY > ~1.2 (mid p25-p50: crescita sana, anti-deflation)": "Occupati in crescita oltre l'1,2% circa: crescita sana",
    "ULC YoY < p50=2.3 (pressione salariale netta contenuta)": "Costo del lavoro per unità di prodotto sotto il 2,3%: pressione salariale contenuta",
    "Productivity YoY > p25=1.0 (peso piccolo: covid composition effect)": "Produttività in crescita oltre l'1%",
    "ULC YoY > p90=7 (wage-price spiral anni 70)": "Costo del lavoro per unità di prodotto sopra il 7%: spirale salari-prezzi come negli anni '70",
    "Petrolio a 12 mesi > p90 storico (+47%): shock d'offerta": "Petrolio salito oltre il 47% in 12 mesi: shock d'offerta",
}


def condition_label_it(description: str) -> str:
    """Etichetta italiana di una condizione; l'originale se manca (mai vuoto)."""
    return CONDITION_LABEL_IT.get(description, description)


def condition_description(regime: str, name: str) -> str | None:
    """Descrizione originale di una condizione, cercata anche fra quelle aggiunte dai pilastri
    (`_*_PILLAR_ADD`): `REGIME_CONDITIONS` da solo non le contiene, e il sito mostrava il nome
    interno (es. «inflation_persistent») al posto della frase."""
    from app.services.regime import classifier

    for attr, value in vars(classifier).items():
        if not isinstance(value, dict) or not (
            attr == "REGIME_CONDITIONS" or attr.endswith(("_PILLAR_ADD", "_PILLARS_ADD"))
        ):
            continue
        cfg = value.get(regime, {}).get(name)
        if isinstance(cfg, dict) and "description" in cfg:
            return cfg["description"]
    return None

