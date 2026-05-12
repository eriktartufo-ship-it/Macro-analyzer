"""SEC EDGAR Form 4 insider trading aggregator.

Form 4 = report che ufficiali, direttori e holder >10% devono presentare
entro 2 giorni dalla transazione su azioni dell'azienda. È un segnale
"smart money": chi conosce il business meglio di tutti scommette i propri
soldi (open-market purchases) o esce (sales).

Pipeline:
1. `fetch_daily_index(date)` -> scarica `form.YYYYMMDD.idx` da SEC EDGAR
2. `parse_daily_index_text()` -> estrae righe Form 4 (CIK + filename)
3. `fetch_form4_xml(path)` -> scarica primary doc XML
4. `parse_form4_xml(text)` -> estrae transactions (code, shares, price)
5. `aggregate_insider_activity()` -> buy/sell ratio + net flow + score

**Transaction codes rilevanti** (SEC § 16):
- P: open-market PURCHASE (segnale bullish forte)
- S: open-market SALE (segnale bearish, ma cautela: sales spesso = 10b5-1
  scheduled o tax-related)
- A: GRANT da compensation (NON segnale, equity award)
- M: EXERCISE option (NON segnale, conversione)
- F: WITHHELD per tax (NON segnale)
- D: DISPOSITION acquired_disposed flag su shares
- Altri: ignorati (rumore)

**Aggregator**:
- buy_dollars = sum(shares * price) per code in {P}
- sell_dollars = sum(shares * price) per code in {S}
- buy_sell_ratio = buy/sell (None se sell=0)
- net_flow_usd = buy - sell
- insider_score = tanh(net_flow / scale) ∈ [-1, +1]

**SEC compliance**:
- User-Agent obbligatorio (settings.SEC_USER_AGENT)
- Rate limit 5 req/s (sleep 0.2s)
- Cache filing XML su disco (immutable)
"""
from __future__ import annotations

import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import requests
from loguru import logger


_USER_AGENT = "MacroAnalyzer/1.0 (eriktartufo@gmail.com)"
_BASE = "https://www.sec.gov"
_CACHE_ROOT = Path(__file__).resolve().parents[3] / ".cache" / "sec_form4"
_REQUEST_DELAY_S = 0.2  # 5 req/s, sotto il limite SEC di 10/s
_TIMEOUT = 30

# Codici considerati "market signal" (non compensation / non option mechanics)
_BUY_CODES = {"P"}     # open-market purchase
_SELL_CODES = {"S"}    # open-market sale
# Altri (M, A, F, G, ecc.) esclusi dall'aggregato

# Scaling per `tanh(net_flow / SCALE)` → score ∈ [-1, +1].
# Calibrazione: $100M net flow giornaliero su S&P 500 = ~tanh(1) ≈ 0.76.
# Stress storici (2008, 2020): insider buying spiked a ~$300M/day → score 0.99.
_SCORE_SCALE_USD = 100_000_000.0


# ---- Data classes -----------------------------------------------------------


@dataclass
class FilingMetadata:
    form_type: str       # "4"
    company: str
    cik: str             # zero-padded 10-digit
    filing_date: date
    filename: str        # es. "edgar/data/320193/0000320193-26-000045-index.htm"


@dataclass
class Form4Transaction:
    code: str                       # P / S / M / A / F / ...
    shares: float                   # positive
    price: float                    # positive (può essere 0 per A grants)
    acquired_disposed: str          # "A" or "D"
    ticker: str                     # issuer trading symbol, può essere ""
    transaction_date: date | None = None
    issuer_cik: str = ""

    def dollar_value(self) -> float:
        return self.shares * self.price


@dataclass
class InsiderActivityReport:
    period_start: date | None = None
    period_end: date | None = None
    n_transactions: int = 0
    n_buy_filings: int = 0
    n_sell_filings: int = 0
    total_buy_usd: float = 0.0
    total_sell_usd: float = 0.0
    buy_sell_ratio: float | None = None
    net_flow_usd: float = 0.0
    insider_score: float = 0.0
    top_buy_tickers: list[tuple[str, float]] = field(default_factory=list)
    top_sell_tickers: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---- Parsers ----------------------------------------------------------------


def parse_daily_index_text(text: str, filing_date: date) -> list[FilingMetadata]:
    """Parse del form.YYYYMMDD.idx (SEC daily index, formato fixed-width text).

    Struttura tipo:
        Form Type   Company Name   CIK   Date Filed   Filename
        --------------------------------
        4   APPLE INC   0000320193   2026-05-11   edgar/data/320193/.../index.htm
        ...

    Restituisce solo le righe con form_type == "4". Tolera columns con
    spazi interni nel company name.
    """
    out: list[FilingMetadata] = []
    in_data = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("-"):
            in_data = True
            continue
        if not in_data:
            continue
        # Token finale = filename (ha "edgar/" o "/"). Tokenizza da sx
        # per form_type, ricavando cik + date + filename dal suffisso.
        # Form type è il primo token. Filename è l'ultimo. CIK + date stanno
        # negli ultimi 3 token prima del filename.
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        form_type = parts[0]
        if form_type != "4":
            continue
        # Ultimi 3 token: CIK, Date Filed, Filename
        filename = parts[-1]
        date_str = parts[-2]
        cik = parts[-3]
        # Tutto il resto in mezzo = company name
        company = " ".join(parts[1:-3]).strip()
        # SEC daily index esiste con 2 formati data: "YYYY-MM-DD" (storico
        # 2003-2010) e "YYYYMMDD" (post-2010). Tollera entrambi.
        date_ok = False
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                datetime.strptime(date_str, fmt)
                date_ok = True
                break
            except ValueError:
                continue
        if not date_ok:
            continue
        out.append(FilingMetadata(
            form_type="4",
            company=company,
            cik=cik,
            filing_date=filing_date,
            filename=filename,
        ))
    return out


def parse_form4_xml(xml_text: str) -> list[Form4Transaction]:
    """Estrae transactions da Form 4 XML (schema X0306).

    Supporta sia nonDerivativeTable che derivativeTable. Per ora estrae solo
    nonDerivative (azioni dirette). Restituisce TUTTE le transactions, il
    filtering codici avviene in `aggregate_insider_activity`.
    """
    if not xml_text or "<" not in xml_text:
        return []
    try:
        # SEC XML può avere namespaces o no — proviamo entrambi
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    # issuer info
    ticker = _findtext(root, ".//issuer/issuerTradingSymbol", default="")
    issuer_cik = _findtext(root, ".//issuer/issuerCik", default="")

    out: list[Form4Transaction] = []
    for tx_node in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        code = _findtext(tx_node, ".//transactionCoding/transactionCode")
        shares_s = _findtext(tx_node, ".//transactionAmounts/transactionShares/value")
        price_s = _findtext(tx_node, ".//transactionAmounts/transactionPricePerShare/value")
        ad = _findtext(tx_node, ".//transactionAmounts/transactionAcquiredDisposedCode/value")
        tx_date_s = _findtext(tx_node, ".//transactionDate/value")

        if not code:
            continue
        try:
            shares = float(shares_s) if shares_s else 0.0
            price = float(price_s) if price_s else 0.0
        except ValueError:
            continue
        tx_date: date | None = None
        if tx_date_s:
            try:
                tx_date = datetime.strptime(tx_date_s, "%Y-%m-%d").date()
            except ValueError:
                tx_date = None

        out.append(Form4Transaction(
            code=code,
            shares=shares,
            price=price,
            acquired_disposed=ad,
            ticker=ticker,
            transaction_date=tx_date,
            issuer_cik=issuer_cik,
        ))
    return out


def _findtext(node: ET.Element, path: str, default: str = "") -> str:
    el = node.find(path)
    if el is None or el.text is None:
        return default
    return el.text.strip()


# ---- Aggregator -------------------------------------------------------------


def aggregate_insider_activity(
    transactions: Iterable[Form4Transaction],
    period_start: date | None = None,
    period_end: date | None = None,
) -> InsiderActivityReport:
    """Aggrega transactions in buy/sell USD + ratio + score normalizzato.

    Esclude codici non-market (M exercise, A grant, F tax withhold).
    Score = tanh(net_flow_usd / _SCORE_SCALE_USD) ∈ [-1, +1].
    """
    buy_total = 0.0
    sell_total = 0.0
    n_buy = 0
    n_sell = 0
    buy_by_ticker: dict[str, float] = {}
    sell_by_ticker: dict[str, float] = {}
    n_total = 0

    for tx in transactions:
        n_total += 1
        value = tx.dollar_value()
        if value <= 0:
            continue
        if tx.code in _BUY_CODES:
            buy_total += value
            n_buy += 1
            if tx.ticker:
                buy_by_ticker[tx.ticker] = buy_by_ticker.get(tx.ticker, 0.0) + value
        elif tx.code in _SELL_CODES:
            sell_total += value
            n_sell += 1
            if tx.ticker:
                sell_by_ticker[tx.ticker] = sell_by_ticker.get(tx.ticker, 0.0) + value

    ratio: float | None = None
    if sell_total > 0:
        ratio = buy_total / sell_total
    elif buy_total > 0:
        ratio = float("inf")  # only buys = bullish extreme

    # tanh-bounded score, scale calibrato per $100M net flow → ~0.76
    net = buy_total - sell_total
    score = math.tanh(net / _SCORE_SCALE_USD) if _SCORE_SCALE_USD > 0 else 0.0

    # Top tickers
    top_buy = sorted(buy_by_ticker.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_sell = sorted(sell_by_ticker.items(), key=lambda kv: kv[1], reverse=True)[:5]

    # Sostituisci infinito con None nel JSON-safe output
    safe_ratio: float | None
    if ratio is None:
        safe_ratio = None
    elif ratio == float("inf"):
        safe_ratio = None
    else:
        safe_ratio = ratio

    return InsiderActivityReport(
        period_start=period_start,
        period_end=period_end,
        n_transactions=n_total,
        n_buy_filings=n_buy,
        n_sell_filings=n_sell,
        total_buy_usd=buy_total,
        total_sell_usd=sell_total,
        buy_sell_ratio=safe_ratio,
        net_flow_usd=net,
        insider_score=score,
        top_buy_tickers=top_buy,
        top_sell_tickers=top_sell,
    )


# ---- HTTP fetchers (cached) -------------------------------------------------


def _ensure_cache() -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _CACHE_ROOT


def _get(url: str) -> str:
    headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip,deflate"}
    time.sleep(_REQUEST_DELAY_S)
    r = requests.get(url, headers=headers, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.text


def fetch_daily_index(filing_date: date) -> list[FilingMetadata]:
    """Fetch del form.YYYYMMDD.idx con cache permanente.

    SEC daily indexes esistono solo per business days. Per weekend/holiday
    ritorna lista vuota (404 silente).
    """
    cache = _ensure_cache()
    fname = f"form.{filing_date.strftime('%Y%m%d')}.idx"
    cache_file = cache / fname
    if cache_file.exists():
        text = cache_file.read_text(encoding="utf-8", errors="replace")
    else:
        qtr = (filing_date.month - 1) // 3 + 1
        url = f"{_BASE}/Archives/edgar/daily-index/{filing_date.year}/QTR{qtr}/{fname}"
        try:
            text = _get(url)
            cache_file.write_text(text, encoding="utf-8")
        except requests.HTTPError as exc:
            # SEC daily index manca per weekend/holiday: ritorna 403 (non 404).
            # Tratta entrambi come "no data" silente.
            status = exc.response.status_code if exc.response is not None else 0
            if status in (403, 404):
                logger.debug(f"SEC daily index {status} (likely weekend/holiday): {filing_date}")
                return []
            raise
    return parse_daily_index_text(text, filing_date=filing_date)


def fetch_form4_xml(index_filename: str) -> str:
    """Fetch del Form 4 ownership XML.

    `index_filename` dal daily index è il path SGML wrapper, es.
    `edgar/data/320193/0000320193-26-000045.txt`. Il file `.txt` è un SGML
    wrapper multi-document che contiene N `<DOCUMENT>` blocks. Form 4 dati
    sono dentro un blocco `<XML>...</XML>` (di solito il primo TYPE=4).

    Strategia: 1 fetch del .txt, estrai il primo <XML>...</XML> block,
    restituisci come stringa XML pulita. Cache permanente del solo blocco
    XML (filings immutabili).
    """
    cache = _ensure_cache()
    # accession from path: edgar/data/{cik}/{acc}.txt
    base = index_filename.replace(".txt", "")
    accession = base.split("/")[-1]
    cik = base.split("/")[-2]
    cache_file = cache / f"form4_{cik}_{accession}.xml"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")

    # filename dal daily index è `edgar/data/...`, URL completo
    # `https://www.sec.gov/Archives/edgar/data/...`
    url = f"{_BASE}/Archives/{index_filename}"
    try:
        sgml_text = _get(url)
    except requests.HTTPError as exc:
        logger.debug(f"Form 4 fetch failed: {exc}")
        return ""

    xml_block = _extract_xml_block(sgml_text)
    if xml_block:
        cache_file.write_text(xml_block, encoding="utf-8")
    return xml_block


def _extract_xml_block(sgml_text: str) -> str:
    """Estrae il primo blocco <XML>...</XML> dal SGML wrapper SEC.

    Form 4 .txt contiene tipicamente:
      <DOCUMENT>
      <TYPE>4
      <SEQUENCE>1
      <FILENAME>...
      <XML>
      <?xml version="1.0"?>
      <ownershipDocument>
        ...
      </ownershipDocument>
      </XML>
      </DOCUMENT>

    Restituisce solo il contenuto fra <XML> e </XML>.
    """
    m = re.search(r"<XML>\s*(.*?)\s*</XML>", sgml_text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    return m.group(1).strip()


# ---- High-level orchestrator ------------------------------------------------


def compute_insider_activity(
    days: int = 30,
    end_date: date | None = None,
    max_filings_per_day: int = 50,
) -> InsiderActivityReport:
    """Orchestratore: fetch + parse + aggregate per finestra rolling.

    Limita `max_filings_per_day` per evitare timeout (Form 4 può essere
    >200/giorno, ~10 min full fetch). Default 50 = sample sufficiente per
    signal aggregato.
    """
    end = end_date or date.today()
    transactions: list[Form4Transaction] = []
    notes: list[str] = []
    days_with_data = 0

    for offset in range(days):
        d = date.fromordinal(end.toordinal() - offset)
        # Skip weekends (SEC daily indexes solo lun-ven)
        if d.weekday() >= 5:
            continue
        try:
            entries = fetch_daily_index(d)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (403, 404):
                # Holiday / weekend / not yet published — silent skip
                continue
            notes.append(f"{d}: index fetch HTTP {status}: {exc}")
            continue
        except Exception as exc:
            notes.append(f"{d}: index fetch failed: {exc}")
            continue
        if not entries:
            continue
        days_with_data += 1
        for entry in entries[:max_filings_per_day]:
            try:
                xml = fetch_form4_xml(entry.filename)
                if xml:
                    transactions.extend(parse_form4_xml(xml))
            except Exception as exc:
                notes.append(f"{entry.cik}: form4 fetch failed: {exc}")
                continue

    report = aggregate_insider_activity(
        transactions,
        period_start=date.fromordinal(end.toordinal() - days + 1),
        period_end=end,
    )
    report.notes = notes
    if days_with_data == 0:
        report.notes.append("Nessun giorno con dati nel periodo richiesto.")
    return report
