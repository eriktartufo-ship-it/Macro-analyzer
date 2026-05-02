"""Fetcher di FOMC statements e minutes dal Federal Reserve.

Sorgenti:
  - RSS feed monetario: https://www.federalreserve.gov/feeds/press_monetary.xml
    contiene tutti i press release (FOMC statements, minutes, speeches).
  - Filtra per quelli con titolo "Federal Reserve issues FOMC statement" o
    "Minutes of the Federal Open Market Committee".

Ritorna metadata + testo plain del documento (HTML stripped).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path

import feedparser
import requests
from loguru import logger


_RSS_URL = "https://www.federalreserve.gov/feeds/press_monetary.xml"
_CACHE_ROOT = Path(__file__).resolve().parents[3] / ".cache" / "fomc"
_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 ore: i statement nuovi escono raramente


@dataclass
class FOMCDocument:
    doc_type: str            # "statement" | "minutes" | "other"
    title: str
    published_date: date
    url: str
    text: str                # plain text estratto, max ~30k char
    fetched_at: datetime


def _classify_doc(title: str) -> str:
    t = title.lower()
    if "fomc statement" in t or "issues fomc" in t:
        return "statement"
    if "minutes" in t and "federal open market committee" in t:
        return "minutes"
    return "other"


def _strip_html(html: str) -> str:
    """Rimuove tag HTML, scripts, style. Restituisce testo plain."""
    # Rimuovi script/style
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Rimuovi tag
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode entities basilari
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
    text = text.replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"')
    # Whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_url(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "MacroAnalyzer/1.0"})
    r.raise_for_status()
    return r.text


def _extract_minutes_full_url(html: str) -> str | None:
    """Estrae dal HTML della press-release il link al documento minutes pieno.

    La press-release dei minutes Fed e' solo un avviso di pubblicazione di
    poche centinaia di parole, NON contiene le deliberations vere. Il
    documento pieno e' linkato come `/monetarypolicy/fomcminutesYYYYMMDD.htm`
    (o variante PDF). Pattern verificato su FOMC 2026-04-08 -> link a
    `/monetarypolicy/fomcminutes20260318.htm` (49k char vs 3k del press release).
    """
    # Match path relativo /monetarypolicy/fomcminutesYYYYMMDD[suffix].htm[#anchor]
    rel = re.search(
        r'href="(/monetarypolicy/fomcminutes\d{8}[^"]*\.htm[^"]*)"',
        html, flags=re.IGNORECASE,
    )
    if rel:
        return f"https://www.federalreserve.gov{rel.group(1)}"
    # Fallback: URL assoluto (raro)
    abs_match = re.search(
        r'href="(https?://[^"]*fomcminutes\d{8}[^"]*\.htm[^"]*)"',
        html, flags=re.IGNORECASE,
    )
    if abs_match:
        return abs_match.group(1)
    return None


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < _CACHE_TTL_SECONDS


def fetch_recent_fomc_documents(limit: int = 6) -> list[FOMCDocument]:
    """Pesca le N piu' recenti FOMC statements/minutes dal feed Fed.

    Cache: ogni documento singolarmente, by URL hash. RSS feed cachato 6h.
    """
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # Parse RSS
    feed = feedparser.parse(_RSS_URL)
    if not feed.entries:
        logger.warning("FOMC: RSS feed vuoto o irraggiungibile")
        return []

    out: list[FOMCDocument] = []
    for entry in feed.entries[:30]:  # esamina i primi 30, filtra dopo
        title = entry.get("title", "")
        doc_type = _classify_doc(title)
        if doc_type == "other":
            continue

        url = entry.get("link", "")
        if not url:
            continue

        # Parse data
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if published:
            pub_date = date(published.tm_year, published.tm_mon, published.tm_mday)
        else:
            pub_date = date.today()

        # Cache file per URL
        cache_key = re.sub(r"[^a-zA-Z0-9]", "_", url)[-80:]
        cache_path = _CACHE_ROOT / f"{cache_key}.txt"

        if _is_fresh(cache_path):
            try:
                text = cache_path.read_text(encoding="utf-8")
            except Exception:
                text = ""
        else:
            try:
                html = _fetch_url(url)
                # Per i minutes, la press-release e' solo un avviso di pubblicazione:
                # segui il link al documento minutes pieno se presente.
                if doc_type == "minutes":
                    full_url = _extract_minutes_full_url(html)
                    if full_url:
                        try:
                            full_html = _fetch_url(full_url)
                            text = _strip_html(full_html)[:30000]
                            logger.info(
                                f"FOMC: minutes full doc from {full_url} ({len(text)} chars)"
                            )
                        except Exception as e:
                            logger.warning(
                                f"FOMC: full minutes fetch failed for {full_url}, "
                                f"falling back to press-release: {e}"
                            )
                            text = _strip_html(html)[:30000]
                    else:
                        logger.warning(
                            f"FOMC: no full minutes link in press-release {url}, "
                            f"using press-release text"
                        )
                        text = _strip_html(html)[:30000]
                else:
                    text = _strip_html(html)[:30000]
                cache_path.write_text(text, encoding="utf-8")
                logger.info(f"FOMC: fetched {doc_type} {pub_date} ({len(text)} chars)")
            except Exception as e:
                logger.warning(f"FOMC: fetch failed for {url}: {e}")
                continue

        out.append(FOMCDocument(
            doc_type=doc_type,
            title=title,
            published_date=pub_date,
            url=url,
            text=text,
            fetched_at=datetime.now(),
        ))
        if len(out) >= limit:
            break

    out.sort(key=lambda d: d.published_date, reverse=True)
    return out
