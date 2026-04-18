"""NewsAPI.org fetcher per notizie macro economiche.

Usa l'endpoint /v2/everything con query mirate alle aree macro (monetary policy,
inflation, labor, growth, rates, FX/dedollar) e restringe a domini di qualità.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from loguru import logger

from app.config import settings

NEWSAPI_URL = "https://newsapi.org/v2/everything"

MACRO_QUERIES = {
    "monetary_policy": '("Federal Reserve" OR FOMC OR "rate decision" OR "rate cut" OR "rate hike" OR ECB OR "Bank of Japan")',
    "inflation": '(CPI OR inflation OR PCE OR "core inflation" OR disinflation OR deflation)',
    "labor": '("unemployment rate" OR "jobs report" OR "nonfarm payrolls" OR "jobless claims" OR "labor market")',
    "growth": '(GDP OR recession OR "PMI" OR "ISM manufacturing" OR "economic growth" OR "business confidence")',
    "rates_markets": '("Treasury yields" OR "yield curve" OR "credit spreads" OR "bond market")',
    "fx_dedollar": '("US dollar" OR DXY OR "gold price" OR "dedollarization" OR "reserve currency")',
}

QUALITY_DOMAINS = ",".join([
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "economist.com",
    "marketwatch.com",
    "apnews.com",
    "barrons.com",
    "investing.com",
])


def _fetch_query(
    query: str,
    category: str,
    api_key: str,
    from_date: str,
    page_size: int = 10,
) -> list[dict[str, str]]:
    """Fa una singola chiamata NewsAPI /v2/everything."""
    try:
        resp = requests.get(
            NEWSAPI_URL,
            params={
                "q": query,
                "from": from_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "domains": QUALITY_DOMAINS,
                "apiKey": api_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            logger.warning(f"NewsAPI {category}: {data.get('message', 'unknown error')}")
            return []

        articles = data.get("articles", []) or []
        results: list[dict[str, str]] = []
        for art in articles:
            title = (art.get("title") or "").strip()
            url = (art.get("url") or "").strip()
            published = art.get("publishedAt")
            source_name = ((art.get("source") or {}).get("name") or "").strip()

            if not title or not url or not published:
                continue

            try:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                date_str = pub_dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                date_str = from_date

            results.append({
                "title": title,
                "url": url,
                "date": date_str,
                "source": f"newsapi_{category}",
                "source_name": source_name,
                "published_at": pub_dt if "pub_dt" in locals() else datetime.now(timezone.utc),
            })

        return results

    except requests.exceptions.RequestException as e:
        logger.warning(f"NewsAPI {category} request error: {e}")
        return []
    except (ValueError, KeyError) as e:
        logger.warning(f"NewsAPI {category} parsing error: {e}")
        return []


def fetch_newsapi_macro(max_age_days: int = 3) -> list[dict[str, Any]]:
    """Scarica notizie macro da NewsAPI lungo diverse query tematiche.

    Returns:
        Lista di {title, url, date, source, source_name} deduplicata per URL.
    """
    if not settings.newsapi_key:
        logger.info("NEWSAPI_KEY non configurata, skip NewsAPI fetch")
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).strftime("%Y-%m-%d")

    all_items: list[dict[str, Any]] = []
    for category, query in MACRO_QUERIES.items():
        items = _fetch_query(query, category, settings.newsapi_key, from_date, page_size=10)
        all_items.extend(items)
        logger.info(f"NewsAPI {category}: {len(items)} articoli")

    # Dedup per URL (mantiene la prima occorrenza)
    seen_urls: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in all_items:
        if item["url"] in seen_urls:
            continue
        seen_urls.add(item["url"])
        deduped.append(item)

    # Ordina per data (più recenti prima)
    deduped.sort(key=lambda x: x.get("published_at", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)

    # Rimuovi campo interno
    for item in deduped:
        item.pop("published_at", None)

    logger.info(f"NewsAPI totale dedup: {len(deduped)} articoli")
    return deduped
