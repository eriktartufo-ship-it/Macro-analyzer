"""RSS fetcher per notizie macro economiche.

Fonti gratuite (RSS/Atom):
  - Reuters Business/Economy
  - CNBC Economy
  - FT Markets (free headlines)
  - Bloomberg (via Google News RSS)
  - Federal Reserve Press Releases
"""

from datetime import datetime, timedelta, timezone
from time import mktime
from typing import Any

import feedparser
from loguru import logger

# Feed RSS gratuiti con focus macro/economia
RSS_FEEDS = {
    "reuters_markets": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "cnbc_economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "ecb_press": "https://www.ecb.europa.eu/rss/press.html",
    "ft_markets": "https://www.ft.com/markets?format=rss",
}


def parse_feed_entries(
    entries: list[Any],
    max_age_days: int = 3,
) -> list[dict[str, str]]:
    """Parsa entry di un feed RSS e filtra per età.

    Args:
        entries: Lista di feedparser entry objects
        max_age_days: Massima età delle notizie in giorni

    Returns:
        Lista di dict {title, url, date, source}
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    results = []

    for entry in entries:
        title = getattr(entry, "title", None)
        link = getattr(entry, "link", None)
        published = getattr(entry, "published_parsed", None)

        if not title or not link:
            continue

        # Converti published_parsed a datetime
        if published:
            try:
                pub_dt = datetime.fromtimestamp(mktime(published), tz=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                pub_dt = datetime.now(timezone.utc)
        else:
            pub_dt = datetime.now(timezone.utc)

        # Filtra notizie vecchie
        if pub_dt < cutoff:
            continue

        results.append({
            "title": title.strip(),
            "url": link.strip(),
            "date": pub_dt.strftime("%Y-%m-%d"),
            "published_at": pub_dt,
        })

    return results


def fetch_rss_macro(max_age_days: int = 3) -> list[dict[str, Any]]:
    """Scarica notizie macro da tutti i feed RSS configurati (solo RSS)."""
    all_news: list[dict[str, Any]] = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            entries = parse_feed_entries(feed.entries, max_age_days=max_age_days)
            for entry in entries:
                entry["source"] = source_name
            all_news.extend(entries)
            logger.info(f"RSS {source_name}: {len(entries)} notizie recenti")
        except Exception as e:
            logger.warning(f"Errore fetch RSS {source_name}: {e}")
            continue

    return all_news


def fetch_macro_news(max_age_days: int = 3) -> list[dict[str, str]]:
    """Scarica notizie macro da RSS + NewsAPI, deduplicate per URL.

    Returns:
        Lista di notizie {title, url, date, source}
    """
    # 1. RSS feeds
    rss_news = fetch_rss_macro(max_age_days=max_age_days)

    # 2. NewsAPI (best-effort)
    newsapi_items: list[dict[str, Any]] = []
    try:
        from app.services.news.newsapi_fetcher import fetch_newsapi_macro
        newsapi_items = fetch_newsapi_macro(max_age_days=max_age_days)
    except Exception as e:
        logger.warning(f"NewsAPI fetch fallito (non bloccante): {e}")

    # 3. Merge + dedup per URL
    all_news: list[dict[str, Any]] = rss_news + newsapi_items
    seen_urls: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in all_news:
        url = item.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(item)

    # Ordina per data (più recenti prima)
    deduped.sort(key=lambda x: x.get("published_at", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)

    # Rimuovi campo interno
    for item in deduped:
        item.pop("published_at", None)

    logger.info(f"Totale notizie macro (RSS+NewsAPI, dedup): {len(deduped)}")
    return deduped
