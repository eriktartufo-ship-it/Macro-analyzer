"""Test TDD per il sistema di news scoring."""

from unittest.mock import patch, MagicMock


class TestNewsAPIFetcher:
    """Test fetch notizie da NewsAPI.org."""

    def test_no_key_returns_empty(self):
        from app.services.news.newsapi_fetcher import fetch_newsapi_macro

        with patch("app.services.news.newsapi_fetcher.settings") as mock_settings:
            mock_settings.newsapi_key = ""
            result = fetch_newsapi_macro()
            assert result == []

    def test_parses_article_payload(self):
        from app.services.news.newsapi_fetcher import _fetch_query

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Fed signals rate cut",
                    "url": "https://reuters.com/x",
                    "publishedAt": "2026-04-14T09:30:00Z",
                    "source": {"name": "Reuters"},
                },
                {
                    "title": "",
                    "url": "https://reuters.com/y",
                    "publishedAt": "2026-04-14T09:00:00Z",
                    "source": {"name": "Reuters"},
                },
            ],
        }

        with patch("app.services.news.newsapi_fetcher.requests.get", return_value=mock_resp):
            items = _fetch_query("test", "monetary_policy", "KEY", "2026-04-10")

        assert len(items) == 1
        assert items[0]["title"] == "Fed signals rate cut"
        assert items[0]["source"] == "newsapi_monetary_policy"
        assert items[0]["source_name"] == "Reuters"

    def test_dedupes_by_url(self):
        from app.services.news.newsapi_fetcher import fetch_newsapi_macro

        def fake_fetch_query(query, category, api_key, from_date, page_size=10):
            # Tutte le categorie ritornano lo stesso URL → dedup lo collassa a 1
            from datetime import datetime, timezone
            return [{
                "title": f"Article {category}",
                "url": "https://example.com/same",
                "date": "2026-04-14",
                "source": f"newsapi_{category}",
                "source_name": "Reuters",
                "published_at": datetime.now(timezone.utc),
            }]

        with patch("app.services.news.newsapi_fetcher.settings") as mock_settings, \
             patch("app.services.news.newsapi_fetcher._fetch_query", side_effect=fake_fetch_query):
            mock_settings.newsapi_key = "KEY"
            result = fetch_newsapi_macro()

        assert len(result) == 1

    def test_api_error_returns_empty(self):
        from app.services.news.newsapi_fetcher import _fetch_query
        import requests

        with patch(
            "app.services.news.newsapi_fetcher.requests.get",
            side_effect=requests.exceptions.ConnectionError("boom"),
        ):
            items = _fetch_query("q", "cat", "KEY", "2026-04-10")

        assert items == []


class TestRSSFetcher:
    """Test fetch notizie da RSS feeds."""

    def test_parse_rss_items(self):
        """Deve estrarre titolo, link, data da un feed RSS."""
        from app.services.news.rss_fetcher import parse_feed_entries

        # Simula un feed entry
        mock_entry = MagicMock()
        mock_entry.title = "Fed raises rates by 25bps"
        mock_entry.link = "https://example.com/article"
        mock_entry.published_parsed = (2026, 4, 12, 10, 0, 0, 0, 0, 0)

        entries = parse_feed_entries([mock_entry])
        assert len(entries) == 1
        assert entries[0]["title"] == "Fed raises rates by 25bps"
        assert entries[0]["url"] == "https://example.com/article"

    def test_empty_feed_returns_empty(self):
        """Feed vuoto deve ritornare lista vuota."""
        from app.services.news.rss_fetcher import parse_feed_entries

        entries = parse_feed_entries([])
        assert entries == []

    def test_filter_old_entries(self):
        """Deve filtrare notizie più vecchie di N giorni."""
        from app.services.news.rss_fetcher import parse_feed_entries
        from datetime import datetime, timedelta
        import time

        # Entry di oggi
        today = datetime.now()
        today_tuple = today.timetuple()

        # Entry di 30 giorni fa
        old = today - timedelta(days=30)
        old_tuple = old.timetuple()

        recent = MagicMock()
        recent.title = "Recent news"
        recent.link = "https://example.com/1"
        recent.published_parsed = today_tuple

        old_entry = MagicMock()
        old_entry.title = "Old news"
        old_entry.link = "https://example.com/2"
        old_entry.published_parsed = old_tuple

        entries = parse_feed_entries([recent, old_entry], max_age_days=7)
        assert len(entries) == 1
        assert entries[0]["title"] == "Recent news"


class TestNewsSentimentScorer:
    """Test scoring sentiment via LLM."""

    def test_score_structure(self):
        """Lo score deve contenere sentiment, relevance, affected_assets."""
        from app.services.news.scorer import parse_llm_response

        mock_response = {
            "sentiment": 0.6,
            "relevance": 0.8,
            "summary": "Fed hawkish, rates going up",
            "affected_assets": {
                "us_bonds_long": -0.7,
                "gold": 0.3,
                "us_equities_growth": -0.4,
            },
        }
        result = parse_llm_response(mock_response)
        assert -1.0 <= result["sentiment"] <= 1.0
        assert 0.0 <= result["relevance"] <= 1.0
        assert "affected_assets" in result
        assert isinstance(result["affected_assets"], dict)

    def test_clamp_sentiment_range(self):
        """Sentiment deve essere clamped a [-1, 1]."""
        from app.services.news.scorer import parse_llm_response

        result = parse_llm_response({
            "sentiment": 2.5,
            "relevance": 1.5,
            "summary": "test",
            "affected_assets": {},
        })
        assert result["sentiment"] == 1.0
        assert result["relevance"] == 1.0

    def test_invalid_response_returns_neutral(self):
        """Risposta invalida deve ritornare score neutro."""
        from app.services.news.scorer import parse_llm_response

        result = parse_llm_response(None)
        assert result["sentiment"] == 0.0
        assert result["relevance"] == 0.0

    def test_aggregate_news_signals(self):
        """Deve aggregare più news in un singolo segnale per asset."""
        from app.services.news.scorer import aggregate_signals

        scored_news = [
            {
                "sentiment": 0.8,
                "relevance": 0.9,
                "affected_assets": {"gold": 0.5, "us_bonds_long": -0.3},
            },
            {
                "sentiment": -0.3,
                "relevance": 0.6,
                "affected_assets": {"gold": -0.2, "us_equities_growth": 0.4},
            },
        ]

        signals = aggregate_signals(scored_news)
        assert "gold" in signals
        assert "us_bonds_long" in signals
        assert "us_equities_growth" in signals
        # Gold: (0.5*0.9 + -0.2*0.6) / (0.9+0.6) weighted avg
        assert -5.0 <= signals["gold"] <= 5.0

    def test_aggregate_empty_returns_zeros(self):
        """Nessuna news = segnali tutti a zero."""
        from app.services.news.scorer import aggregate_signals

        signals = aggregate_signals([])
        assert all(v == 0.0 for v in signals.values())

    def test_signal_range(self):
        """I segnali aggregati devono essere nel range [-5, +5]."""
        from app.services.news.scorer import aggregate_signals

        scored_news = [
            {
                "sentiment": 1.0,
                "relevance": 1.0,
                "affected_assets": {"gold": 1.0, "bitcoin": 1.0},
            },
        ] * 10

        signals = aggregate_signals(scored_news)
        for asset, val in signals.items():
            assert -5.0 <= val <= 5.0
