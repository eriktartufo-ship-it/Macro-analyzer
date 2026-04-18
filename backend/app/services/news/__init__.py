def fetch_macro_news(*args, **kwargs):
    from app.services.news.rss_fetcher import fetch_macro_news as _fn
    return _fn(*args, **kwargs)


def score_news_batch(*args, **kwargs):
    from app.services.news.scorer import score_news_batch as _fn
    return _fn(*args, **kwargs)


def aggregate_signals(*args, **kwargs):
    from app.services.news.scorer import aggregate_signals as _fn
    return _fn(*args, **kwargs)


__all__ = ["fetch_macro_news", "score_news_batch", "aggregate_signals"]
