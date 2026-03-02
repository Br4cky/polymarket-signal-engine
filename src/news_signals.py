"""
News Signals — External confirmation layer using news APIs and Google Trends.
Finnhub free tier: 60 calls/min, general news + market news.
SerpApi Google Trends: 100 searches/month free.
"""

import logging
import time
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import requests

from src.utils import CacheManager, extract_keywords

logger = logging.getLogger(__name__)


class NewsSignals:
    """External signal detection via news and search trend APIs."""

    def __init__(self, config: dict, cache: CacheManager):
        self.config = config
        self.cache = cache
        self.finnhub_key = config.get('finnhub_api_key', '')
        self.finnhub_enabled = config.get('finnhub_enabled', False) and bool(self.finnhub_key)
        self.serpapi_key = config.get('serpapi_key', '')
        self.trends_enabled = config.get('google_trends_enabled', False) and bool(self.serpapi_key)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'PolySignalEngine/2.0'})

    # ── Finnhub News ──

    def fetch_general_news(self, category: str = 'general') -> List[dict]:
        """
        Fetch recent news from Finnhub.
        Categories: general, forex, crypto, merger
        """
        if not self.finnhub_enabled:
            return []

        cache_key = f'finnhub_news_{category}'
        cached = self.cache.get(cache_key, 600)  # 10 min cache
        if cached:
            return cached

        try:
            url = 'https://finnhub.io/api/v1/news'
            params = {
                'category': category,
                'token': self.finnhub_key
            }
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json()

            if not isinstance(articles, list):
                return []

            normalized = []
            for a in articles[:50]:  # Cap at 50 most recent
                normalized.append({
                    'headline': a.get('headline', ''),
                    'summary': a.get('summary', ''),
                    'source': a.get('source', ''),
                    'url': a.get('url', ''),
                    'datetime': a.get('datetime', 0),
                    'category': category,
                    'keywords': extract_keywords(
                        (a.get('headline', '') + ' ' + a.get('summary', ''))[:200]
                    )
                })

            self.cache.set(cache_key, normalized)
            logger.info(f"Fetched {len(normalized)} news articles ({category})")
            return normalized

        except Exception as e:
            logger.warning(f"Finnhub news fetch failed: {e}")
            return []

    def fetch_all_news(self) -> List[dict]:
        """Fetch news across all relevant categories."""
        all_articles = []
        for cat in ['general', 'crypto']:
            articles = self.fetch_general_news(cat)
            all_articles.extend(articles)
        return all_articles

    def score_news_relevance(self, market_question: str, articles: List[dict]) -> dict:
        """
        Score how relevant recent news is to a specific market.

        Returns: {
            news_score: float (0-8),
            matching_articles: int,
            most_relevant_headline: str,
            recency_hours: float  # how recent the best match is
        }
        """
        market_keywords = set(extract_keywords(market_question))
        if not market_keywords:
            return {'news_score': 0, 'matching_articles': 0,
                    'most_relevant_headline': '', 'recency_hours': 999}

        now = time.time()
        best_score = 0
        best_headline = ''
        best_recency = 999
        match_count = 0

        for article in articles:
            article_keywords = set(article.get('keywords', []))
            overlap = market_keywords & article_keywords
            overlap_ratio = len(overlap) / max(1, len(market_keywords))

            if overlap_ratio < 0.2:  # Need at least 20% keyword overlap
                continue

            # Recency factor: articles from last 6h score higher
            article_time = article.get('datetime', 0)
            hours_ago = (now - article_time) / 3600 if article_time > 0 else 999

            # Combined relevance = keyword overlap * recency decay
            recency_factor = max(0, 1.0 - (hours_ago / 72))  # Decays over 3 days
            relevance = overlap_ratio * recency_factor
            match_count += 1

            if relevance > best_score:
                best_score = relevance
                best_headline = article.get('headline', '')[:100]
                best_recency = hours_ago

        # Scale to 0-8 points
        news_score = min(8.0, best_score * 10.0)

        # Bonus for multiple matching articles (suggests trending topic)
        if match_count >= 3:
            news_score = min(8.0, news_score + 1.0)
        if match_count >= 5:
            news_score = min(8.0, news_score + 1.0)

        return {
            'news_score': round(news_score, 2),
            'matching_articles': match_count,
            'most_relevant_headline': best_headline,
            'recency_hours': round(best_recency, 1)
        }

    # ── Google Trends ──

    def fetch_google_trends(self, keyword: str) -> Optional[dict]:
        """
        Fetch Google Trends interest over time for a keyword.
        Uses SerpApi free tier.
        """
        if not self.trends_enabled:
            return None

        cache_key = f'gtrends_{keyword.lower().replace(" ", "_")}'
        cached = self.cache.get(cache_key, 3600)  # 1 hour cache
        if cached:
            return cached

        try:
            url = 'https://serpapi.com/search.json'
            params = {
                'engine': 'google_trends',
                'q': keyword,
                'data_type': 'TIMESERIES',
                'api_key': self.serpapi_key
            }
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # Extract interest over time
            timeline = data.get('interest_over_time', {}).get('timeline_data', [])
            if not timeline:
                return None

            values = []
            for point in timeline:
                vals = point.get('values', [])
                if vals:
                    values.append(int(vals[0].get('extracted_value', 0)))

            if not values:
                return None

            result = {
                'keyword': keyword,
                'current_interest': values[-1] if values else 0,
                'avg_interest': sum(values) / len(values) if values else 0,
                'max_interest': max(values) if values else 0,
                'trend_direction': 'up' if len(values) >= 2 and values[-1] > values[-2] else 'down',
                'data_points': len(values)
            }

            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Google Trends fetch failed for '{keyword}': {e}")
            return None

    def score_trends(self, market_question: str) -> dict:
        """
        Score Google Trends interest for a market topic.

        Returns: {
            trends_score: float (0-7),
            current_interest: int,
            spike_detected: bool
        }
        """
        if not self.trends_enabled:
            return {'trends_score': 0, 'current_interest': 0, 'spike_detected': False}

        # Extract the most meaningful keyword phrase (2-3 words)
        keywords = extract_keywords(market_question)
        if not keywords:
            return {'trends_score': 0, 'current_interest': 0, 'spike_detected': False}

        # Use top 2-3 keywords as search query
        query = ' '.join(keywords[:3])
        trends = self.fetch_google_trends(query)

        if not trends:
            return {'trends_score': 0, 'current_interest': 0, 'spike_detected': False}

        current = trends['current_interest']
        avg = trends['avg_interest']

        # Spike detection: current interest > 2x average
        spike = current > (avg * 2) if avg > 5 else False

        # Score: based on spike ratio
        if avg > 0:
            spike_ratio = current / avg
            score = min(7.0, (spike_ratio - 1.0) * 3.5)
        else:
            score = 0.0

        if score < 0:
            score = 0.0

        return {
            'trends_score': round(score, 2),
            'current_interest': current,
            'spike_detected': spike
        }

    # ── Combined External Score ──

    def compute_external_score(self, market_question: str, articles: List[dict] = None) -> dict:
        """
        Compute combined external confirmation score for a market.

        Returns: {
            news_score: float (0-8),
            trends_score: float (0-7),
            external_total: float (0-15),
            details: dict
        }
        """
        # News scoring
        if articles is None:
            articles = self.fetch_all_news()

        news_result = self.score_news_relevance(market_question, articles)
        trends_result = self.score_trends(market_question)

        total = news_result['news_score'] + trends_result['trends_score']

        return {
            'news_score': news_result['news_score'],
            'trends_score': trends_result['trends_score'],
            'external_total': round(min(15.0, total), 2),
            'details': {
                'matching_articles': news_result['matching_articles'],
                'most_relevant_headline': news_result['most_relevant_headline'],
                'recency_hours': news_result['recency_hours'],
                'current_interest': trends_result['current_interest'],
                'spike_detected': trends_result['spike_detected']
            }
        }
