"""
News Signals — External confirmation layer using free, keyless APIs.

Replaces paid Finnhub/SerpApi with:
  1. GDELT Project API — free global news search (no key, no documented rate limit)
  2. Wikipedia Pageviews API — free trend/spike detection (no key, 200 req/sec)

Both provide *targeted* signals: we search for the specific market topic rather
than keyword-matching against a generic news firehose.
"""

import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from urllib.parse import quote

import requests

from src.utils import CacheManager, extract_keywords

logger = logging.getLogger(__name__)


class NewsSignals:
    """External signal detection via GDELT news search and Wikipedia pageview spikes."""

    def __init__(self, config: dict, cache: CacheManager):
        self.config = config
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolySignalEngine/2.0 (https://github.com/Br4cky/polymarket-signal-engine)',
            'Accept': 'application/json',
        })

        # Feature flags — both default ON since they need no keys
        self.gdelt_enabled = config.get('gdelt_enabled', True)
        self.wiki_enabled = config.get('wiki_pageviews_enabled', True)

        # Tuning
        self.gdelt_max_records = config.get('gdelt_max_records', 75)
        self.gdelt_cache_ttl = config.get('gdelt_cache_ttl', 900)      # 15 min
        self.wiki_cache_ttl = config.get('wiki_cache_ttl', 3600)        # 1 hour
        self.wiki_lookback_days = config.get('wiki_lookback_days', 14)   # baseline window
        self.wiki_spike_threshold = config.get('wiki_spike_threshold', 3.0)  # 3x avg = spike

    # ─────────────────────────────────────────────────────────────────────
    #  GDELT News Search (replaces Finnhub)
    # ─────────────────────────────────────────────────────────────────────

    def _build_gdelt_query(self, market_question: str) -> str:
        """
        Build an effective GDELT search query from a market question.

        Strategy: extract the 3-5 most meaningful keywords, wrap multi-word
        entities in quotes where possible. GDELT supports boolean operators.
        """
        keywords = extract_keywords(market_question)
        if not keywords:
            return ''

        # Try to detect named entities (consecutive capitalised words in original)
        # For now, just use top keywords joined by space (GDELT does AND by default)
        # Cap at 5 terms to keep queries focused
        query_terms = keywords[:5]
        return ' '.join(query_terms)

    def fetch_gdelt_articles(self, query: str) -> List[dict]:
        """
        Search GDELT for recent news articles matching a query.

        GDELT DOC 2.0 API: completely free, no API key, returns articles
        from the last 3 months with title, URL, source, datetime.
        """
        if not self.gdelt_enabled or not query:
            return []

        cache_key = f'gdelt_{query.lower().replace(" ", "_")[:60]}'
        cached = self.cache.get(cache_key, self.gdelt_cache_ttl)
        if cached is not None:
            return cached

        try:
            url = 'https://api.gdeltproject.org/api/v2/doc/doc'
            params = {
                'query': query,
                'mode': 'artlist',
                'maxrecords': self.gdelt_max_records,
                'sort': 'datedesc',
                'format': 'json',
            }
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()

            data = resp.json()
            raw_articles = data.get('articles', [])

            articles = []
            for a in raw_articles:
                # Parse GDELT datetime format (YYYYMMDDHHmmSS or ISO)
                dt_raw = a.get('seendate', '')
                article_ts = self._parse_gdelt_datetime(dt_raw)

                articles.append({
                    'title': a.get('title', ''),
                    'url': a.get('url', ''),
                    'source': a.get('domain', a.get('source', '')),
                    'language': a.get('language', ''),
                    'datetime': article_ts,
                    'source_country': a.get('sourcecountry', ''),
                })

            self.cache.set(cache_key, articles)
            logger.info(f"GDELT: {len(articles)} articles for query '{query[:40]}'")
            return articles

        except Exception as e:
            logger.warning(f"GDELT fetch failed for '{query[:30]}': {e}")
            return []

    def _parse_gdelt_datetime(self, dt_str: str) -> float:
        """Parse GDELT datetime string to Unix timestamp."""
        if not dt_str:
            return 0
        try:
            # GDELT uses YYYYMMDDTHHmmSS or YYYYMMDDHHMMSS format
            clean = dt_str.replace('T', '').replace('-', '').replace(':', '').replace(' ', '')
            if len(clean) >= 14:
                dt = datetime(
                    int(clean[:4]), int(clean[4:6]), int(clean[6:8]),
                    int(clean[8:10]), int(clean[10:12]), int(clean[12:14]),
                    tzinfo=timezone.utc
                )
                return dt.timestamp()
        except (ValueError, IndexError):
            pass
        return 0

    def score_news_relevance(self, market_question: str, articles: List[dict] = None) -> dict:
        """
        Score how relevant recent GDELT news is to a specific market.

        Unlike Finnhub's category-based approach, GDELT articles are already
        topic-filtered by search query — so we focus on recency and volume
        rather than keyword overlap (which is baked into the search).

        Returns: {
            news_score: float (0-8),
            matching_articles: int,
            most_relevant_headline: str,
            recency_hours: float
        }
        """
        query = self._build_gdelt_query(market_question)
        if not query:
            return {'news_score': 0, 'matching_articles': 0,
                    'most_relevant_headline': '', 'recency_hours': 999}

        if articles is None:
            articles = self.fetch_gdelt_articles(query)

        if not articles:
            return {'news_score': 0, 'matching_articles': 0,
                    'most_relevant_headline': '', 'recency_hours': 999}

        now = time.time()
        best_headline = ''
        best_recency = 999

        # Count articles by recency bucket
        recent_6h = 0
        recent_24h = 0
        recent_72h = 0

        for article in articles:
            article_ts = article.get('datetime', 0)
            if article_ts <= 0:
                continue

            hours_ago = (now - article_ts) / 3600

            if hours_ago < best_recency:
                best_recency = hours_ago
                best_headline = article.get('title', '')[:120]

            if hours_ago <= 6:
                recent_6h += 1
            if hours_ago <= 24:
                recent_24h += 1
            if hours_ago <= 72:
                recent_72h += 1

        # Scoring logic:
        # - Recency matters most: articles in last 6h = strong catalyst signal
        # - Volume matters: many articles = trending topic (likely priced in)
        #   but FEW very recent articles = possible not-yet-priced catalyst
        # - Sweet spot: 1-5 articles in last 6h with many more in 24-72h range
        #   means news is BREAKING and market may not have fully adjusted

        score = 0.0

        # Base: any recent coverage at all
        if recent_72h > 0:
            score += 1.0

        # Recency bonus: articles in last 6 hours (strongest catalyst signal)
        if recent_6h >= 1:
            score += 2.0
        if recent_6h >= 3:
            score += 1.0

        # Coverage depth: sustained attention over 24h
        if recent_24h >= 3:
            score += 1.5
        if recent_24h >= 8:
            score += 1.0

        # "Breaking news" pattern: recent surge vs background
        # If >50% of 72h articles appeared in last 6h → news is breaking
        if recent_72h >= 3 and recent_6h > 0:
            breaking_ratio = recent_6h / recent_72h
            if breaking_ratio > 0.4:
                score += 1.5  # News acceleration

        # Cap at 8
        news_score = min(8.0, score)

        return {
            'news_score': round(news_score, 2),
            'matching_articles': recent_72h,
            'articles_6h': recent_6h,
            'articles_24h': recent_24h,
            'most_relevant_headline': best_headline,
            'recency_hours': round(best_recency, 1) if best_recency < 999 else 999
        }

    # ─────────────────────────────────────────────────────────────────────
    #  Wikipedia Pageview Spikes (replaces Google Trends / SerpApi)
    # ─────────────────────────────────────────────────────────────────────

    def _market_to_wiki_titles(self, market_question: str) -> List[str]:
        """
        Extract likely Wikipedia article titles from a market question.

        Heuristic: pull capitalised multi-word sequences (proper nouns)
        and key topic words. Returns 1-3 candidate titles.
        """
        # First pass: extract capitalised sequences (likely named entities)
        # e.g. "Will Donald Trump win the 2026 election?" → ["Donald Trump"]
        entity_pattern = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'
        entities = re.findall(entity_pattern, market_question)

        # Deduplicate and limit
        seen = set()
        titles = []
        for e in entities:
            normalised = e.strip()
            if normalised not in seen and len(normalised) > 3:
                seen.add(normalised)
                titles.append(normalised)

        # Fallback: use top 1-2 extracted keywords if no entities found
        if not titles:
            keywords = extract_keywords(market_question)
            # Take nouns/proper nouns (longer keywords are usually more specific)
            for kw in sorted(keywords, key=len, reverse=True)[:2]:
                if len(kw) > 3:
                    titles.append(kw.capitalize())

        return titles[:3]  # Max 3 API calls per market

    def fetch_wiki_pageviews(self, title: str) -> Optional[dict]:
        """
        Fetch daily pageview counts for a Wikipedia article.

        Wikimedia REST API: free, no key, 200 req/sec.
        Returns recent + baseline daily views for spike detection.
        """
        if not self.wiki_enabled or not title:
            return None

        cache_key = f'wiki_pv_{title.lower().replace(" ", "_")[:50]}'
        cached = self.cache.get(cache_key, self.wiki_cache_ttl)
        if cached is not None:
            return cached

        try:
            # URL-encode the title (spaces become underscores in Wikipedia)
            wiki_title = title.replace(' ', '_')
            encoded_title = quote(wiki_title, safe='')

            # Fetch last N+2 days of data (extra buffer for API lag)
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.wiki_lookback_days + 2)

            url = (
                f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article'
                f'/en.wikipedia/all-access/all-agents/{encoded_title}/daily'
                f'/{start_date.strftime("%Y%m%d")}/{end_date.strftime("%Y%m%d")}'
            )

            resp = self.session.get(url, timeout=10)

            if resp.status_code == 404:
                # Article doesn't exist — not an error, just no data
                self.cache.set(cache_key, None)
                return None

            resp.raise_for_status()
            data = resp.json()

            items = data.get('items', [])
            if not items:
                self.cache.set(cache_key, None)
                return None

            # Extract daily view counts
            daily_views = [item.get('views', 0) for item in items]

            if len(daily_views) < 3:
                self.cache.set(cache_key, None)
                return None

            # Recent = last 2 days, baseline = everything before that
            recent_views = daily_views[-2:]
            baseline_views = daily_views[:-2] if len(daily_views) > 2 else daily_views

            recent_avg = sum(recent_views) / len(recent_views)
            baseline_avg = sum(baseline_views) / max(1, len(baseline_views))
            baseline_max = max(baseline_views) if baseline_views else 0

            # Spike ratio
            spike_ratio = recent_avg / max(1, baseline_avg)

            result = {
                'title': title,
                'recent_avg_views': round(recent_avg),
                'baseline_avg_views': round(baseline_avg),
                'baseline_max_views': baseline_max,
                'spike_ratio': round(spike_ratio, 2),
                'spike_detected': spike_ratio >= self.wiki_spike_threshold,
                'data_points': len(daily_views),
            }

            self.cache.set(cache_key, result)
            logger.debug(f"Wiki pageviews: {title} → ratio={spike_ratio:.1f}x")
            return result

        except Exception as e:
            logger.warning(f"Wiki pageviews failed for '{title}': {e}")
            return None

    def score_wiki_trends(self, market_question: str) -> dict:
        """
        Score Wikipedia pageview trends for a market's topic.

        Detects attention spikes: if pageviews for related Wikipedia
        articles are significantly above their baseline, something
        newsworthy is happening.

        Returns: {
            trends_score: float (0-7),
            spike_detected: bool,
            best_spike_ratio: float,
            entities_checked: int
        }
        """
        if not self.wiki_enabled:
            return {'trends_score': 0, 'spike_detected': False,
                    'best_spike_ratio': 0, 'entities_checked': 0}

        titles = self._market_to_wiki_titles(market_question)
        if not titles:
            return {'trends_score': 0, 'spike_detected': False,
                    'best_spike_ratio': 0, 'entities_checked': 0}

        best_ratio = 0.0
        any_spike = False
        checked = 0

        for title in titles:
            result = self.fetch_wiki_pageviews(title)
            if result is None:
                continue

            checked += 1
            ratio = result.get('spike_ratio', 0)

            if ratio > best_ratio:
                best_ratio = ratio

            if result.get('spike_detected', False):
                any_spike = True

        if checked == 0:
            return {'trends_score': 0, 'spike_detected': False,
                    'best_spike_ratio': 0, 'entities_checked': 0}

        # Scoring: based on how extreme the pageview spike is
        # 1-2x = normal fluctuation (score 0)
        # 2-3x = mild interest (score 1-2)
        # 3-5x = significant attention (score 3-4)
        # 5-10x = major event (score 5-6)
        # 10x+ = viral/breaking (score 7)
        score = 0.0
        if best_ratio >= 2.0:
            score = min(7.0, (best_ratio - 1.0) * 1.2)

        # Bonus if multiple entities are spiking (corroborates the signal)
        if any_spike and checked >= 2:
            score = min(7.0, score + 0.5)

        return {
            'trends_score': round(score, 2),
            'spike_detected': any_spike,
            'best_spike_ratio': round(best_ratio, 2),
            'entities_checked': checked,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  Crypto Fear & Greed Index (replaces nothing — new macro signal)
    # ─────────────────────────────────────────────────────────────────────

    def fetch_fear_greed(self) -> Optional[dict]:
        """
        Fetch the Crypto Fear & Greed Index from alternative.me.

        Completely free, no API key, no documented rate limit.
        One call per day is sufficient (index updates daily).

        Returns: {value: int (0-100), classification: str, timestamp: str}
        """
        cache_key = 'fear_greed_index'
        cached = self.cache.get(cache_key, 7200)  # 2 hour cache (index is daily)
        if cached is not None:
            return cached

        try:
            url = 'https://api.alternative.me/fng/?limit=1&format=json'
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            entries = data.get('data', [])
            if not entries:
                return None

            entry = entries[0]
            result = {
                'value': int(entry.get('value', 50)),
                'classification': entry.get('value_classification', 'Neutral'),
                'timestamp': entry.get('timestamp', ''),
            }

            self.cache.set(cache_key, result)
            logger.info(f"Fear & Greed: {result['value']} ({result['classification']})")
            return result

        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return None

    def score_fear_greed(self, market_category: str) -> dict:
        """
        Score macro sentiment from Fear & Greed for a market.

        Extreme fear (0-25) on crypto markets = contrarian buy signal.
        Extreme greed (75-100) on crypto markets = caution signal.

        Only applies to crypto-adjacent markets. Non-crypto markets
        get a small sentiment context but not a strong signal.

        Returns: {
            sentiment_score: float (0-3),
            fear_greed_value: int,
            classification: str,
            is_extreme: bool
        }
        """
        fg = self.fetch_fear_greed()
        if fg is None:
            return {'sentiment_score': 0, 'fear_greed_value': 50,
                    'classification': 'Neutral', 'is_extreme': False}

        value = fg['value']
        is_crypto = market_category.lower() in ('crypto', 'cryptocurrency', 'defi', 'nft')

        score = 0.0
        is_extreme = False

        if value <= 20:
            # Extreme Fear — strong contrarian buy for crypto markets
            score = 3.0 if is_crypto else 1.0
            is_extreme = True
        elif value <= 35:
            # Fear — mild contrarian signal
            score = 1.5 if is_crypto else 0.5
        elif value >= 80:
            # Extreme Greed — reduces confidence (markets may be overpriced)
            # We don't subtract, just don't add. The absence of fear signal
            # is itself useful information.
            score = 0.0
            is_extreme = True
        elif value >= 65:
            # Greed — neutral
            score = 0.0

        return {
            'sentiment_score': round(score, 2),
            'fear_greed_value': value,
            'classification': fg['classification'],
            'is_extreme': is_extreme,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  Polymarket Comment Velocity (new — activity-based catalyst detection)
    # ─────────────────────────────────────────────────────────────────────

    def score_comment_velocity(self, comment_count: int, volume_24h: float,
                                volume_total: float) -> dict:
        """
        Score market activity signals from comment count and volume patterns.

        High comment counts relative to volume suggest retail attention/catalyst.
        We can't track velocity (change over time) without historical comment counts,
        but absolute count + volume ratio gives a useful proxy.

        Args:
            comment_count: Current comment count from Gamma API
            volume_24h: 24h trading volume
            volume_total: Total lifetime volume

        Returns: {
            activity_score: float (0-4),
            comment_count: int,
            engagement_ratio: float
        }
        """
        if comment_count <= 0:
            return {'activity_score': 0, 'comment_count': 0, 'engagement_ratio': 0}

        score = 0.0

        # Raw comment count signal: more comments = more attention
        if comment_count >= 100:
            score += 2.0
        elif comment_count >= 50:
            score += 1.5
        elif comment_count >= 20:
            score += 1.0
        elif comment_count >= 5:
            score += 0.5

        # Engagement ratio: comments relative to volume
        # High comments + low volume = retail attention not yet reflected in price
        if volume_total > 0:
            # Normalise: comments per $10k of volume
            engagement = (comment_count / (volume_total / 10000)) if volume_total > 100 else 0
            if engagement > 5.0:
                score += 2.0  # Very high engagement per dollar traded
            elif engagement > 2.0:
                score += 1.0
            elif engagement > 0.5:
                score += 0.5
        else:
            engagement = 0

        return {
            'activity_score': round(min(4.0, score), 2),
            'comment_count': comment_count,
            'engagement_ratio': round(engagement, 2) if volume_total > 0 else 0,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  Combined External Score
    # ─────────────────────────────────────────────────────────────────────

    def compute_external_score(self, market_question: str,
                                market_category: str = '',
                                comment_count: int = 0,
                                volume_24h: float = 0,
                                volume_total: float = 0,
                                articles: List[dict] = None) -> dict:
        """
        Compute combined external confirmation score for a market.

        Sub-signals:
          - GDELT news relevance (0-8): targeted search for market topic
          - Wikipedia pageview spikes (0-7): trend/attention detection
          - Crypto Fear & Greed (0-3): macro sentiment (strongest for crypto)
          - Comment velocity (0-4): Polymarket community attention

        Max raw sum: 22. Capped at 15 to maintain layer weight balance.
        This means a market needs 2-3 sub-signals firing to max out,
        which is exactly the behaviour we want — one noisy signal alone
        isn't enough to dominate the score.

        Returns: {
            news_score: float (0-8),
            trends_score: float (0-7),
            sentiment_score: float (0-3),
            activity_score: float (0-4),
            external_total: float (0-15),
            details: dict
        }
        """
        # News scoring via GDELT
        news_result = self.score_news_relevance(market_question, articles)

        # Trend scoring via Wikipedia pageviews
        trends_result = self.score_wiki_trends(market_question)

        # Macro sentiment via Fear & Greed
        sentiment_result = self.score_fear_greed(market_category)

        # Community activity via comment velocity
        activity_result = self.score_comment_velocity(
            comment_count, volume_24h, volume_total
        )

        total = (
            news_result['news_score'] +
            trends_result['trends_score'] +
            sentiment_result['sentiment_score'] +
            activity_result['activity_score']
        )

        return {
            'news_score': news_result['news_score'],
            'trends_score': trends_result['trends_score'],
            'sentiment_score': sentiment_result['sentiment_score'],
            'activity_score': activity_result['activity_score'],
            'external_total': round(min(15.0, total), 2),
            'details': {
                'matching_articles': news_result.get('matching_articles', 0),
                'articles_6h': news_result.get('articles_6h', 0),
                'articles_24h': news_result.get('articles_24h', 0),
                'most_relevant_headline': news_result.get('most_relevant_headline', ''),
                'recency_hours': news_result.get('recency_hours', 999),
                'spike_detected': trends_result.get('spike_detected', False),
                'best_spike_ratio': trends_result.get('best_spike_ratio', 0),
                'entities_checked': trends_result.get('entities_checked', 0),
                'fear_greed_value': sentiment_result.get('fear_greed_value', 50),
                'fear_greed_class': sentiment_result.get('classification', 'Neutral'),
                'comment_count': activity_result.get('comment_count', 0),
                'engagement_ratio': activity_result.get('engagement_ratio', 0),
            }
        }
