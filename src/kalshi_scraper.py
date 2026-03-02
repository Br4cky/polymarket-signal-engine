"""
Kalshi Scraper — Cross-platform price comparison for detecting divergence.
Kalshi API is public for market data (no auth needed for reading).
Base URL: https://api.elections.kalshi.com/trade-api/v2
"""

import logging
import requests
from typing import List, Optional, Dict
from src.utils import CacheManager, extract_keywords

logger = logging.getLogger(__name__)


class KalshiClient:
    """Kalshi API client for cross-platform price comparison."""

    def __init__(self, config: dict, cache_manager: CacheManager):
        """
        Initialize Kalshi API client.

        Args:
            config: Configuration dict (kalshi section) with keys like api_base, cache_ttl
            cache_manager: CacheManager instance for caching API responses
        """
        self.api_base = config.get('api_base', 'https://api.elections.kalshi.com/trade-api/v2')
        self.cache_ttl = config.get('cache_ttl', 300)
        self.cache_manager = cache_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolymarketSignalEngine/1.0'
        })

    def fetch_events(self) -> List[dict]:
        """
        Fetch open events from Kalshi API with pagination.

        Returns:
            List of event dicts, cached for cache_ttl seconds
        """
        cache_key = 'kalshi_events'
        cached = self.cache_manager.get(cache_key)
        if cached is not None:
            return cached

        try:
            events = []
            offset = 0
            limit = 100

            while True:
                url = f"{self.api_base}/events"
                params = {
                    'status': 'open',
                    'limit': limit,
                    'offset': offset
                }
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                batch = data.get('events', [])
                if not batch:
                    break

                events.extend(batch)

                if len(batch) < limit:
                    break

                offset += limit

            self.cache_manager.set(cache_key, events, ttl=self.cache_ttl)
            return events

        except Exception as e:
            logger.error(f"Error fetching Kalshi events: {e}")
            return []

    def fetch_markets_for_event(self, event_ticker: str) -> List[dict]:
        """
        Fetch open markets for a specific event.

        Args:
            event_ticker: Event ticker symbol

        Returns:
            List of market dicts with ticker, title, yes_bid, yes_ask, no_bid, no_ask, volume
        """
        cache_key = f'kalshi_markets_{event_ticker}'
        cached = self.cache_manager.get(cache_key)
        if cached is not None:
            return cached

        try:
            url = f"{self.api_base}/markets"
            params = {
                'event_ticker': event_ticker,
                'status': 'open'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            markets = data.get('markets', [])
            self.cache_manager.set(cache_key, markets, ttl=self.cache_ttl)
            return markets

        except Exception as e:
            logger.error(f"Error fetching Kalshi markets for event {event_ticker}: {e}")
            return []

    def find_matching_markets(self, polymarket_question: str, kalshi_events: List[dict]) -> Optional[dict]:
        """
        Find Kalshi markets matching a Polymarket question via keyword overlap.

        Args:
            polymarket_question: Question text from Polymarket
            kalshi_events: List of Kalshi event dicts

        Returns:
            Best matching market dict if overlap > 3 keywords, else None
        """
        try:
            poly_keywords = extract_keywords(polymarket_question)

            best_match = None
            best_overlap = 0

            for event in kalshi_events:
                event_title = event.get('title', '')
                event_ticker = event.get('ticker', '')

                event_keywords = extract_keywords(f"{event_title} {event_ticker}")

                overlap = len(set(poly_keywords) & set(event_keywords))

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = event

            if best_overlap >= 3:
                return best_match
            else:
                return None

        except Exception as e:
            logger.error(f"Error finding matching markets: {e}")
            return None

    def compute_divergence(self, poly_price: float, kalshi_price: float) -> dict:
        """
        Compute price divergence between Polymarket and Kalshi.

        Args:
            poly_price: Yes price on Polymarket (0-1)
            kalshi_price: Yes price on Kalshi (0-1)

        Returns:
            Dict with divergence_pct, direction, and signal_strength (0-15)
        """
        try:
            divergence = abs(poly_price - kalshi_price)

            if poly_price > kalshi_price:
                direction = "poly_higher"
            else:
                direction = "kalshi_higher"

            if divergence > 0.03:
                signal_strength = min(15, divergence * 300)
            else:
                signal_strength = 0

            return {
                'divergence_pct': divergence * 100,
                'direction': direction,
                'signal_strength': signal_strength
            }

        except Exception as e:
            logger.error(f"Error computing divergence: {e}")
            return {
                'divergence_pct': 0,
                'direction': 'unknown',
                'signal_strength': 0
            }
