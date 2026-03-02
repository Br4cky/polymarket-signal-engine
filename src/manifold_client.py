"""
Manifold Markets client — cross-platform probability comparison.

Manifold is a play-money prediction market with decent calibration.
Their API is completely free with no authentication required.

We use it to detect probability divergence: if Polymarket says 12%
and Manifold says 25% for the same event, that's a structural signal.
"""

import logging
import re
from typing import Dict, List, Optional

import requests

from src.utils import CacheManager, extract_keywords, safe_float

logger = logging.getLogger(__name__)


class ManifoldClient:
    """Client for Manifold Markets free API."""

    def __init__(self, config: dict, cache: CacheManager):
        self.config = config
        self.cache = cache
        self.enabled = config.get('enabled', True)
        self.api_base = config.get('api_base', 'https://api.manifold.markets/v0')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolySignalEngine/2.0',
            'Accept': 'application/json',
        })
        self.cache_ttl = config.get('cache_ttl', 1800)  # 30 min

        # Pre-fetched search index (populated by build_manifold_index)
        self._manifold_markets: List[dict] = []
        self._index_built = False

    def build_manifold_index(self) -> None:
        """
        Fetch trending/active Manifold markets to use as comparison pool.

        Strategy: fetch top markets by volume. We don't need all markets —
        just enough to find matches for our Polymarket opportunities.
        One batch fetch, cached for 30 min.
        """
        if not self.enabled:
            return

        cache_key = 'manifold_markets_index'
        cached = self.cache.get(cache_key, self.cache_ttl)
        if cached is not None:
            self._manifold_markets = cached
            self._index_built = True
            logger.info(f"Manifold: loaded {len(cached)} markets from cache")
            return

        try:
            # Fetch markets sorted by activity, limit to 200
            # Manifold API: GET /v0/search-markets
            url = f'{self.api_base}/search-markets'
            params = {
                'limit': 200,
                'sort': 'liquidity',
                'filter': 'open',
            }

            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw_markets = resp.json()

            if not isinstance(raw_markets, list):
                logger.warning("Manifold: unexpected response format")
                return

            markets = []
            for m in raw_markets:
                # Only binary markets (YES/NO) are comparable to Polymarket
                if m.get('outcomeType') != 'BINARY':
                    continue

                prob = safe_float(m.get('probability', 0))
                if prob <= 0 or prob >= 1:
                    continue

                question = m.get('question', '')
                keywords = extract_keywords(question)

                markets.append({
                    'id': m.get('id', ''),
                    'question': question,
                    'probability': prob,
                    'keywords': set(keywords),
                    'volume': safe_float(m.get('volume', 0)),
                    'url': m.get('url', ''),
                })

            self._manifold_markets = markets
            self._index_built = True
            self.cache.set(cache_key, markets)
            logger.info(f"Manifold: indexed {len(markets)} binary markets")

        except Exception as e:
            logger.warning(f"Manifold index build failed: {e}")

    def find_matching_probability(self, poly_question: str) -> Optional[float]:
        """
        Find a Manifold market matching a Polymarket question and return
        its probability.

        Matching strategy: keyword overlap with minimum threshold.
        We require >= 40% keyword overlap to consider it a match,
        then take the best match by overlap ratio.

        Args:
            poly_question: Polymarket market question text

        Returns:
            Manifold probability (0-1) or None if no match found
        """
        if not self._index_built or not self._manifold_markets:
            return None

        poly_keywords = set(extract_keywords(poly_question))
        if not poly_keywords or len(poly_keywords) < 2:
            return None

        best_match = None
        best_overlap = 0.0

        for m in self._manifold_markets:
            manifold_keywords = m.get('keywords', set())
            # Handle case where keywords were serialized as list (from cache)
            if isinstance(manifold_keywords, list):
                manifold_keywords = set(manifold_keywords)

            overlap = poly_keywords & manifold_keywords
            overlap_ratio = len(overlap) / max(1, len(poly_keywords))

            if overlap_ratio >= 0.4 and overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_match = m

        if best_match and best_overlap >= 0.4:
            logger.debug(
                f"Manifold match ({best_overlap:.0%}): "
                f"'{poly_question[:40]}' → '{best_match['question'][:40]}' "
                f"(prob={best_match['probability']:.2f})"
            )
            return best_match['probability']

        return None
