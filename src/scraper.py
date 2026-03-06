"""
Polymarket API client for fetching and normalizing market data.
Handles market fetching, order book data, and price history.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import requests

from src.utils import CacheManager, RateLimiter, safe_float

logger = logging.getLogger(__name__)


class PolymarketClient:
    """Client for Polymarket Gamma and CLOB APIs."""

    def __init__(self, config: dict, cache: CacheManager):
        """
        Initialize Polymarket client.

        Args:
            config: Configuration dict with keys:
                - gamma_api_base: Base URL for Gamma API (e.g., https://gamma-api.polymarket.com)
                - clob_api_base: Base URL for CLOB API
                - cache_ttl_markets: TTL in seconds for market cache
                - cache_ttl_prices: TTL in seconds for price/book cache
            cache: CacheManager instance for caching responses
        """
        self.config = config
        self.cache = cache
        self.rate_limiter = RateLimiter(max_per_minute=60)

        # Create session with User-Agent
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Polymarket-Signal-Engine/1.0"
        })

    def _get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 15
    ) -> Optional[dict]:
        """
        Make GET request with rate limiting and error handling.

        Args:
            url: Full URL to request
            params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response or None if request failed
        """
        self.rate_limiter.wait_if_needed()

        try:
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            return None

    def fetch_all_markets(self) -> List[dict]:
        """
        Fetch all active, non-closed markets from Gamma API.
        Paginates through results and caches the full list.

        Returns:
            List of raw market dicts from API
        """
        cache_key = "all_markets"
        ttl = self.config.get("cache_ttl_markets", 300)

        # Check cache first
        cached = self.cache.get(cache_key, ttl)
        if cached is not None:
            logger.debug(f"Using cached markets: {len(cached)} markets")
            return cached

        base_url = self.config.get("gamma_api_base", "https://gamma-api.polymarket.com")
        markets = []
        offset = 0
        max_markets = 2000

        while len(markets) < max_markets:
            params = {
                "closed": "false",
                "active": "true",
                "limit": 100,
                "offset": offset,
                "order": "volume24hr",
                "ascending": "false"
            }

            result = self._get(f"{base_url}/markets", params=params)
            if not result:
                logger.warning(f"Failed to fetch markets at offset {offset}")
                break

            batch = result.get("data", []) if isinstance(result, dict) else result

            if not batch:
                logger.debug(f"No more markets at offset {offset}")
                break

            markets.extend(batch)
            offset += 100

            if len(batch) < 100:
                break

        markets = markets[:max_markets]
        logger.info(f"Fetched {len(markets)} markets")

        # Cache the results
        self.cache.set(cache_key, markets)

        return markets

    def normalize_market(self, raw: dict) -> Optional[dict]:
        """
        Parse raw Gamma API market into clean normalized format.

        Args:
            raw: Raw market dict from Gamma API

        Returns:
            Normalized market dict or None if invalid
        """
        try:
            # Extract basic fields
            market_id = raw.get("id") or raw.get("market_id")
            if not market_id:
                return None

            question = raw.get("question", "")
            category = raw.get("category", "")
            slug = raw.get("slug", "")

            # Parse tokens and prices
            tokens = []
            try:
                outcomes_raw = raw.get("outcomes", [])
                if isinstance(outcomes_raw, str):
                    outcomes = json.loads(outcomes_raw)
                else:
                    outcomes = outcomes_raw

                prices_raw = raw.get("outcomePrices", [])
                if isinstance(prices_raw, str):
                    prices = json.loads(prices_raw)
                else:
                    prices = prices_raw

                token_ids_raw = raw.get("clobTokenIds", [])
                if isinstance(token_ids_raw, str):
                    token_ids = json.loads(token_ids_raw)
                else:
                    token_ids = token_ids_raw

                for i, outcome in enumerate(outcomes):
                    token = {
                        "token_id": token_ids[i] if i < len(token_ids) else None,
                        "outcome": outcome,
                        "current_price": safe_float(prices[i] if i < len(prices) else 0)
                    }
                    if token["token_id"]:
                        tokens.append(token)

            except (json.JSONDecodeError, IndexError, TypeError) as e:
                logger.warning(f"Failed to parse tokens for market {market_id}: {e}")
                return None

            if not tokens:
                logger.debug(f"Market {market_id} has no valid tokens")
                return None

            # Parse volume and liquidity
            volume_24h = safe_float(raw.get("volume24hr", 0))
            volume_total = safe_float(raw.get("volume", 0))
            liquidity = safe_float(raw.get("liquidity", 0))

            # Parse dates
            resolution_date = raw.get("endDate") or raw.get("endDateIso")
            created_at = raw.get("createdAt") or raw.get("creationTime")

            condition_id = raw.get("conditionId", raw.get("condition_id", ""))

            # Activity signals: comment count and competitive markets
            comment_count = int(raw.get("commentCount", 0) or 0)

            return {
                "market_id": market_id,
                "condition_id": condition_id,
                "question": question,
                "category": category,
                "slug": slug,
                "tokens": tokens,
                "volume_24h": volume_24h,
                "volume_total": volume_total,
                "liquidity": liquidity,
                "resolution_date": resolution_date,
                "created_at": created_at,
                "comment_count": comment_count,
            }

        except Exception as e:
            logger.error(f"Error normalizing market {raw.get('id', 'unknown')}: {e}")
            return None

    def fetch_order_book(self, token_id: str) -> Optional[dict]:
        """
        Fetch order book (bids/asks) for a token from CLOB API.
        Caches result.

        Args:
            token_id: CLOB token ID

        Returns:
            Raw order book dict or None if request failed
        """
        cache_key = f"orderbook:{token_id}"
        ttl = self.config.get("cache_ttl_prices", 60)

        cached = self.cache.get(cache_key, ttl)
        if cached is not None:
            return cached

        clob_base = self.config.get("clob_api_base", "https://clob.polymarket.com")
        result = self._get(f"{clob_base}/book", params={"token_id": token_id})

        if result:
            self.cache.set(cache_key, result)

        return result

    def fetch_price_history(self, token_id: str) -> List[dict]:
        """
        Fetch price history for a token.
        Caches result for 1 hour.

        Args:
            token_id: CLOB token ID

        Returns:
            List of {timestamp, price} dicts
        """
        cache_key = f"price_history:{token_id}"
        ttl = 3600  # 1 hour

        cached = self.cache.get(cache_key, ttl)
        if cached is not None:
            return cached

        clob_base = self.config.get("clob_api_base", "https://clob.polymarket.com")
        params = {
            "market": token_id,
            "interval": "max",
            "fidelity": "60"
        }

        result = self._get(f"{clob_base}/prices-history", params=params)
        if not result:
            return []

        # Normalize response format
        # CLOB API returns {"history": [...]} — NOT {"data": [...]}
        history = []
        if isinstance(result, dict):
            data = result.get("history") or result.get("data") or []
        elif isinstance(result, list):
            data = result
        else:
            data = []

        if not data:
            logger.debug(
                f"Empty price history for token {token_id[:30]}... "
                f"(response keys: {list(result.keys()) if isinstance(result, dict) else type(result).__name__})"
            )

        for entry in data:
            if isinstance(entry, dict):
                # Handle various response formats
                timestamp = (
                    entry.get("timestamp") or
                    entry.get("time") or
                    entry.get("t")
                )
                price = (
                    entry.get("price") or
                    entry.get("p") or
                    entry.get("mid_price")
                )

                if timestamp and price is not None:
                    # Normalise timestamp to numeric (CLOB often returns string)
                    try:
                        timestamp = float(timestamp)
                    except (ValueError, TypeError):
                        pass  # keep as-is (likely ISO string)
                    history.append({
                        "timestamp": timestamp,
                        "price": safe_float(price)
                    })

        self.cache.set(cache_key, history)
        return history

    def fetch_enriched_markets(self, max_markets: int = 300) -> List[dict]:
        """
        Fetch markets and enrich top markets with order book data.
        Enriches top 80 markets by 24h volume with bid/ask information.

        Args:
            max_markets: Maximum number of markets to return

        Returns:
            List of normalized market dicts with optional enrichment
        """
        # Fetch and normalize all markets
        raw_markets = self.fetch_all_markets()
        markets = []

        for raw in raw_markets:
            normalized = self.normalize_market(raw)
            if normalized:
                markets.append(normalized)

        logger.info(f"Normalized {len(markets)} markets")

        # Sort by 24h volume
        markets.sort(key=lambda m: m["volume_24h"], reverse=True)

        # Enrich top 80 by volume
        num_to_enrich = min(80, len(markets))
        for i in range(num_to_enrich):
            market = markets[i]
            for token in market.get("tokens", []):
                try:
                    book = self.fetch_order_book(token["token_id"])

                    if book:
                        # Parse bid/ask data
                        bids = book.get("bids", [])
                        asks = book.get("asks", [])

                        if bids and isinstance(bids[0], (list, dict)):
                            bid_price = safe_float(
                                bids[0][0] if isinstance(bids[0], list) else bids[0].get("price", 0)
                            )
                            bid_depth = sum(
                                safe_float(b[1] if isinstance(b, list) else b.get("size", 0))
                                for b in bids
                            )
                        else:
                            bid_price = 0.0
                            bid_depth = 0.0

                        if asks and isinstance(asks[0], (list, dict)):
                            ask_price = safe_float(
                                asks[0][0] if isinstance(asks[0], list) else asks[0].get("price", 0)
                            )
                            ask_depth = sum(
                                safe_float(a[1] if isinstance(a, list) else a.get("size", 0))
                                for a in asks
                            )
                        else:
                            ask_price = 0.0
                            ask_depth = 0.0

                        # Calculate spread
                        spread = ask_price - bid_price if (ask_price and bid_price) else 0.0
                        spread_pct = (
                            (spread / bid_price * 100) if (spread and bid_price) else 0.0
                        )

                        token.update({
                            "bid": bid_price,
                            "ask": ask_price,
                            "bid_depth": bid_depth,
                            "ask_depth": ask_depth,
                            "spread": spread,
                            "spread_pct": spread_pct
                        })

                except Exception as e:
                    logger.warning(f"Failed to enrich token {token.get('token_id')}: {e}")

        # Set defaults for ALL markets (including enrichment failures)
        for market in markets:
            for token in market.get("tokens", []):
                token.setdefault("bid", 0.0)
                token.setdefault("ask", 0.0)
                token.setdefault("bid_depth", 0.0)
                token.setdefault("ask_depth", 0.0)
                token.setdefault("spread", 0.0)
                token.setdefault("spread_pct", 0.0)

        return markets[:max_markets]
