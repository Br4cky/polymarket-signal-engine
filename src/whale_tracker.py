"""
Whale Tracker — Smart money detection via on-chain data and leaderboards.
Uses The Graph subgraph for position concentration.
Uses Polymarket leaderboard for top trader tracking.
"""

import logging
import requests
import json
from typing import List, Dict, Optional
from src.utils import CacheManager

logger = logging.getLogger(__name__)


class WhaleTracker:
    """Smart money detection via Polymarket leaderboard and The Graph subgraph."""

    def __init__(self, config: dict, cache_manager: CacheManager):
        """
        Initialize Whale Tracker.

        Args:
            config: Configuration dict (whale_tracking section) with keys like
                    subgraph_url, cache_ttl_leaderboard, cache_ttl_positions
            cache_manager: CacheManager instance for caching API responses
        """
        self.subgraph_url = config.get('subgraph_url')
        self.cache_ttl_leaderboard = config.get('cache_ttl_leaderboard', 3600)
        self.cache_ttl_positions = config.get('cache_ttl_positions', 1800)
        self.cache_manager = cache_manager
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolymarketSignalEngine/1.0'
        })

    def fetch_leaderboard(self, limit: int = 50) -> List[dict]:
        """
        Fetch top traders from Polymarket leaderboard.

        Args:
            limit: Maximum number of traders to fetch (default 50)

        Returns:
            List of dicts with address, username, volume, profit, positions_count
            Cached for 1 hour. Returns empty list on API failure.
        """
        cache_key = f'polymarket_leaderboard_{limit}'
        cached = self.cache_manager.get(cache_key, self.cache_ttl_leaderboard)
        if cached is not None:
            return cached

        try:
            url = "https://gamma-api.polymarket.com/users"
            params = {
                'limit': limit,
                'sortBy': 'volume'
            }
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            leaderboard = []
            users = data if isinstance(data, list) else data.get('users', [])

            for user in users:
                try:
                    entry = {
                        'address': user.get('address', ''),
                        'username': user.get('username', user.get('name', '')),
                        'volume': float(user.get('volume', 0)),
                        'profit': float(user.get('profit', 0)),
                        'positions_count': int(user.get('positions_count', 0))
                    }
                    leaderboard.append(entry)
                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(f"Skipping malformed leaderboard entry: {e}")
                    continue

            self.cache_manager.set(cache_key, leaderboard, ttl=self.cache_ttl_leaderboard)
            return leaderboard

        except Exception as e:
            logger.error(f"Error fetching Polymarket leaderboard: {e}")
            return []

    def fetch_top_positions_for_market(self, market_id: str) -> dict:
        """
        Fetch top positions for a market from The Graph subgraph.

        Args:
            market_id: Market identifier

        Returns:
            Dict with top_wallets, concentration_pct, and whale_count.
            Returns empty dict if subgraph not configured or fails.
        """
        cache_key = f'market_positions_{market_id}'
        cached = self.cache_manager.get(cache_key, self.cache_ttl_positions)
        if cached is not None:
            return cached

        if not self.subgraph_url:
            logger.debug("Subgraph URL not configured, skipping position fetch")
            return {'top_wallets': [], 'concentration_pct': 0, 'whale_count': 0}

        try:
            query = """
            {
              positions(
                where: {market: "%s"}
                orderBy: value
                orderDirection: desc
                first: 10
              ) {
                user {
                  id
                }
                value
                outcome
              }
            }
            """ % market_id

            payload = {'query': query}
            response = self.session.post(
                self.subgraph_url,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            positions = data.get('data', {}).get('positions', [])

            top_wallets = []
            total_value = 0

            for position in positions:
                try:
                    user_id = position.get('user', {}).get('id', '')
                    value = float(position.get('value', 0))
                    outcome = position.get('outcome', '')

                    if user_id and value > 0:
                        top_wallets.append({
                            'address': user_id,
                            'value': value,
                            'side': outcome
                        })
                        total_value += value

                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(f"Skipping malformed position: {e}")
                    continue

            concentration_pct = 0
            if total_value > 0 and top_wallets:
                top_5_value = sum(w['value'] for w in top_wallets[:5])
                concentration_pct = (top_5_value / total_value) * 100

            result = {
                'top_wallets': top_wallets,
                'concentration_pct': concentration_pct,
                'whale_count': len(top_wallets)
            }

            self.cache_manager.set(cache_key, result, ttl=self.cache_ttl_positions)
            return result

        except Exception as e:
            logger.error(f"Error fetching top positions for market {market_id}: {e}")
            return {'top_wallets': [], 'concentration_pct': 0, 'whale_count': 0}

    def compute_smart_money_score(self, market_id: str, token_id: str) -> dict:
        """
        Compute smart money score based on whale activity and concentration.

        Args:
            market_id: Market identifier
            token_id: Token identifier (outcome)

        Returns:
            Dict with whale_accumulation, leaderboard_alignment, position_concentration,
            and smart_money_total (0-25). All zeros if data unavailable (graceful degradation).
        """
        try:
            leaderboard = self.fetch_leaderboard()
            positions = self.fetch_top_positions_for_market(market_id)

            whale_accumulation = 0.0
            leaderboard_alignment = 0.0
            position_concentration = 0.0

            # Compute whale_accumulation (0-10)
            # Based on number of top traders active in this market
            top_wallet_addresses = {w['address'].lower() for w in positions.get('top_wallets', [])}
            leaderboard_addresses = {u['address'].lower() for u in leaderboard if u.get('address')}

            whale_count_in_market = len(top_wallet_addresses & leaderboard_addresses)
            max_whales = min(len(leaderboard), 10)
            if max_whales > 0:
                whale_accumulation = min(10, (whale_count_in_market / max_whales) * 10)

            # Compute leaderboard_alignment (0-8)
            # How many top 20 traders are present in this market's top positions
            if leaderboard:
                top_20 = set(u['address'].lower() for u in leaderboard[:20])
                alignment_count = len(top_wallet_addresses & top_20)
                leaderboard_alignment = min(8, (alignment_count / 20) * 8)

            # Compute position_concentration (0-7)
            # Top 5 wallets concentration as percentage
            concentration_pct = positions.get('concentration_pct', 0)
            if concentration_pct > 50:
                position_concentration = min(7, (concentration_pct - 50) / 10)
            else:
                position_concentration = 0

            smart_money_total = whale_accumulation + leaderboard_alignment + position_concentration

            return {
                'whale_accumulation': round(whale_accumulation, 2),
                'leaderboard_alignment': round(leaderboard_alignment, 2),
                'position_concentration': round(position_concentration, 2),
                'smart_money_total': round(smart_money_total, 2)
            }

        except Exception as e:
            logger.error(f"Error computing smart money score for {market_id}: {e}")
            return {
                'whale_accumulation': 0.0,
                'leaderboard_alignment': 0.0,
                'position_concentration': 0.0,
                'smart_money_total': 0.0
            }
