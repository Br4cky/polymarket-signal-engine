"""
Whale Tracker — Smart money detection via Polymarket's free REST APIs.

Uses:
  - data-api.polymarket.com/leaderboard  → top traders by profit/volume
  - data-api.polymarket.com/positions    → each trader's open positions
  - data-api.polymarket.com/activity     → recent trade activity

No paid APIs, no subgraph, no API keys required.
Rate limit: ~1,000 calls/hour on the data API (free).
"""

import logging
import requests
from typing import List, Dict, Optional
from src.utils import CacheManager

logger = logging.getLogger(__name__)


class WhaleTracker:
    """Smart money detection via Polymarket leaderboard and position APIs."""

    def __init__(self, config: dict, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self.data_api = config.get('data_api_base', 'https://data-api.polymarket.com')
        self.top_n_traders = config.get('top_n_traders', 50)
        self.cache_ttl_leaderboard = config.get('cache_ttl_leaderboard', 3600)
        self.cache_ttl_positions = config.get('cache_ttl_positions', 1800)
        self._leaderboard_failed = False
        self._positions_failed = False
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolymarketSignalEngine/2.0',
            'Accept': 'application/json'
        })

        # Pre-built whale position index: {condition_id: [wallet_addresses]}
        self._whale_market_index: Dict[str, List[dict]] = {}
        self._index_built = False

    # ── Leaderboard ──────────────────────────────────────────────────────────

    def fetch_leaderboard(self, limit: int = None) -> List[dict]:
        """
        Fetch top traders from Polymarket leaderboard.
        Cached for 1 hour. Returns list of {address, username, volume, profit}.
        """
        if self._leaderboard_failed:
            return []

        limit = limit or self.top_n_traders
        cache_key = f'pm_leaderboard_{limit}'
        cached = self.cache.get(cache_key, self.cache_ttl_leaderboard)
        if cached is not None:
            return cached

        try:
            resp = self.session.get(
                f"{self.data_api}/leaderboard",
                params={
                    'limit': min(limit, 100),
                    'orderBy': 'PNL',
                    'timePeriod': 'ALL'
                },
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()

            users = data if isinstance(data, list) else data.get('leaderboard', data.get('users', []))
            leaderboard = []

            for user in users:
                try:
                    entry = {
                        'address': user.get('proxyWallet', user.get('address', '')),
                        'username': user.get('userName', user.get('username', '')),
                        'volume': float(user.get('vol', user.get('volume', 0))),
                        'profit': float(user.get('pnl', user.get('profit', 0))),
                    }
                    if entry['address']:
                        leaderboard.append(entry)
                except (KeyError, ValueError, TypeError):
                    continue

            self.cache.set(cache_key, leaderboard)
            logger.info(f"Fetched {len(leaderboard)} traders from leaderboard")
            return leaderboard

        except Exception as e:
            logger.warning(f"Leaderboard API failed ({e}), disabling for this run")
            self._leaderboard_failed = True
            return []

    # ── Trader Positions ─────────────────────────────────────────────────────

    def fetch_trader_positions(self, wallet_address: str) -> List[dict]:
        """
        Fetch all open positions for a specific trader wallet.
        Returns list of {conditionId, market_slug, size, avgPrice, currentValue, pnl}.
        Cached for 30 min per wallet.
        """
        if self._positions_failed:
            return []

        cache_key = f'trader_pos_{wallet_address[:16]}'
        cached = self.cache.get(cache_key, self.cache_ttl_positions)
        if cached is not None:
            return cached

        try:
            resp = self.session.get(
                f"{self.data_api}/positions",
                params={
                    'user': wallet_address,
                    'sortBy': 'CURRENT',
                    'sortDirection': 'DESC',
                    'sizeThreshold': 1  # Min 1 share
                },
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()

            positions_list = data if isinstance(data, list) else data.get('positions', [])
            positions = []

            for pos in positions_list:
                try:
                    positions.append({
                        'conditionId': pos.get('conditionId', pos.get('asset', {}).get('conditionId', '')),
                        'market_slug': pos.get('market_slug', pos.get('slug', '')),
                        'title': pos.get('title', ''),
                        'size': float(pos.get('size', pos.get('currentShares', 0))),
                        'avgPrice': float(pos.get('avgPrice', pos.get('averagePrice', 0))),
                        'currentValue': float(pos.get('curVal', pos.get('currentValue', 0))),
                        'cashPnl': float(pos.get('cashPnl', 0)),
                        'percentPnl': float(pos.get('percentPnl', 0)),
                        'outcome': pos.get('outcome', pos.get('outcomeIndex', '')),
                    })
                except (KeyError, ValueError, TypeError):
                    continue

            self.cache.set(cache_key, positions)
            return positions

        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                logger.warning("Rate limited on positions API, pausing")
                self._positions_failed = True
            else:
                logger.warning(f"Positions API failed for {wallet_address[:10]}...: {e}")
            return []
        except Exception as e:
            logger.warning(f"Positions API failed for {wallet_address[:10]}...: {e}")
            return []

    # ── Build Whale Index ────────────────────────────────────────────────────

    def build_whale_index(self) -> None:
        """
        Build an index of which markets the top traders are positioned in.
        Called once per engine run. Maps condition_id → list of whale entries.

        This is the key optimisation: instead of querying per-market,
        we query per-trader (50 calls) and build a reverse index.
        """
        if self._index_built:
            return

        cache_key = 'whale_market_index'
        cached = self.cache.get(cache_key, self.cache_ttl_positions)
        if cached is not None:
            self._whale_market_index = cached
            self._index_built = True
            logger.info(f"Loaded whale index from cache ({len(cached)} markets)")
            return

        leaderboard = self.fetch_leaderboard()
        if not leaderboard:
            self._index_built = True
            return

        index: Dict[str, List[dict]] = {}
        traders_fetched = 0
        total_positions = 0

        for trader in leaderboard:
            address = trader.get('address', '')
            if not address:
                continue

            positions = self.fetch_trader_positions(address)
            traders_fetched += 1

            for pos in positions:
                cid = pos.get('conditionId', '')
                if not cid:
                    continue

                if cid not in index:
                    index[cid] = []

                index[cid].append({
                    'address': address,
                    'username': trader.get('username', ''),
                    'profit': trader.get('profit', 0),
                    'size': pos.get('size', 0),
                    'currentValue': pos.get('currentValue', 0),
                    'outcome': pos.get('outcome', ''),
                })
                total_positions += 1

            # Stop early if rate limited
            if self._positions_failed:
                logger.warning(f"Rate limited after {traders_fetched} traders, using partial index")
                break

        self._whale_market_index = index
        self._index_built = True
        self.cache.set(cache_key, index)
        logger.info(
            f"Built whale index: {traders_fetched} traders → "
            f"{total_positions} positions across {len(index)} markets"
        )

    # ── Smart Money Score ────────────────────────────────────────────────────

    def compute_smart_money_score(self, market_id: str, token_id: str) -> dict:
        """
        Compute smart money score based on whale presence in a market.

        Uses the pre-built whale index (condition_id → whale positions).

        Returns:
            whale_accumulation (0-10): How many top traders are in this market
            leaderboard_alignment (0-8): Are the most profitable traders here
            position_concentration (0-7): How concentrated is whale capital
            smart_money_total (0-25): Combined score
        """
        try:
            # Ensure index is built (no-op if already done)
            self.build_whale_index()

            whale_accumulation = 0.0
            leaderboard_alignment = 0.0
            position_concentration = 0.0

            # Look up this market in the whale index
            # Try both market_id and condition_id patterns
            whales_in_market = self._whale_market_index.get(market_id, [])

            if not whales_in_market:
                return {
                    'whale_accumulation': 0.0,
                    'leaderboard_alignment': 0.0,
                    'position_concentration': 0.0,
                    'smart_money_total': 0.0
                }

            # ── Whale Accumulation (0-10) ──
            # How many top traders are positioned in this market
            unique_whales = len(set(w['address'] for w in whales_in_market))
            max_possible = min(self.top_n_traders, 50)
            whale_accumulation = min(10.0, (unique_whales / max(1, max_possible)) * 30)
            # Scale: 1 whale ≈ 0.6pts, 5 whales ≈ 3pts, 15 whales ≈ 9pts

            # ── Leaderboard Alignment (0-8) ──
            # Are the most profitable traders (top 20 by PnL) in this market?
            leaderboard = self.fetch_leaderboard()
            if leaderboard:
                top_20_addresses = set(
                    t['address'].lower() for t in leaderboard[:20] if t.get('address')
                )
                whale_addresses = set(w['address'].lower() for w in whales_in_market)
                alignment_count = len(whale_addresses & top_20_addresses)
                leaderboard_alignment = min(8.0, alignment_count * 2.0)
                # Scale: 1 top-20 trader = 2pts, 4 top-20 traders = 8pts (max)

            # ── Position Concentration (0-7) ──
            # How much capital do whales have concentrated here?
            total_whale_value = sum(w.get('currentValue', 0) for w in whales_in_market)
            if total_whale_value > 0 and len(whales_in_market) >= 2:
                # Sort by value, check if top holders dominate
                sorted_whales = sorted(whales_in_market, key=lambda w: w.get('currentValue', 0), reverse=True)
                top_3_value = sum(w.get('currentValue', 0) for w in sorted_whales[:3])
                concentration_pct = (top_3_value / total_whale_value) * 100 if total_whale_value > 0 else 0

                if concentration_pct > 50:
                    position_concentration = min(7.0, (concentration_pct - 50) / 7.0)

                # Bonus for sheer capital size (>$50k total whale capital = extra points)
                if total_whale_value > 50000:
                    position_concentration = min(7.0, position_concentration + 2.0)
                elif total_whale_value > 10000:
                    position_concentration = min(7.0, position_concentration + 1.0)

            smart_money_total = whale_accumulation + leaderboard_alignment + position_concentration

            return {
                'whale_accumulation': round(whale_accumulation, 2),
                'leaderboard_alignment': round(leaderboard_alignment, 2),
                'position_concentration': round(position_concentration, 2),
                'smart_money_total': round(min(25.0, smart_money_total), 2)
            }

        except Exception as e:
            logger.error(f"Error computing smart money score for {market_id}: {e}")
            return {
                'whale_accumulation': 0.0,
                'leaderboard_alignment': 0.0,
                'position_concentration': 0.0,
                'smart_money_total': 0.0
            }
