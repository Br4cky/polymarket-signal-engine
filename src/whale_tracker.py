"""
Whale Tracker — Smart money detection via Polymarket's free REST APIs.

Strategy: Pull leaderboard across multiple time windows (ALL, MONTH, WEEK)
and only trust traders who are consistently profitable — not just one-hit
wonders from a lucky bet. A trader appearing on multiple period leaderboards
gets a higher "quality" score.

Uses:
  - data-api.polymarket.com/leaderboard  → top traders by PnL across periods
  - data-api.polymarket.com/positions    → each trader's open positions

No paid APIs, no subgraph, no API keys required.
Rate limit: ~1,000 calls/hour on the data API (free).
"""

import json
import logging
import requests
import time
from typing import List, Dict, Optional
from src.utils import CacheManager, safe_float

logger = logging.getLogger(__name__)


def _normalize_condition_id(cid: str) -> str:
    """
    Normalize a conditionId to lowercase for consistent matching.
    Handles both hex (0x...) and numeric formats.
    Returns empty string if input is falsy.
    """
    if not cid:
        return ''
    return str(cid).strip().lower()


def _extract_condition_id(pos: dict) -> str:
    """
    Extract conditionId from a position object, trying multiple field paths.

    Known Polymarket positions API formats:
      - Top-level 'conditionId' (standard)
      - Nested under 'asset.conditionId' (if asset is a dict)
      - 'asset' itself may BE the conditionId (some API versions)
      - 'condition_id' (snake_case variant)
      - Under 'market.conditionId' (if nested market object)
    """
    # 1. Direct top-level conditionId (most common)
    cid = pos.get('conditionId', '')
    if cid:
        return _normalize_condition_id(cid)

    # 2. Snake-case variant
    cid = pos.get('condition_id', '')
    if cid:
        return _normalize_condition_id(cid)

    # 3. Nested under asset (if asset is a dict)
    asset = pos.get('asset')
    if isinstance(asset, dict):
        cid = asset.get('conditionId', asset.get('condition_id', ''))
        if cid:
            return _normalize_condition_id(cid)

    # 4. asset itself might be the conditionId (hex string starting with 0x)
    if isinstance(asset, str) and asset.startswith('0x') and len(asset) > 10:
        # This is likely the token asset ID, not the conditionId — skip
        pass

    # 5. Nested under market object
    market = pos.get('market')
    if isinstance(market, dict):
        cid = market.get('conditionId', market.get('condition_id', ''))
        if cid:
            return _normalize_condition_id(cid)

    return ''


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
        self._positions_empty_count = 0  # Track how many traders returned 0 positions
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolymarketSignalEngine/3.1',
            'Accept': 'application/json'
        })

        # Pre-built whale position index: {condition_id: [whale entries]}
        self._whale_market_index: Dict[str, List[dict]] = {}
        self._index_built = False
        # Quality-filtered trader list
        self._quality_traders: List[dict] = []

    # ── Multi-Period Leaderboard ─────────────────────────────────────────────

    def _fetch_leaderboard_period(self, time_period: str, limit: int = 50) -> List[dict]:
        """Fetch leaderboard for a single time period. Cached per period."""
        if self._leaderboard_failed:
            return []

        cache_key = f'pm_lb_{time_period}_{limit}'
        cached = self.cache.get(cache_key, self.cache_ttl_leaderboard)
        if cached is not None:
            return cached

        try:
            resp = self.session.get(
                f"{self.data_api}/v1/leaderboard",
                params={
                    'limit': min(limit, 50),
                    'orderBy': 'PNL',
                    'timePeriod': time_period,
                    'category': 'OVERALL'
                },
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()

            users = data if isinstance(data, list) else data.get('leaderboard', data.get('users', []))
            result = []

            for user in users:
                try:
                    entry = {
                        'address': user.get('proxyWallet', user.get('address', '')),
                        'username': user.get('userName', user.get('username', '')),
                        'volume': safe_float(user.get('vol', user.get('volume', 0))),
                        'profit': safe_float(user.get('pnl', user.get('profit', 0))),
                    }
                    if entry['address'] and entry['profit'] > 0:
                        result.append(entry)
                except (KeyError, ValueError, TypeError):
                    continue

            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"Leaderboard API failed for {time_period}: {e}")
            self._leaderboard_failed = True
            return []

    def fetch_quality_traders(self) -> List[dict]:
        """
        Fetch traders who are CONSISTENTLY profitable across multiple time windows.

        Scoring:
          - On ALL-TIME top 50 by PnL:  +1 point
          - On MONTHLY  top 50 by PnL:  +2 points (recent consistency matters more)
          - On WEEKLY   top 50 by PnL:  +1 point  (very recent form)

        Only traders with quality_score >= 2 are kept (must appear on at least
        2 leaderboards, or be on the monthly which counts double).

        This filters out:
          - All-time winners who are currently losing (score=1, filtered out)
          - Weekly flukes who got lucky on one bet (score=1, filtered out)
          - Consistent performers (score 2-4, kept and ranked)
        """
        if self._quality_traders:
            return self._quality_traders

        cache_key = 'quality_traders'
        cached = self.cache.get(cache_key, self.cache_ttl_leaderboard)
        if cached is not None:
            self._quality_traders = cached
            return cached

        # Pull all three time windows (3 API calls, each cached 1 hour)
        lb_all = self._fetch_leaderboard_period('ALL', 50)
        lb_month = self._fetch_leaderboard_period('MONTH', 50)
        lb_week = self._fetch_leaderboard_period('WEEK', 50)

        # Build address → trader info + quality score
        trader_map: Dict[str, dict] = {}

        for trader in lb_all:
            addr = trader['address'].lower()
            trader_map[addr] = {
                **trader,
                'quality_score': 1,
                'periods': ['ALL'],
                'all_time_profit': trader['profit'],
                'monthly_profit': 0,
                'weekly_profit': 0,
            }

        for trader in lb_month:
            addr = trader['address'].lower()
            if addr in trader_map:
                trader_map[addr]['quality_score'] += 2
                trader_map[addr]['periods'].append('MONTH')
                trader_map[addr]['monthly_profit'] = trader['profit']
            else:
                trader_map[addr] = {
                    **trader,
                    'quality_score': 2,
                    'periods': ['MONTH'],
                    'all_time_profit': 0,
                    'monthly_profit': trader['profit'],
                    'weekly_profit': 0,
                }

        for trader in lb_week:
            addr = trader['address'].lower()
            if addr in trader_map:
                trader_map[addr]['quality_score'] += 1
                trader_map[addr]['periods'].append('WEEK')
                trader_map[addr]['weekly_profit'] = trader['profit']
            else:
                trader_map[addr] = {
                    **trader,
                    'quality_score': 1,
                    'periods': ['WEEK'],
                    'all_time_profit': 0,
                    'monthly_profit': 0,
                    'weekly_profit': trader['profit'],
                }

        # Filter: quality_score >= 2 (must be on 2+ leaderboards or monthly)
        quality = [t for t in trader_map.values() if t['quality_score'] >= 2]
        quality.sort(key=lambda t: t['quality_score'], reverse=True)

        self._quality_traders = quality
        self.cache.set(cache_key, quality)

        logger.info(
            f"Quality trader filter: {len(lb_all)} all-time + {len(lb_month)} monthly + "
            f"{len(lb_week)} weekly → {len(quality)} quality traders "
            f"(score distribution: "
            f"4={sum(1 for t in quality if t['quality_score']==4)}, "
            f"3={sum(1 for t in quality if t['quality_score']==3)}, "
            f"2={sum(1 for t in quality if t['quality_score']==2)})"
        )

        return quality

    # ── Trader Positions ─────────────────────────────────────────────────────

    def _parse_positions_response(self, data, wallet_address: str, is_first: bool = False) -> List[dict]:
        """
        Parse positions API response robustly. Handles multiple response formats.
        Logs diagnostic info for the first trader to help debug format issues.
        """
        # Unwrap response envelope
        if isinstance(data, list):
            positions_list = data
        elif isinstance(data, dict):
            # Try common wrapper keys
            positions_list = (
                data.get('positions') or
                data.get('data') or
                data.get('history') or
                data.get('results') or
                []
            )
            if is_first and not positions_list:
                logger.info(
                    f"WHALE_DIAG: Dict response with no known list key. "
                    f"Top-level keys: {list(data.keys())[:15]}"
                )
                # Maybe the dict IS a single position? Check for position-like keys
                if 'conditionId' in data or 'size' in data or 'curVal' in data:
                    positions_list = [data]
        else:
            positions_list = []

        if is_first:
            logger.info(
                f"WHALE_DIAG: Response type={type(data).__name__}, "
                f"positions_list type={type(positions_list).__name__}, "
                f"count={len(positions_list) if isinstance(positions_list, list) else 'N/A'}"
            )

        if not isinstance(positions_list, list):
            if is_first:
                logger.warning(
                    f"WHALE_DIAG: positions_list is {type(positions_list).__name__}, not list. "
                    f"Value preview: {str(positions_list)[:200]}"
                )
            return []

        positions = []
        parse_failures = 0

        for i, pos in enumerate(positions_list):
            try:
                # Handle stringified JSON
                if isinstance(pos, str):
                    try:
                        pos = json.loads(pos)
                    except (json.JSONDecodeError, TypeError):
                        parse_failures += 1
                        continue
                if not isinstance(pos, dict):
                    parse_failures += 1
                    continue

                # Log first position's raw keys for diagnostics
                if is_first and i == 0:
                    logger.info(
                        f"WHALE_DIAG: First position keys: {sorted(pos.keys())}"
                    )
                    # Log conditionId-related fields specifically
                    logger.info(
                        f"WHALE_DIAG: conditionId={pos.get('conditionId', 'MISSING')}, "
                        f"condition_id={pos.get('condition_id', 'MISSING')}, "
                        f"asset type={type(pos.get('asset')).__name__}, "
                        f"asset={str(pos.get('asset', ''))[:80]}, "
                        f"slug={pos.get('slug', 'MISSING')}, "
                        f"market_slug={pos.get('market_slug', 'MISSING')}, "
                        f"eventSlug={pos.get('eventSlug', 'MISSING')}, "
                        f"title={pos.get('title', 'MISSING')[:60]}"
                    )

                # Extract conditionId using robust multi-path extractor
                condition_id = _extract_condition_id(pos)

                # Extract slug (try multiple field names)
                slug = (
                    pos.get('slug') or
                    pos.get('market_slug') or
                    pos.get('marketSlug') or
                    ''
                )

                # Extract event slug for cluster-level matching
                event_slug = pos.get('eventSlug', '')

                positions.append({
                    'conditionId': condition_id,
                    'market_slug': slug,
                    'event_slug': event_slug,
                    'title': pos.get('title', ''),
                    'size': safe_float(pos.get('size', pos.get('currentShares', 0))),
                    'avgPrice': safe_float(pos.get('avgPrice', pos.get('averagePrice', 0))),
                    'currentValue': safe_float(pos.get('curVal', pos.get('currentValue', 0))),
                    'cashPnl': safe_float(pos.get('cashPnl', 0)),
                    'percentPnl': safe_float(pos.get('percentPnl', 0)),
                    'outcome': pos.get('outcome', str(pos.get('outcomeIndex', ''))),
                })
            except (KeyError, ValueError, TypeError) as e:
                parse_failures += 1
                if is_first and i < 3:
                    logger.warning(f"WHALE_DIAG: Failed to parse position {i}: {e}")
                continue

        if is_first:
            cids_present = sum(1 for p in positions if p['conditionId'])
            slugs_present = sum(1 for p in positions if p['market_slug'])
            logger.info(
                f"WHALE_DIAG: Parsed {len(positions)} positions from {len(positions_list)} raw items "
                f"({parse_failures} parse failures). "
                f"conditionId present: {cids_present}/{len(positions)}, "
                f"slug present: {slugs_present}/{len(positions)}"
            )

        return positions

    def fetch_trader_positions(self, wallet_address: str, is_first: bool = False) -> List[dict]:
        """
        Fetch all open positions for a specific trader wallet.
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
                    'sizeThreshold': 0.01  # Lower threshold to catch more positions
                },
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()

            positions = self._parse_positions_response(data, wallet_address, is_first=is_first)

            if is_first and not positions:
                # Log raw response snippet for the first trader to diagnose empty results
                raw_str = json.dumps(data, default=str)[:500] if data else 'null/empty'
                logger.warning(
                    f"WHALE_DIAG: First trader {wallet_address[:10]}... returned 0 positions. "
                    f"HTTP {resp.status_code}, response preview: {raw_str}"
                )

            if not positions:
                self._positions_empty_count += 1

            self.cache.set(cache_key, positions)
            return positions

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                logger.warning("Rate limited on positions API, pausing")
                self._positions_failed = True
            else:
                logger.warning(f"Positions API failed for {wallet_address[:10]}...: {e}")
                if is_first:
                    logger.warning(f"WHALE_DIAG: HTTP error details: status={getattr(e.response, 'status_code', '?')}, body={getattr(e.response, 'text', '')[:300]}")
            return []
        except Exception as e:
            logger.warning(f"Positions API failed for {wallet_address[:10]}...: {e}")
            return []

    # ── Build Whale Index ────────────────────────────────────────────────────

    def build_whale_index(self) -> None:
        """
        Build an index of which markets quality traders are positioned in.
        Called once per engine run.

        Uses quality-filtered traders only (consistent performers).
        Maps condition_id → list of whale entries with quality metadata.

        Index keys (all normalized to lowercase):
          - conditionId (primary — hex string from positions API)
          - slug:{market_slug} (fallback for slug-based matching)
        """
        if self._index_built:
            return

        cache_key = 'whale_market_index_v3'  # Bumped version for new format
        cached = self.cache.get(cache_key, self.cache_ttl_positions)
        if cached is not None:
            self._whale_market_index = cached
            self._index_built = True
            logger.info(f"Loaded whale index from cache ({len(cached)} markets)")
            return

        traders = self.fetch_quality_traders()
        if not traders:
            self._index_built = True
            return

        index: Dict[str, List[dict]] = {}
        traders_fetched = 0
        total_positions = 0
        total_raw_positions = 0
        keys_by_type = {'conditionId': 0, 'slug': 0}  # Diagnostic counters
        self._positions_empty_count = 0

        for i, trader in enumerate(traders):
            address = trader.get('address', '')
            if not address:
                continue

            is_first = (i == 0)  # Enable detailed logging for first trader
            positions = self.fetch_trader_positions(address, is_first=is_first)
            traders_fetched += 1
            total_raw_positions += len(positions)

            for pos in positions:
                entry = {
                    'address': address,
                    'username': trader.get('username', ''),
                    'quality_score': trader.get('quality_score', 0),
                    'all_time_profit': trader.get('all_time_profit', 0),
                    'monthly_profit': trader.get('monthly_profit', 0),
                    'size': pos.get('size', 0),
                    'currentValue': pos.get('currentValue', 0),
                    'outcome': pos.get('outcome', ''),
                }

                indexed = False

                # Index under conditionId (primary key, already normalized)
                cid = pos.get('conditionId', '')
                if cid:
                    index.setdefault(cid, []).append(entry)
                    keys_by_type['conditionId'] += 1
                    indexed = True

                # Also index under slug (fallback key for matching)
                slug = pos.get('market_slug', '')
                if slug:
                    slug_key = f"slug:{slug}"
                    index.setdefault(slug_key, []).append(entry)
                    keys_by_type['slug'] += 1
                    indexed = True

                if indexed:
                    total_positions += 1

            if self._positions_failed:
                logger.warning(f"Rate limited after {traders_fetched} traders, using partial index")
                break

            # Brief pause between traders to avoid rate limiting
            if i > 0 and i % 10 == 0:
                time.sleep(0.5)

        self._whale_market_index = index
        self._index_built = True
        self.cache.set(cache_key, index)

        # Diagnostic: log sample keys so we can verify matching
        sample_cid_keys = [k for k in index.keys() if not k.startswith('slug:')][:5]
        sample_slug_keys = [k for k in index.keys() if k.startswith('slug:')][:3]
        logger.info(
            f"Built whale index: {traders_fetched} quality traders → "
            f"{total_raw_positions} raw positions → "
            f"{total_positions} indexed across {len(index)} index entries "
            f"(keys: conditionId={keys_by_type['conditionId']}, slug={keys_by_type['slug']}) "
            f"empty_responses={self._positions_empty_count}/{traders_fetched} "
            f"sample_cids={sample_cid_keys} "
            f"sample_slugs={sample_slug_keys}"
        )

    # ── Smart Money Score ────────────────────────────────────────────────────

    def compute_smart_money_score(self, market_id: str, token_id: str, market_slug: str = '') -> dict:
        """
        Compute smart money score based on quality whale presence in a market.

        Sub-scores:
          whale_accumulation (0-10):
            How many quality traders are in this market, weighted by their
            quality_score. A score-4 trader (all three leaderboards) counts
            more than a score-2 trader (just monthly).

          leaderboard_alignment (0-8):
            Are the highest-quality traders (score 3+) here?
            Measures conviction from the best performers.

          position_concentration (0-7):
            How much capital do whales have concentrated here?
            Bonus for large absolute capital (>$10k, >$50k).

        Returns dict with sub-scores and smart_money_total (0-25).
        """
        try:
            self.build_whale_index()

            # Normalize the lookup key to match index format
            normalized_id = _normalize_condition_id(market_id)

            # Try conditionId first (normalized), fall back to slug-based lookup
            whales = self._whale_market_index.get(normalized_id, [])
            if not whales and market_slug:
                whales = self._whale_market_index.get(f"slug:{market_slug}", [])

            if not whales:
                return {
                    'whale_accumulation': 0.0,
                    'leaderboard_alignment': 0.0,
                    'position_concentration': 0.0,
                    'smart_money_total': 0.0
                }

            # ── Whale Accumulation (0-10) ──
            # Weight by quality: score-4 trader = 4 points, score-2 = 2 points
            # Normalise against max possible (if all 50 traders had score 4 = 200)
            unique_whales = {}
            for w in whales:
                addr = w['address'].lower()
                if addr not in unique_whales or w['quality_score'] > unique_whales[addr]:
                    unique_whales[addr] = w['quality_score']

            weighted_count = sum(unique_whales.values())
            # Scale: realistically, 3-5 quality traders in one market is strong
            # 1 whale (q=3) → 1.5pts, 3 whales (avg q=3) → 4.5pts, 8 whales → 10pts
            whale_accumulation = min(10.0, weighted_count * 0.5)

            # ── Leaderboard Alignment (0-8) ──
            # Only count high-quality traders (score >= 3 = on 2+ leaderboards
            # with at least one being monthly or ALL+WEEK)
            elite_count = sum(1 for q in unique_whales.values() if q >= 3)
            # 1 elite = 2pts, 2 elites = 4pts, 4+ elites = 8pts
            leaderboard_alignment = min(8.0, elite_count * 2.0)

            # ── Position Concentration (0-7) ──
            total_whale_value = sum(w.get('currentValue', 0) for w in whales)
            position_concentration = 0.0

            if total_whale_value > 0 and len(whales) >= 2:
                sorted_whales = sorted(whales, key=lambda w: w.get('currentValue', 0), reverse=True)
                top_3_value = sum(w.get('currentValue', 0) for w in sorted_whales[:3])
                conc_pct = (top_3_value / total_whale_value) * 100

                if conc_pct > 50:
                    position_concentration = min(4.0, (conc_pct - 50) / 12.5)

            # Capital size bonus (whales putting serious money here)
            if total_whale_value > 100000:
                position_concentration = min(7.0, position_concentration + 3.0)
            elif total_whale_value > 50000:
                position_concentration = min(7.0, position_concentration + 2.0)
            elif total_whale_value > 10000:
                position_concentration = min(7.0, position_concentration + 1.0)

            smart_money_total = whale_accumulation + leaderboard_alignment + position_concentration

            return {
                'whale_accumulation': round(whale_accumulation, 2),
                'leaderboard_alignment': round(leaderboard_alignment, 2),
                'position_concentration': round(position_concentration, 2),
                'smart_money_total': round(min(25.0, smart_money_total), 2),
                'whale_count': len(unique_whales),
                'elite_count': elite_count,
                'total_whale_value': round(total_whale_value, 2),
            }

        except Exception as e:
            logger.error(f"Error computing smart money score for {market_id}: {e}")
            return {
                'whale_accumulation': 0.0,
                'leaderboard_alignment': 0.0,
                'position_concentration': 0.0,
                'smart_money_total': 0.0
            }
