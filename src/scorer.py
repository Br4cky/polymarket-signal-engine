"""
Composite Scorer — Combines all 4 layers into a single edge score.
Ranks opportunities by mispricing magnitude. No band routing — all markets
compete in a single pool.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

from src.signals import (
    compute_structural_score,
    compute_dislocation_score,
    classify_convexity,
    compute_market_age_bonus,
)
from src.event_clustering import extract_base_event, are_conflicting, extract_underlying_asset, extract_direction
from src.utils import safe_float

logger = logging.getLogger(__name__)


def compute_edge_score(
    token: dict,
    market: dict,
    price_history: List[dict],
    kalshi_price: Optional[float],
    smart_money: dict,
    external: dict,
    config: dict,
    manifold_prob: Optional[float] = None,
    created_at: Optional[str] = None
) -> dict:
    """
    Compute the composite edge score for a market token across all 4 layers.
    Returns: dict with layer scores, composite edge score (0-100), and classification.
    """
    current_price = safe_float(token.get('current_price', 0))

    weights = config.get('scoring', {}).get('layer_weights', {
        'structural': 0.25, 'smart_money': 0.15,
        'dislocation': 0.50, 'external': 0.10
    })

    # ── Layer 1: Structural (0-30) ──
    structural = compute_structural_score(
        tokens=market.get('tokens', []),
        poly_price=current_price,
        kalshi_price=kalshi_price,
        manifold_prob=manifold_prob
    )

    # ── Layer 2: Smart Money (0-25) ──
    smart_money_total = safe_float(smart_money.get('smart_money_total', 0))

    # ── Layer 3: Dislocation (0-30) ──
    days_to_close = _days_to_close(market.get('resolution_date'))
    dislocation = compute_dislocation_score(
        current_price=current_price,
        price_history=price_history,
        volume_24h=safe_float(market.get('volume_24h', 0)),
        volume_total=safe_float(market.get('volume_total', 0)),
        bid=safe_float(token.get('bid', current_price - 0.01)),
        ask=safe_float(token.get('ask', current_price + 0.01)),
        bid_depth=safe_float(token.get('bid_depth', 0)),
        ask_depth=safe_float(token.get('ask_depth', 0)),
        days_to_close=days_to_close
    )

    # ── Layer 4: External (0-15) ──
    external_total = safe_float(external.get('external_total', 0))

    # ── Composite Score ──
    max_structural = 30.0
    max_smart_money = 25.0
    max_dislocation = 30.0
    max_external = 15.0

    structural_norm = structural['structural_total'] / max_structural
    smart_money_norm = smart_money_total / max_smart_money
    dislocation_norm = dislocation['dislocation_total'] / max_dislocation
    external_norm = external_total / max_external

    layers = [
        (structural_norm, weights.get('structural', 0.25), structural['structural_total'] > 0),
        (smart_money_norm, weights.get('smart_money', 0.15), smart_money_total > 0),
        (dislocation_norm, weights.get('dislocation', 0.50), dislocation['dislocation_total'] > 0),
        (external_norm, weights.get('external', 0.10), external_total > 0),
    ]

    active_weight = sum(w for _, w, active in layers if active)
    total_weight = sum(w for _, w, _ in layers)
    weighted = sum(norm * w for norm, w, _ in layers)

    if active_weight > 0:
        weighted = weighted * (total_weight / active_weight)

    edge_score = round(min(100.0, weighted * 100.0), 1)

    # ── Market Age Bonus (repricing strategy) ──
    age_bonus = compute_market_age_bonus(created_at)
    edge_score = round(min(100.0, edge_score + age_bonus), 1)

    # ── Classification (metadata only — does NOT drive trading decisions) ──
    convexity = classify_convexity(current_price, token.get('outcome', 'YES'))

    return {
        'edge_score': edge_score,
        'layer_scores': {
            'structural': structural['structural_total'],
            'smart_money': round(smart_money_total, 2),
            'dislocation': dislocation['dislocation_total'],
            'external': round(external_total, 2)
        },
        'structural_detail': structural,
        'smart_money_detail': smart_money,
        'dislocation_detail': dislocation,
        'external_detail': external,
        'convexity': convexity,
        'days_to_close': days_to_close,
        'age_bonus': age_bonus
    }


def _days_to_close(resolution_date_str: Optional[str]) -> int:
    """Parse resolution date and return days until close.
    Returns 0 for markets whose scheduled resolution date has passed but
    are still actively trading (e.g. a football match still in progress).
    The Gamma API already filters active=true / closed=false, so if we
    see the market it is still live — don't filter it out."""
    if not resolution_date_str:
        return 999

    try:
        res = datetime.fromisoformat(str(resolution_date_str))
        if res.tzinfo is None:
            res = res.replace(tzinfo=timezone.utc)
        delta = (res - datetime.now(timezone.utc)).days
        return max(0, delta)  # clamp to 0 — market still live if Gamma says so
    except (ValueError, TypeError):
        return 999


def rank_opportunities(
    scored_items: List[dict],
    config: dict
) -> List[dict]:
    """
    Filter and rank all scored market tokens into tradeable opportunities.

    Single pool — no band routing. All markets above edge threshold with
    sufficient liquidity enter the pool, ranked by edge score.
    """
    threshold = config.get('scoring', {}).get('edge_threshold', 20)
    min_liquidity = config.get('trading', {}).get('min_liquidity_usd', 500)
    min_days = config.get('trading', {}).get('min_days_to_close', 1)
    max_days = config.get('trading', {}).get('max_days_to_close', 20)

    opportunities = []

    for item in scored_items:
        scores = item.get('scores', {})
        market = item.get('market', {})
        token = item.get('token', {})

        edge = scores.get('edge_score', 0)
        layer_scores = scores.get('layer_scores', {})

        # ── Dynamic threshold by layer composition ──
        effective_threshold = threshold
        if edge > 0:
            dislocation_pct = layer_scores.get('dislocation', 0) / edge * 100
            structural_score = layer_scores.get('structural', 0)

            if dislocation_pct > 65:
                effective_threshold = max(threshold, threshold * 1.30)
            elif structural_score > 8:
                effective_threshold = max(threshold * 0.80, 10)

        if edge < effective_threshold:
            continue

        liquidity = safe_float(market.get('liquidity', 0))
        if liquidity < min_liquidity:
            continue

        days = scores.get('days_to_close', 999)
        if days < min_days:
            continue
        if days > max_days:
            continue

        # Convexity band is metadata only — still useful for dashboard display
        convexity = scores.get('convexity', {})
        band = convexity.get('band', 'unknown')
        if band == 'invalid':
            continue

        current_price = safe_float(token.get('current_price', 0))

        opportunities.append({
            'market_id': market.get('market_id', ''),
            'question': market.get('question', ''),
            'category': market.get('category', 'other'),
            'slug': market.get('slug', ''),
            'token_id': token.get('token_id', ''),
            'outcome': token.get('outcome', ''),
            'current_price': current_price,
            'bid': safe_float(token.get('bid', 0)),
            'ask': safe_float(token.get('ask', 0)),
            'spread_pct': safe_float(token.get('spread_pct', 0)),
            'edge_score': edge,
            'layer_scores': scores.get('layer_scores', {}),
            'structural_detail': scores.get('structural_detail', {}),
            'dislocation_detail': scores.get('dislocation_detail', {}),
            'smart_money_detail': scores.get('smart_money_detail', {}),
            'external_detail': scores.get('external_detail', {}),
            'convexity_band': band,
            'potential_multiple': convexity.get('potential_multiple', 0),
            'days_to_close': days,
            'resolution_date': market.get('resolution_date', ''),
            'liquidity_usd': liquidity,
            'volume_24h': safe_float(market.get('volume_24h', 0)),
            'recommended_action': 'BUY',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    opportunities.sort(key=lambda x: x['edge_score'], reverse=True)

    # ── Conflict filter ──
    # Remove opportunities that would create opposing bets on the same asset.
    pre_filter_count = len(opportunities)

    asset_directions = {}
    for opp in opportunities:
        question = opp.get('question', '')
        opp['base_event'] = extract_base_event(question, opp.get('slug', ''))
        asset = extract_underlying_asset(question)
        if not asset:
            continue
        direction = extract_direction(question)
        if direction == 'neutral':
            continue
        asset_directions.setdefault(asset, {}).setdefault(direction, []).append(opp)

    blocked_opps = set()
    for asset, dirs in asset_directions.items():
        if 'bullish' in dirs and 'bearish' in dirs:
            bull_best = max(o['edge_score'] for o in dirs['bullish'])
            bear_best = max(o['edge_score'] for o in dirs['bearish'])
            loser = 'bearish' if bull_best >= bear_best else 'bullish'
            for opp in dirs[loser]:
                blocked_opps.add(id(opp))
            logger.info(
                f"Conflict filter: {asset} has bullish (best={bull_best:.1f}) vs "
                f"bearish (best={bear_best:.1f}) — dropping {loser} "
                f"({len(dirs[loser])} opps)"
            )

    if blocked_opps:
        opportunities = [o for o in opportunities if id(o) not in blocked_opps]

    if pre_filter_count != len(opportunities):
        logger.info(
            f"Conflict filter: {pre_filter_count} → {len(opportunities)} opportunities"
        )

    logger.info(
        f"Ranked {len(opportunities)} opportunities (threshold={threshold})"
    )

    return opportunities
