"""
Composite Scorer — Combines all 4 layers into a single edge score.
Ranks opportunities and filters by convexity band and liquidity.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

from src.signals import (
    compute_structural_score,
    compute_dislocation_score,
    classify_convexity,
)
from src.utils import safe_float

logger = logging.getLogger(__name__)


def compute_edge_score(
    token: dict,
    market: dict,
    price_history: List[dict],
    kalshi_price: Optional[float],
    smart_money: dict,
    external: dict,
    config: dict
) -> dict:
    """
    Compute the composite edge score for a market token across all 4 layers.

    Parameters:
        token: {token_id, outcome, current_price, bid, ask, bid_depth, ask_depth, spread_pct}
        market: {market_id, question, tokens, volume_24h, volume_total, liquidity, resolution_date}
        price_history: [{timestamp, price}, ...]
        kalshi_price: matched Kalshi price or None
        smart_money: output from WhaleTracker.compute_smart_money_score()
        external: output from NewsSignals.compute_external_score()
        config: scoring section of config.json

    Returns: dict with layer scores, composite edge score (0-100), and classification
    """
    current_price = safe_float(token.get('current_price', 0))
    weights = config.get('scoring', {}).get('layer_weights', {
        'structural': 0.30, 'smart_money': 0.25,
        'dislocation': 0.30, 'external': 0.15
    })

    # ── Layer 1: Structural (0-30) ──
    structural = compute_structural_score(
        tokens=market.get('tokens', []),
        poly_price=current_price,
        kalshi_price=kalshi_price
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
    # Each layer normalised to its max, weighted, then rescaled
    # so the score reflects only layers that have data
    max_structural = 30.0
    max_smart_money = 25.0
    max_dislocation = 30.0
    max_external = 15.0

    structural_norm = structural['structural_total'] / max_structural
    smart_money_norm = smart_money_total / max_smart_money
    dislocation_norm = dislocation['dislocation_total'] / max_dislocation
    external_norm = external_total / max_external

    # Track which layers have data so we can rescale
    layers = [
        (structural_norm, weights.get('structural', 0.30), structural['structural_total'] > 0),
        (smart_money_norm, weights.get('smart_money', 0.25), smart_money_total > 0),
        (dislocation_norm, weights.get('dislocation', 0.30), dislocation['dislocation_total'] > 0),
        (external_norm, weights.get('external', 0.15), external_total > 0),
    ]

    active_weight = sum(w for _, w, active in layers if active)
    total_weight = sum(w for _, w, _ in layers)

    weighted = sum(norm * w for norm, w, _ in layers)

    # Rescale: if only 60% of weight has data, scale up proportionally
    # so a perfect score on active layers still reaches ~100
    if active_weight > 0:
        weighted = weighted * (total_weight / active_weight)

    # Scale to 0-100
    edge_score = round(min(100.0, weighted * 100.0), 1)

    # ── Classification ──
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
        'days_to_close': days_to_close
    }


def _days_to_close(resolution_date_str: Optional[str]) -> int:
    """Parse resolution date and return days until close."""
    if not resolution_date_str:
        return 999

    try:
        res = datetime.fromisoformat(str(resolution_date_str))
        if res.tzinfo is None:
            res = res.replace(tzinfo=timezone.utc)
        delta = (res - datetime.now(timezone.utc)).days
        return max(0, delta)
    except (ValueError, TypeError):
        return 999


def rank_opportunities(
    scored_items: List[dict],
    config: dict
) -> List[dict]:
    """
    Filter and rank all scored market tokens into a list of opportunities.

    Filters:
    - Edge score above threshold
    - Minimum liquidity
    - Minimum days to close
    - Valid convexity band (not 'invalid')
    - Only 5x and 10x bands (per fund structure)

    Returns sorted list, highest edge score first.
    """
    threshold = config.get('scoring', {}).get('edge_threshold', 50)
    min_liquidity = config.get('trading', {}).get('min_liquidity_usd', 500)
    min_days = config.get('trading', {}).get('min_days_to_close', 2)
    max_days = config.get('trading', {}).get('max_days_to_close', 999)

    # Bands we're trading (collect from all funds)
    active_bands = set()
    for fund_key in ['fund_a', 'fund_b']:
        fund = config.get('funds', {}).get(fund_key, {})
        for b in fund.get('bands', [fund.get('band', '')]):
            active_bands.add(b)

    opportunities = []

    for item in scored_items:
        scores = item.get('scores', {})
        market = item.get('market', {})
        token = item.get('token', {})

        edge = scores.get('edge_score', 0)
        if edge < threshold:
            continue

        liquidity = safe_float(market.get('liquidity', 0))
        if liquidity < min_liquidity:
            continue

        days = scores.get('days_to_close', 999)
        if days < min_days:
            continue

        # Skip markets that settle too far in the future
        if days > max_days:
            continue

        convexity = scores.get('convexity', {})
        band = convexity.get('band', 'invalid')

        if band == 'invalid':
            continue

        # Determine which fund this belongs to (if any)
        fund_assignment = None
        for fund_key in ['fund_a', 'fund_b']:
            fund = config.get('funds', {}).get(fund_key, {})
            fund_bands = set(fund.get('bands', [fund.get('band', '')]))
            if band in fund_bands:
                fund_assignment = fund_key
                break

        opportunities.append({
            'market_id': market.get('market_id', ''),
            'question': market.get('question', ''),
            'category': market.get('category', 'other'),
            'slug': market.get('slug', ''),
            'token_id': token.get('token_id', ''),
            'outcome': token.get('outcome', ''),
            'current_price': safe_float(token.get('current_price', 0)),
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
            'liquidity_usd': liquidity,
            'volume_24h': safe_float(market.get('volume_24h', 0)),
            'fund_assignment': fund_assignment,
            'recommended_action': 'BUY',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    opportunities.sort(key=lambda x: x['edge_score'], reverse=True)

    logger.info(
        f"Ranked {len(opportunities)} opportunities "
        f"(threshold={threshold}, bands={active_bands})"
    )

    return opportunities
