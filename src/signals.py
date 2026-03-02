"""
Signal Computations — All 4 layers of mispricing detection.

Layer 1: Structural (0-30 pts) — combinatorial check + cross-platform divergence
Layer 2: Smart Money (0-25 pts) — whale tracking + leaderboard + concentration
Layer 3: Price Dislocation (0-30 pts) — velocity, volume, order book, trajectory, time
Layer 4: External (0-15 pts) — news relevance + Google Trends
"""

import math
import logging
from typing import List, Dict, Optional

from src.utils import mean, std_dev, z_score, safe_float

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1 — STRUCTURAL MISPRICING (0-30 points)
# ═══════════════════════════════════════════════════════════════════════════

def compute_combinatorial_mispricing(tokens: List[dict]) -> dict:
    """
    Check if YES + NO prices sum to != $1.00.
    If sum < 0.97 or sum > 1.03, there's structural mispricing.

    Returns: {score: float (0-15), yes_no_sum: float, mispricing_type: str}
    """
    if len(tokens) < 2:
        return {'score': 0, 'yes_no_sum': 0, 'mispricing_type': 'none'}

    # Sum all outcome prices
    total = sum(safe_float(t.get('current_price', 0)) for t in tokens)

    score = 0.0
    mispricing_type = 'none'

    if total < 0.97:
        # Can buy all outcomes for less than $1 → guaranteed profit
        gap = 0.97 - total
        score = min(15.0, gap * 500)
        mispricing_type = 'underpriced'
    elif total > 1.03:
        # All outcomes overpriced → can sell short
        gap = total - 1.03
        score = min(15.0, gap * 500)
        mispricing_type = 'overpriced'

    return {
        'score': round(score, 2),
        'yes_no_sum': round(total, 4),
        'mispricing_type': mispricing_type
    }


def compute_cross_platform_divergence(
    poly_price: float,
    kalshi_price: Optional[float]
) -> dict:
    """
    Compare Polymarket price vs Kalshi price for the same event.
    Divergence > 3% = signal.

    Returns: {score: float (0-15), divergence_pct: float, direction: str}
    """
    if kalshi_price is None or kalshi_price <= 0:
        return {'score': 0, 'divergence_pct': 0, 'direction': 'none'}

    divergence = abs(poly_price - kalshi_price)
    direction = 'poly_higher' if poly_price > kalshi_price else 'kalshi_higher'

    if divergence < 0.03:
        score = 0.0
    else:
        score = min(15.0, divergence * 300)

    return {
        'score': round(score, 2),
        'divergence_pct': round(divergence * 100, 2),
        'direction': direction
    }


def compute_structural_score(
    tokens: List[dict],
    poly_price: float,
    kalshi_price: Optional[float] = None
) -> dict:
    """
    Combined Layer 1 score (0-30 points).
    """
    combinatorial = compute_combinatorial_mispricing(tokens)
    cross_platform = compute_cross_platform_divergence(poly_price, kalshi_price)

    total = combinatorial['score'] + cross_platform['score']

    return {
        'structural_total': round(min(30.0, total), 2),
        'combinatorial': combinatorial,
        'cross_platform': cross_platform
    }


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2 — SMART MONEY (0-25 points)
# Computed in whale_tracker.py — this module just provides the interface
# ═══════════════════════════════════════════════════════════════════════════

# Smart money scores come directly from WhaleTracker.compute_smart_money_score()
# which returns {whale_accumulation: 0-10, leaderboard_alignment: 0-8,
#                position_concentration: 0-7, smart_money_total: 0-25}


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3 — PRICE DISLOCATION (0-30 points)
# ═══════════════════════════════════════════════════════════════════════════

def compute_price_velocity(
    current_price: float,
    price_history: List[dict],
    lookback_hours: int = 24
) -> float:
    """
    Price Velocity (0-8 points).
    Measures how fast price moved recently vs historical volatility.
    A 3+ std dev move in 24h suggests overreaction.
    """
    if len(price_history) < 48:
        return 0.0

    prices = [safe_float(p.get('price', 0)) for p in price_history]
    prices = [p for p in prices if p > 0.001]

    if len(prices) < 48:
        return 0.0

    recent_count = min(lookback_hours, len(prices) - 1)
    price_start = prices[-(recent_count + 1)]
    if price_start < 0.001:
        return 0.0

    recent_return = (current_price - price_start) / price_start

    # Historical hourly returns for volatility
    hourly_returns = []
    lookback = min(168, len(prices) - 1)  # 7 days
    for i in range(lookback):
        idx = -(i + 1)
        prev_idx = -(i + 2)
        if abs(prev_idx) <= len(prices) and prices[prev_idx] > 0.001:
            ret = (prices[idx] - prices[prev_idx]) / prices[prev_idx]
            hourly_returns.append(ret)

    if not hourly_returns:
        return 0.0

    vol = std_dev(hourly_returns)
    if vol < 1e-10:
        return 0.0

    daily_vol = vol * math.sqrt(24)
    velocity_z = abs(recent_return) / max(0.001, daily_vol)

    return round(min(8.0, max(0.0, velocity_z * 2.0)), 2)


def compute_volume_anomaly(volume_24h: float, volume_total: float) -> float:
    """
    Volume Anomaly (0-7 points).
    Current 24h volume vs estimated daily average.
    3-5x spike = informed trading or panic.
    """
    if volume_total <= 0:
        return 0.0

    # Rough daily average from total volume (assume market ~30 days old)
    avg_daily = volume_total / 30.0
    if avg_daily < 10:
        return 0.0

    ratio = volume_24h / avg_daily

    if ratio <= 1.0:
        return 0.0
    elif ratio <= 2.0:
        score = (ratio - 1.0) * 2.0
    elif ratio <= 5.0:
        score = 2.0 + (ratio - 2.0) * 1.67
    else:
        score = 7.0

    return round(min(7.0, max(0.0, score)), 2)


def compute_order_book_score(
    bid: float,
    ask: float,
    bid_depth: float,
    ask_depth: float
) -> float:
    """
    Order Book Analysis (0-5 points).
    Wide spreads + thin books = mispricing opportunity.
    But also need enough liquidity to actually trade.
    """
    mid = (bid + ask) / 2.0
    if mid < 0.001:
        return 0.0

    spread_pct = ((ask - bid) / mid) * 100.0

    # Spread component: moderate spreads (1-5%) are interesting
    # Too tight = well-priced, too wide = untradeable
    if spread_pct < 0.5:
        spread_score = 1.0  # Very efficient, less opportunity
    elif spread_pct < 2.0:
        spread_score = 2.5  # Some room for edge
    elif spread_pct < 5.0:
        spread_score = 2.0  # Wider but still tradeable
    elif spread_pct < 10.0:
        spread_score = 1.0  # Getting thin
    else:
        spread_score = 0.0  # Untradeable

    # Depth component
    total_depth = bid_depth + ask_depth
    if total_depth > 5000:
        depth_score = 2.5
    elif total_depth > 1000:
        depth_score = 2.0
    elif total_depth > 100:
        depth_score = 1.0
    else:
        depth_score = 0.0

    return round(min(5.0, spread_score + depth_score), 2)


def compute_price_trajectory(
    current_price: float,
    price_history: List[dict]
) -> float:
    """
    Price Trajectory (0-5 points).
    How far current price is from 30-day mean.
    Large deviations suggest mean reversion opportunity.
    """
    if len(price_history) < 24:
        return 0.0

    prices = [safe_float(p.get('price', 0)) for p in price_history]
    prices = [p for p in prices if p > 0.001]

    if len(prices) < 24:
        return 0.0

    hist_mean = mean(prices)
    hist_std = std_dev(prices)

    if hist_std < 0.001:
        return 0.0

    deviation = abs(current_price - hist_mean) / hist_std

    return round(min(5.0, max(0.0, deviation * 1.5)), 2)


def compute_time_decay(days_to_close: int) -> float:
    """
    Time Decay (0-5 points).
    Markets approaching resolution see more volatility = more opportunity.
    """
    if days_to_close < 0:
        return 0.0
    elif days_to_close <= 3:
        return 5.0
    elif days_to_close <= 7:
        return 4.0
    elif days_to_close <= 14:
        return 3.0
    elif days_to_close <= 30:
        return 1.0
    else:
        return 0.0


def compute_dislocation_score(
    current_price: float,
    price_history: List[dict],
    volume_24h: float,
    volume_total: float,
    bid: float,
    ask: float,
    bid_depth: float,
    ask_depth: float,
    days_to_close: int
) -> dict:
    """
    Combined Layer 3 score (0-30 points).
    """
    velocity = compute_price_velocity(current_price, price_history)
    volume = compute_volume_anomaly(volume_24h, volume_total)
    order_book = compute_order_book_score(bid, ask, bid_depth, ask_depth)
    trajectory = compute_price_trajectory(current_price, price_history)
    time_decay = compute_time_decay(days_to_close)

    total = velocity + volume + order_book + trajectory + time_decay

    return {
        'dislocation_total': round(min(30.0, total), 2),
        'price_velocity': velocity,
        'volume_anomaly': volume,
        'order_book': order_book,
        'price_trajectory': trajectory,
        'time_decay': time_decay
    }


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 4 — EXTERNAL (0-15 points)
# Computed in news_signals.py — returns {external_total, news_score, trends_score}
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# CONVEXITY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def classify_convexity(current_price: float, outcome: str) -> dict:
    """
    Classify a token by its payout potential.

    For YES tokens at low prices → high multiple if they win.
    For NO tokens, invert the logic.

    Returns: {band: str, potential_multiple: float, probability_implied: float}
    """
    # Price IS the implied probability of THIS outcome winning,
    # regardless of YES/NO.  A NO token at 0.10 pays 10x if correct,
    # just like a YES token at 0.10.
    prob = current_price

    if prob < 0.001:
        return {'band': 'invalid', 'potential_multiple': 0, 'probability_implied': 0}

    multiple = 1.0 / prob

    if prob <= 0.05:
        band = '20x'
    elif prob <= 0.12:
        band = '10x'
    elif prob <= 0.25:
        band = '5x'
    elif prob <= 0.50:
        band = '2x'
    else:
        band = 'yield'

    return {
        'band': band,
        'potential_multiple': round(multiple, 1),
        'probability_implied': round(prob * 100, 2)
    }
