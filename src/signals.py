"""
Signal Computations — All 4 layers of mispricing detection.

Layer 1: Structural (0-30 pts) — combinatorial check + cross-platform divergence (Kalshi + Manifold)
Layer 2: Smart Money (0-25 pts) — whale tracking + leaderboard + concentration
Layer 3: Price Dislocation (0-30 pts) — velocity, volume, order book, trajectory, time
Layer 4: External (0-15 pts) — GDELT news + Wikipedia pageviews + Fear/Greed + comment velocity
"""

import math
import logging
from datetime import datetime, timezone
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

    # Graduated scoring: even small deviations from 1.0 are a signal
    # Perfect efficiency = sum of 1.00. Spread causes ~0.01-0.02 deviation.
    # Anything beyond spread (>0.02) is interesting; >0.05 is strong.
    deviation = abs(total - 1.0)

    if total < 1.0:
        mispricing_type = 'underpriced'
    elif total > 1.0:
        mispricing_type = 'overpriced'

    if deviation > 0.02:
        # Graduated: 0.02-0.05 = 1-4pts, 0.05-0.10 = 4-8pts, >0.10 = 8-15pts
        score = min(15.0, (deviation - 0.02) * 150)
    else:
        mispricing_type = 'none'

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


def compute_manifold_divergence(
    poly_price: float,
    manifold_prob: Optional[float]
) -> dict:
    """
    Compare Polymarket price vs Manifold Markets probability.
    Manifold is a play-money prediction market with decent calibration.
    Divergence > 5% = signal (wider threshold than Kalshi since different
    market type and liquidity profile).

    Args:
        poly_price: Polymarket YES token price (0-1)
        manifold_prob: Manifold probability (0-1) or None

    Returns: {score: float (0-10), divergence_pct: float, direction: str}
    """
    if manifold_prob is None or manifold_prob <= 0:
        return {'score': 0, 'divergence_pct': 0, 'direction': 'none'}

    divergence = abs(poly_price - manifold_prob)
    direction = 'poly_higher' if poly_price > manifold_prob else 'manifold_higher'

    if divergence < 0.05:
        score = 0.0
    else:
        # 5-10% = 1-5pts, 10-20% = 5-8pts, >20% = 8-10pts
        score = min(10.0, (divergence - 0.05) * 60)

    return {
        'score': round(score, 2),
        'divergence_pct': round(divergence * 100, 2),
        'direction': direction
    }


def compute_structural_score(
    tokens: List[dict],
    poly_price: float,
    kalshi_price: Optional[float] = None,
    manifold_prob: Optional[float] = None
) -> dict:
    """
    Combined Layer 1 score (0-30 points).

    Sub-signals:
      - Combinatorial mispricing (0-15): YES+NO != $1.00
      - Cross-platform Kalshi (0-15): Polymarket vs Kalshi divergence
      - Cross-platform Manifold (0-10): Polymarket vs Manifold divergence

    Note: total is capped at 30 even though sub-signals sum to 40 max.
    This means multiple cross-platform divergences compound nicely.
    """
    combinatorial = compute_combinatorial_mispricing(tokens)
    cross_platform = compute_cross_platform_divergence(poly_price, kalshi_price)
    manifold = compute_manifold_divergence(poly_price, manifold_prob)

    total = combinatorial['score'] + cross_platform['score'] + manifold['score']

    return {
        'structural_total': round(min(30.0, total), 2),
        'combinatorial': combinatorial,
        'cross_platform': cross_platform,
        'manifold': manifold
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
    Price Velocity (0-10 points).
    Measures how fast price moved recently vs historical volatility.
    A 2+ std dev move in 24h suggests overreaction or new information.

    Scoring (z-score based):
        < 1.0σ  → 0 pts  (normal movement)
        1.0-1.5σ → 2 pts  (notable)
        1.5-2.0σ → 4 pts  (unusual)
        2.0-3.0σ → 6-8 pts (significant dislocation)
        3.0+σ    → 10 pts (extreme)
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

    # Stepped scoring for better differentiation
    if velocity_z < 1.0:
        score = 0.0
    elif velocity_z < 1.5:
        score = 2.0 + (velocity_z - 1.0) * 4.0  # 2-4
    elif velocity_z < 2.0:
        score = 4.0 + (velocity_z - 1.5) * 4.0  # 4-6
    elif velocity_z < 3.0:
        score = 6.0 + (velocity_z - 2.0) * 2.0  # 6-8
    else:
        score = 8.0 + min(2.0, (velocity_z - 3.0))  # 8-10

    return round(min(10.0, max(0.0, score)), 2)


def compute_volume_anomaly(volume_24h: float, volume_total: float) -> float:
    """
    Volume Anomaly (0-8 points).
    Current 24h volume vs estimated daily average.
    A spike suggests informed trading or a catalyst event.

    Scoring (ratio of 24h vol to avg daily):
        ≤ 1.5x → 0 pts  (normal)
        2x     → 2 pts  (elevated)
        3x     → 4 pts  (notable spike)
        5x     → 6 pts  (significant)
        8x+    → 8 pts  (extreme)
    """
    if volume_total <= 0 or volume_24h <= 0:
        return 0.0

    # Estimate avg daily volume. We don't know the exact market age,
    # so use a conservative 30-day assumption. If total volume is
    # very small (< $5k), the data is too thin to be meaningful.
    avg_daily = volume_total / 30.0

    if avg_daily < 10:
        return 0.0

    ratio = volume_24h / avg_daily

    if ratio <= 1.5:
        return 0.0
    elif ratio <= 3.0:
        score = (ratio - 1.5) / 1.5 * 4.0  # 0-4
    elif ratio <= 5.0:
        score = 4.0 + (ratio - 3.0) / 2.0 * 2.0  # 4-6
    elif ratio <= 8.0:
        score = 6.0 + (ratio - 5.0) / 3.0 * 2.0  # 6-8
    else:
        score = 8.0

    return round(min(8.0, max(0.0, score)), 2)


def compute_order_book_score(
    bid: float,
    ask: float,
    bid_depth: float,
    ask_depth: float
) -> float:
    """
    Order Book Imbalance (0-5 points).
    Scores DIRECTIONAL PRESSURE — not market existence.

    A balanced order book (50/50 depth) scores 0 — that's normal.
    A skewed book (e.g. 80% depth on bid side) suggests smart money
    is loading up on one side, which is actual mispricing evidence.

    Sub-components:
      - Depth imbalance (0-3): how skewed is bid vs ask depth
      - Spread stress (0-2): abnormally wide spreads suggest
        market makers are uncertain (potential catalyst/repricing)

    Normal spreads + balanced depth = 0 (not evidence of mispricing).
    """
    mid = (bid + ask) / 2.0
    if mid < 0.001:
        return 0.0

    total_depth = bid_depth + ask_depth
    if total_depth < 50:
        return 0.0  # Not enough data to assess

    # ── Depth imbalance (0-3) ──
    # Ratio of larger side to total. 50% = balanced, 80%+ = strong imbalance
    larger_side = max(bid_depth, ask_depth)
    imbalance_ratio = larger_side / total_depth  # 0.5 to 1.0

    if imbalance_ratio < 0.60:
        imbalance_score = 0.0  # Balanced — no signal
    elif imbalance_ratio < 0.70:
        imbalance_score = 1.0  # Mild lean
    elif imbalance_ratio < 0.80:
        imbalance_score = 2.0  # Notable imbalance
    else:
        imbalance_score = 3.0  # Heavy one-sided pressure

    # ── Spread stress (0-2) ──
    # Only score WIDE spreads — they indicate market maker uncertainty
    spread_pct = ((ask - bid) / mid) * 100.0

    if spread_pct < 3.0:
        spread_score = 0.0  # Normal — no signal
    elif spread_pct < 6.0:
        spread_score = 1.0  # Elevated — some uncertainty
    elif spread_pct < 15.0:
        spread_score = 2.0  # Wide — market makers pulling back
    else:
        spread_score = 0.0  # Too wide — untradeable, not useful

    return round(min(5.0, imbalance_score + spread_score), 2)


def compute_price_trajectory(
    current_price: float,
    price_history: List[dict]
) -> float:
    """
    Price Trajectory (0-7 points).
    How far current price is from historical mean, measured in
    standard deviations. Large deviations suggest mean-reversion
    opportunity or genuine repricing event.

    Scoring:
        < 1.0σ  → 0 pts  (within normal range)
        1.0-1.5σ → 2 pts  (moderately displaced)
        1.5-2.0σ → 3-4 pts (significantly displaced)
        2.0-3.0σ → 5-6 pts (heavily displaced)
        3.0+σ    → 7 pts  (extreme displacement)
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

    if deviation < 1.0:
        score = 0.0
    elif deviation < 1.5:
        score = 2.0 + (deviation - 1.0) * 2.0  # 2-3
    elif deviation < 2.0:
        score = 3.0 + (deviation - 1.5) * 2.0  # 3-4
    elif deviation < 3.0:
        score = 4.0 + (deviation - 2.0) * 2.0  # 4-6
    else:
        score = 6.0 + min(1.0, (deviation - 3.0))  # 6-7

    return round(min(7.0, max(0.0, score)), 2)


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

    Sub-components measure ACTUAL price dislocation evidence:
      - Price velocity (0-10): abnormal recent price movement
      - Volume anomaly (0-8): unusual volume spike vs history
      - Order book (0-5): spread/depth imbalances
      - Price trajectory (0-7): deviation from historical mean

    time_decay removed — calendar proximity is not dislocation.
    """
    velocity = compute_price_velocity(current_price, price_history)
    volume = compute_volume_anomaly(volume_24h, volume_total)
    order_book = compute_order_book_score(bid, ask, bid_depth, ask_depth)
    trajectory = compute_price_trajectory(current_price, price_history)

    total = velocity + volume + order_book + trajectory

    return {
        'dislocation_total': round(min(30.0, total), 2),
        'price_velocity': velocity,
        'volume_anomaly': volume,
        'order_book': order_book,
        'price_trajectory': trajectory,
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


# ═══════════════════════════════════════════════════════════════════════════
# MARKET EFFICIENCY DISCOUNT — Scale edge score by price discovery maturity
# ═══════════════════════════════════════════════════════════════════════════

def compute_market_efficiency_multiplier(
    created_at: Optional[str],
    volume_total: float,
) -> float:
    """
    Markets with more volume and longer trading history have more efficient
    prices — apparent mispricings are less likely to be real. This returns
    a multiplier (0.50–1.0) applied to the raw edge score.

    Two factors combine:
      - Age factor (40% weight): how long the market has been open
      - Volume factor (60% weight): how much money has priced it

    Age factor (0–1, higher = more discovered):
        < 1 day    → 0.00   (brand new, price is very soft)
        3 days     → 0.15
        7 days     → 0.30
        14 days    → 0.50
        30 days    → 0.75
        60+ days   → 1.00   (well established)

    Volume factor (0–1, higher = more discovered):
        < $5k      → 0.00   (negligible activity)
        $25k       → 0.20
        $100k      → 0.45
        $250k      → 0.65
        $500k      → 0.80
        $1M+       → 1.00   (heavily traded)

    Combined efficiency → multiplier:
        efficiency 0.0  → multiplier 1.00  (full confidence in edge)
        efficiency 0.5  → multiplier 0.75
        efficiency 1.0  → multiplier 0.50  (halved — strong evidence required)

    Args:
        created_at: ISO date string of when the market was created
        volume_total: total lifetime volume in USD

    Returns: multiplier between 0.50 and 1.00
    """
    # ── Age factor ──
    age_factor = 0.5  # default: assume moderately established if unknown
    if created_at:
        try:
            created = datetime.fromisoformat(str(created_at))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400.0
            # Smooth curve: 0 at day 0, ~1.0 at 60 days
            age_factor = min(1.0, age_days / 60.0)
        except (ValueError, TypeError):
            age_factor = 0.5

    # ── Volume factor ──
    # Log-scale mapping: $0→0, $5k→~0.08, $25k→~0.25, $100k→~0.50,
    #                     $250k→~0.65, $500k→~0.77, $1M→~0.88, $5M→1.0
    import math
    if volume_total <= 0:
        vol_factor = 0.0
    else:
        # log10(5000)≈3.7, log10(5_000_000)≈6.7 → normalise to 0–1
        log_vol = math.log10(max(1.0, volume_total))
        vol_factor = min(1.0, max(0.0, (log_vol - 3.7) / 3.0))

    # ── Combine: volume is the stronger signal ──
    efficiency = 0.4 * age_factor + 0.6 * vol_factor

    # ── Map to multiplier: 1.0 (no discount) to 0.50 (max discount) ──
    max_discount = 0.50
    multiplier = 1.0 - (efficiency * max_discount)

    return round(max(0.50, min(1.0, multiplier)), 3)
