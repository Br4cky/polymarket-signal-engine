"""
Signal Manager — Replaces paper trading with signal emission + retroactive verification.

Instead of pretending to trade (and getting inaccurate P&L from 15-min scan gaps),
we emit signals with clear parameters (entry, TP, SL, max entry) and verify them
retroactively by checking actual price history between scans.

Signal lifecycle:
  ACTIVE          → Signal issued, awaiting resolution
  TP_HIT          → Price reached take-profit target (win)
  SL_HIT          → Price hit stop-loss level (loss)
  MARKET_RESOLVED → Contract settled at $1 or $0 (win or loss based on P&L)
  INVALIDATED     → Entry price moved past max_entry before anyone could act
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Callable

from src.utils import safe_float

logger = logging.getLogger(__name__)


# ─── Signal Creation ────────────────────────────────────────────────────────

def compute_tp_sl(
    entry_price: float,
    edge_tier: dict,
    edge_tier_name: str,
    opportunity: dict = None,
) -> dict:
    """
    Compute take-profit, stop-loss, and entry range for a binary contract.

    Key principle: every Polymarket contract resolves at $1 or $0.
    A token at 30¢ has 70¢ of upside and 30¢ of downside. TP/SL must
    respect this structure rather than using arbitrary fixed percentages.

    TP logic — fraction of available upside:
      available_upside = 1.0 - entry_price
      tp_fraction = base_fraction (from edge tier) scaled by:
        - Edge score strength (higher edge → larger fraction)
        - Number of confirming layers (multi-layer → more confident)
        - Price level (cheap tokens have more room, so fraction is conservative)
      TP = entry + (tp_fraction × available_upside)

    SL logic — fraction of downside, informed by order book:
      available_downside = entry_price (goes to $0 at worst)
      sl_fraction = base_fraction scaled by:
        - Order book depth on bid side (thin support → tighter SL)
        - Edge score (higher confidence → slightly more room)
      SL = entry - (sl_fraction × available_downside)

    Entry range — based on order book spread and liquidity:
      min_entry = bid price (best realistic fill)
      max_entry = entry + buffer derived from spread width and depth
      If spread is very tight, buffer is small. If wide, buffer is larger.

    Returns dict with tp_price, sl_price, min_entry_price, max_entry_price.
    """
    import math

    opp = opportunity or {}
    edge_score = opp.get('edge_score', 0)
    layer_scores = opp.get('layer_scores', {})
    bid_raw = opp.get('bid', 0)
    ask_raw = opp.get('ask', 0)
    bid_depth = opp.get('bid_depth', 0)
    ask_depth = opp.get('ask_depth', 0)
    spread_pct = opp.get('spread_pct', 0)

    # ── Sanity-check bid/ask data ──
    # Order book data can be garbage (bid=0.01, ask=0.99) for thin markets.
    # Only trust bid/ask if they're within 15% of entry price.
    bid = bid_raw if (bid_raw > 0 and abs(bid_raw - entry_price) / entry_price < 0.15) else 0
    ask = ask_raw if (ask_raw > 0 and abs(ask_raw - entry_price) / entry_price < 0.15) else 0

    # ── How many scoring layers actually contributed? ──
    active_layers = sum(1 for v in layer_scores.values() if v and v > 0)

    # ══════════════════════════════════════════════════════════════════
    # TAKE-PROFIT: fraction of available upside
    # ══════════════════════════════════════════════════════════════════
    available_upside = max(0.01, 1.0 - entry_price)

    # Base TP fraction by tier (conservative starting points)
    # These represent how much of the available upside we target
    base_tp = {'high': 0.20, 'medium': 0.15, 'low': 0.10}
    tp_fraction = base_tp.get(edge_tier_name, 0.12)

    # Scale by edge strength: edge 15-70 maps to 0.8x-1.4x multiplier
    # Stronger edge = we think mispricing is larger = target more upside
    edge_mult = 0.8 + (min(edge_score, 70) - 15) * (0.6 / 55)
    tp_fraction *= edge_mult

    # Multi-layer confirmation bonus: each additional confirming layer
    # adds confidence, so we can target slightly more
    if active_layers >= 3:
        tp_fraction *= 1.20   # 3+ layers: 20% more aggressive
    elif active_layers >= 2:
        tp_fraction *= 1.10   # 2 layers: 10% more aggressive

    # Cheap token dampener: tokens under 15¢ have huge theoretical upside
    # but price moves are noisy, so be conservative with TP target
    if entry_price < 0.08:
        tp_fraction *= 0.70
    elif entry_price < 0.15:
        tp_fraction *= 0.85

    # Clamp: never target less than 5% or more than 50% of available upside
    tp_fraction = max(0.05, min(0.50, tp_fraction))

    tp_price = entry_price + (tp_fraction * available_upside)

    # ══════════════════════════════════════════════════════════════════
    # STOP-LOSS: fraction of downside, informed by order book
    # ══════════════════════════════════════════════════════════════════
    available_downside = max(0.01, entry_price)  # price can go to $0

    # Base SL fraction by tier
    base_sl = {'high': 0.30, 'medium': 0.25, 'low': 0.20}
    sl_fraction = base_sl.get(edge_tier_name, 0.22)

    # Order book depth adjustment:
    # If there's substantial bid depth, price has support → can use wider SL
    # If bid side is thin, price could gap → tighten SL
    total_depth = bid_depth + ask_depth
    if total_depth > 0:
        bid_ratio = bid_depth / total_depth  # 0-1, how much depth on our side
        if bid_ratio > 0.6:
            sl_fraction *= 1.10  # Good support, slightly wider OK
        elif bid_ratio < 0.3:
            sl_fraction *= 0.80  # Thin support, tighten up
    else:
        # No depth data — be conservative
        sl_fraction *= 0.85

    # High edge → slightly more room (more conviction, give it time to work)
    if edge_score >= 50 and active_layers >= 2:
        sl_fraction *= 1.10

    # Cheap token tightening: under 10¢, a 25% drop is only 2.5¢
    # which could happen from one order in a thin book. Tighten to limit damage.
    if entry_price < 0.05:
        sl_fraction = min(sl_fraction, 0.25)
    elif entry_price < 0.10:
        sl_fraction = min(sl_fraction, 0.30)

    # Clamp: never less than 10% or more than 40% of downside
    sl_fraction = max(0.10, min(0.40, sl_fraction))

    sl_price = entry_price - (sl_fraction * available_downside)

    # ══════════════════════════════════════════════════════════════════
    # RISK/REWARD ENFORCEMENT: ensure SL never risks more than TP gains
    # ══════════════════════════════════════════════════════════════════
    # Minimum R:R of 1.0 — if the current SL risks more than TP rewards,
    # tighten the SL until R:R >= 1.0. This is critical for expensive
    # tokens (60¢+) where available upside is small but downside is large.
    reward = tp_price - entry_price
    risk = entry_price - sl_price
    min_rr = 1.0  # at minimum, reward must equal risk

    if risk > 0 and reward > 0 and (reward / risk) < min_rr:
        # Tighten SL so risk = reward / min_rr
        max_risk = reward / min_rr
        sl_price = entry_price - max_risk

    # ══════════════════════════════════════════════════════════════════
    # ENTRY RANGE: grounded in order book reality
    # ══════════════════════════════════════════════════════════════════
    # Min entry: the bid price if trusted, otherwise entry - small buffer
    if bid > 0:
        min_entry = bid
    else:
        # No trusted bid — use price-level heuristic
        if entry_price < 0.10:
            min_entry = entry_price - 0.005
        elif entry_price < 0.50:
            min_entry = entry_price - 0.01
        else:
            min_entry = entry_price - 0.015

    # Max entry buffer: based on trusted spread or price-level heuristic
    if bid > 0 and ask > 0 and ask > bid:
        # Trusted spread data — buffer is ~60% of the spread
        spread_abs = ask - bid
        max_entry_buffer = max(0.005, spread_abs * 0.6)
    else:
        # No trusted spread — use price-level heuristic
        if entry_price < 0.05:
            max_entry_buffer = 0.008
        elif entry_price < 0.15:
            max_entry_buffer = 0.012
        elif entry_price < 0.50:
            max_entry_buffer = 0.02
        else:
            max_entry_buffer = 0.03

    # Liquidity adjustment: very liquid markets have tighter fills
    liquidity = opp.get('liquidity_usd', 0)
    if liquidity > 100000:
        max_entry_buffer *= 0.7  # Deep liquidity, tighter fill
    elif liquidity < 10000:
        max_entry_buffer *= 1.3  # Thin, may need wider entry

    max_entry_price = entry_price + max_entry_buffer

    # ── Final clamps ──
    tp_price = min(tp_price, 0.99)
    sl_price = max(sl_price, 0.001)
    min_entry = max(min_entry, 0.001)
    max_entry_price = min(max_entry_price, 0.99)

    # Ensure entry range makes sense
    if min_entry >= max_entry_price:
        min_entry = entry_price - 0.005
        max_entry_price = entry_price + 0.01

    # TP/SL percentages for display
    tp_pct = ((tp_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
    sl_pct = ((sl_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

    return {
        'tp_price': round(tp_price, 4),
        'sl_price': round(sl_price, 4),
        'min_entry_price': round(min_entry, 4),
        'max_entry_price': round(max_entry_price, 4),
        'tp_pct': round(tp_pct, 1),
        'sl_pct': round(sl_pct, 1),
        'tp_fraction': round(tp_fraction, 3),   # what % of upside we're targeting
        'sl_fraction': round(sl_fraction, 3),   # what % of downside we're risking
    }


def generate_rationale(opportunity: dict) -> str:
    """
    Generate a human-readable rationale for why this signal was issued.
    Highlights which layers contributed and their key sub-signals.
    """
    layers = opportunity.get('layer_scores', {})
    parts = []

    structural = layers.get('structural', 0)
    if structural > 0:
        detail = opportunity.get('structural_detail', {})
        sub = []
        kalshi = detail.get('cross_platform', {})
        if kalshi.get('score', 0) > 0:
            sub.append(f"Kalshi div {kalshi.get('divergence_pct', 0):.1f}%")
        manifold = detail.get('manifold', {})
        if manifold.get('score', 0) > 0:
            sub.append(f"Manifold div {manifold.get('divergence_pct', 0):.1f}%")
        comb = detail.get('combinatorial', {})
        if comb.get('score', 0) > 0:
            sub.append(f"combo {comb.get('mispricing_type', '')}")
        detail_str = f" ({', '.join(sub)})" if sub else ""
        parts.append(f"Structural {structural:.0f}/30{detail_str}")

    smart_money = layers.get('smart_money', 0)
    if smart_money > 0:
        sm_detail = opportunity.get('smart_money_detail', {})
        whale_count = sm_detail.get('whale_count', 0)
        parts.append(f"Smart money {smart_money:.0f}/25 ({whale_count} whales)")

    dislocation = layers.get('dislocation', 0)
    if dislocation > 0:
        d_detail = opportunity.get('dislocation_detail', {})
        sub = []
        if d_detail.get('price_velocity', 0) > 2:
            sub.append("velocity")
        if d_detail.get('volume_anomaly', 0) > 2:
            sub.append("vol spike")
        if d_detail.get('order_book', 0) > 2:
            sub.append("book imbalance")
        if d_detail.get('price_trajectory', 0) > 2:
            sub.append("trajectory")
        detail_str = f" ({', '.join(sub)})" if sub else ""
        parts.append(f"Dislocation {dislocation:.0f}/30{detail_str}")

    external = layers.get('external', 0)
    if external > 0:
        parts.append(f"External {external:.1f}/15")

    if not parts:
        return "Edge detected above threshold"

    return " | ".join(parts)


def emit_signal(opportunity: dict, config: dict) -> Optional[dict]:
    """
    Create a signal from a scored opportunity.
    Returns a fully-formed signal dict ready for persistence and messaging,
    or None if the signal fails quality gates (e.g. TP too small to be
    distinguishable from noise).
    """
    from src.portfolio import get_edge_tier, get_edge_tier_name

    edge_score = opportunity['edge_score']
    entry_price = opportunity['current_price']
    tier = get_edge_tier(edge_score, config)
    tier_name = get_edge_tier_name(edge_score, config)

    # Compute TP/SL/entry range
    targets = compute_tp_sl(entry_price, tier, tier_name, opportunity=opportunity)

    # ══════════════════════════════════════════════════════════════════
    # QUALITY GATES — data-backed filters from signal performance analysis
    # ══════════════════════════════════════════════════════════════════

    layer_scores = opportunity.get('layer_scores', {})
    active_layers = sum(1 for v in layer_scores.values() if v and v > 0)
    has_smart_money = layer_scores.get('smart_money', 0) > 0
    has_dislocation = layer_scores.get('dislocation', 0) > 0
    has_external = layer_scores.get('external', 0) > 0
    has_structural = layer_scores.get('structural', 0) > 0

    # ── Gate 1: Layer quality requirement ──
    # Data: 2-layer signals without smart money = -361% total P&L
    # Data: Only disl+exte+smar (3-layer) combo is profitable (+335%, 46% WR)
    # Rule: Require either 3+ active layers, OR 2 layers with smart money confirmed
    if active_layers < 2:
        logger.info(
            f"SIGNAL REJECTED (insufficient layers): {opportunity['outcome']} @ ${entry_price:.4f} "
            f"layers={active_layers} ({opportunity['question'][:50]}...)"
        )
        return None

    if active_layers == 2 and not has_smart_money:
        logger.info(
            f"SIGNAL REJECTED (2 layers without smart money): {opportunity['outcome']} @ ${entry_price:.4f} "
            f"layers={active_layers}, smart_money={layer_scores.get('smart_money', 0)} "
            f"({opportunity['question'][:50]}...)"
        )
        return None

    # ── Gate 2: Minimum edge score ──
    # Data: Edge 15-25 = 35% WR, -17% avg P&L; Edge 25-35 = 27% WR, -19% avg
    # Data: Edge 35-50 = 42% WR, -3.5% avg; Edge 50+ = 50% WR, +28% avg
    # Rule: Raise effective minimum to 35 (config threshold still filters at 15
    # in rank_opportunities for dashboard display, but signals need 35+)
    MIN_SIGNAL_EDGE = 35.0
    if edge_score < MIN_SIGNAL_EDGE:
        logger.info(
            f"SIGNAL REJECTED (edge too low): {opportunity['outcome']} @ ${entry_price:.4f} "
            f"edge={edge_score:.1f} (need >={MIN_SIGNAL_EDGE}) "
            f"({opportunity['question'][:50]}...)"
        )
        return None

    # ── Gate 3: Cheap token filter ──
    # Data: Under 15¢ tokens = -789% total P&L (wipeouts from $0 resolution dominate)
    # Rule: Tokens under 10¢ need edge 50+ AND smart money. 10-20¢ need edge 40+.
    if entry_price < 0.10:
        if edge_score < 50 or not has_smart_money:
            logger.info(
                f"SIGNAL REJECTED (cheap token <10¢ needs edge>=50 + smart money): "
                f"{opportunity['outcome']} @ ${entry_price:.4f} "
                f"edge={edge_score:.1f}, smart_money={has_smart_money} "
                f"({opportunity['question'][:50]}...)"
            )
            return None
    elif entry_price < 0.20:
        if edge_score < 40:
            logger.info(
                f"SIGNAL REJECTED (token <20¢ needs edge>=40): "
                f"{opportunity['outcome']} @ ${entry_price:.4f} "
                f"edge={edge_score:.1f} ({opportunity['question'][:50]}...)"
            )
            return None

    # ── Gate 4: Category filter for known-losing patterns ──
    # Data: Crypto price targets = 0% WR; Musk tweet markets = -250%
    question_lower = opportunity.get('question', '').lower()
    slug_lower = opportunity.get('slug', '').lower()

    TOXIC_PATTERNS = [
        # Crypto price targets: "Will BTC hit $X" — 0% win rate
        ('crypto_price_target', lambda q, s: any(
            tok in q for tok in ['hit $', 'reach $', 'above $', 'below $']
        ) and any(
            coin in q for coin in ['bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'xrp', 'doge']
        )),
        # Musk tweet markets — extremely noisy, -250% P&L
        ('musk_tweets', lambda q, s: 'musk' in q and ('tweet' in q or 'post' in q or 'x.com' in q)),
    ]

    for pattern_name, pattern_fn in TOXIC_PATTERNS:
        if pattern_fn(question_lower, slug_lower):
            logger.info(
                f"SIGNAL REJECTED (toxic category: {pattern_name}): "
                f"{opportunity['outcome']} @ ${entry_price:.4f} "
                f"({opportunity['question'][:50]}...)"
            )
            return None

    # ── Gate 5: TP quality — reject noise signals ──
    # A 2-3% TP on a high-priced token isn't a mispricing signal, it's
    # random fluctuation. Minimum TP must be meaningful.
    MIN_TP_PCT = 8.0   # at least 8% TP to be worth signalling
    MIN_TP_ABS = 0.03  # at least 3¢ absolute move to TP

    tp_pct_abs = abs(targets['tp_pct'])
    tp_abs = abs(targets['tp_price'] - entry_price)

    if tp_pct_abs < MIN_TP_PCT and tp_abs < MIN_TP_ABS:
        logger.info(
            f"SIGNAL REJECTED (TP too small): {opportunity['outcome']} @ ${entry_price:.4f} "
            f"TP={targets['tp_price']:.4f} ({targets['tp_pct']:+.1f}%) "
            f"— need >={MIN_TP_PCT}% or >=${MIN_TP_ABS:.2f} abs "
            f"({opportunity['question'][:50]}...)"
        )
        return None

    # Expiry = contract resolution date (every Polymarket contract resolves)
    # No artificial max_hold_days — signals run until TP, SL, or market resolution.
    now = datetime.now(timezone.utc)
    resolution_date = opportunity.get('resolution_date', '')
    expiry = None
    if resolution_date:
        try:
            res = datetime.fromisoformat(str(resolution_date))
            if res.tzinfo is None:
                res = res.replace(tzinfo=timezone.utc)
            if res > now:
                expiry = res
        except (ValueError, TypeError):
            pass

    signal = {
        'signal_id': str(uuid.uuid4())[:8],
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'status': 'active',

        # Market info
        'market_id': opportunity['market_id'],
        'condition_id': opportunity.get('condition_id', ''),
        'token_id': opportunity['token_id'],
        'question': opportunity['question'],
        'outcome': opportunity['outcome'],
        'slug': opportunity.get('slug', ''),
        'category': opportunity.get('category', ''),
        'resolution_date': resolution_date,

        # Call parameters — this is what we tell users
        'entry_price': round(entry_price, 4),
        'min_entry_price': targets['min_entry_price'],
        'max_entry_price': targets['max_entry_price'],
        'tp_price': targets['tp_price'],
        'sl_price': targets['sl_price'],
        'tp_pct': targets['tp_pct'],
        'sl_pct': targets['sl_pct'],
        'expiry': expiry.isoformat() if expiry else None,

        # Scoring context
        'edge_score': round(edge_score, 1),
        'edge_tier': tier_name,
        'layer_scores': opportunity.get('layer_scores', {}),
        'convexity_band': opportunity.get('convexity_band', ''),
        'potential_multiple': opportunity.get('potential_multiple', 0),
        'days_to_close': opportunity.get('days_to_close', 0),
        'liquidity_usd': opportunity.get('liquidity_usd', 0),
        'volume_24h': opportunity.get('volume_24h', 0),
        'rationale': generate_rationale(opportunity),
        'tp_fraction': targets.get('tp_fraction', 0),
        'sl_fraction': targets.get('sl_fraction', 0),

        # Resolution (filled when signal resolves)
        'resolved_at': None,
        'resolution_type': None,
        'peak_price': None,
        'trough_price': None,
        'final_price': None,
        'hypothetical_pnl_pct': None,
        'current_price': entry_price,
        'live_pnl_pct': 0.0,
    }

    logger.info(
        f"NEW SIGNAL: {opportunity['outcome']} @ ${entry_price:.4f} "
        f"Enter ${targets['min_entry_price']:.4f}-${targets['max_entry_price']:.4f} "
        f"TP ${targets['tp_price']:.4f} ({targets['tp_pct']:+.0f}%) "
        f"SL ${targets['sl_price']:.4f} ({targets['sl_pct']:.0f}%) "
        f"[edge={edge_score:.0f}, {tier_name}] "
        f"({opportunity['question'][:50]}...)"
    )

    return signal


# ─── Signal Verification ────────────────────────────────────────────────────

def _parse_ts(ts) -> datetime:
    """Parse a timestamp (int/float epoch, string epoch, or ISO string) to datetime."""
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    ts_str = str(ts).strip()
    # Handle string-encoded Unix timestamps (e.g. "1709683200" from CLOB API)
    try:
        epoch = float(ts_str)
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (ValueError, OverflowError):
        pass
    # Python 3.10 doesn't support 'Z' suffix in fromisoformat
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _check_price_history_for_resolution(
    signal: dict,
    price_history: List[dict]
) -> dict:
    """
    Check if a signal's TP or SL was hit in the price history.

    Scans price history from signal timestamp onwards.
    If both TP and SL are hit, the one that happened FIRST wins.

    Also tracks the actual price at which TP/SL was first crossed — this
    distinguishes genuine TP/SL hits (price near TP/SL level) from market
    resolution jumps (price leaps from below TP straight to ~$1 or ~$0
    when the contract settles).

    Returns dict: either {resolved: True, ...} or {resolved: False, peak_price, trough_price}
    """
    tp_price = signal['tp_price']
    sl_price = signal['sl_price']
    entry_price = signal['entry_price']
    signal_ts = signal['timestamp']

    try:
        signal_time = _parse_ts(signal_ts)
    except (ValueError, TypeError):
        signal_time = None

    tp_hit_time = None
    tp_hit_price = None  # actual price when TP was first crossed
    sl_hit_time = None
    sl_hit_price = None  # actual price when SL was first crossed
    peak_price = entry_price
    trough_price = entry_price

    for point in price_history:
        ts = point.get('timestamp') or point.get('time') or point.get('t')
        price = safe_float(point.get('price', 0))

        if not price or price <= 0:
            continue

        # Skip data points before signal was issued
        if signal_time and ts:
            try:
                point_time = _parse_ts(ts)
                if point_time < signal_time:
                    continue
            except (ValueError, TypeError, OSError):
                # Can't parse timestamp — still use the data point rather than
                # silently skipping it (which previously caused 0 resolutions)
                pass

        # Track extremes
        peak_price = max(peak_price, price)
        trough_price = min(trough_price, price)

        # Check TP hit (price >= tp_price) — record the actual price too
        if tp_hit_time is None and price >= tp_price:
            tp_hit_time = ts
            tp_hit_price = price

        # Check SL hit (price <= sl_price)
        if sl_hit_time is None and price <= sl_price:
            sl_hit_time = ts
            sl_hit_price = price

    # ── Determine resolution type ──
    # Key distinction: if the crossing price is way beyond TP/SL (near $1 or $0),
    # it's likely a market resolution jump, not a gradual TP/SL hit.
    # A genuine TP hit looks like price moving from 0.30 → 0.35 → 0.41 (near TP).
    # A resolution jump looks like price going from 0.30 → 0.99 in one candle.

    def _is_resolution_jump_tp(hit_price, tp):
        """Was this TP crossing a resolution jump rather than gradual?"""
        if hit_price is None:
            return False
        # If price jumped past 90¢ AND is more than 30% beyond TP, it's a jump
        return hit_price >= 0.90 and hit_price > tp * 1.30

    def _is_resolution_jump_sl(hit_price, sl):
        """Was this SL crossing a resolution crash rather than gradual?"""
        if hit_price is None:
            return False
        # If price dropped below 10¢ AND is more than 50% below SL, it's a crash
        return hit_price <= 0.10 and hit_price < sl * 0.50

    # Determine which event happened first
    first_event = None
    first_time = None
    first_price = None

    if tp_hit_time is not None and sl_hit_time is not None:
        try:
            tp_t = _parse_ts(tp_hit_time)
            sl_t = _parse_ts(sl_hit_time)
            if tp_t <= sl_t:
                first_event, first_time, first_price = 'tp', tp_hit_time, tp_hit_price
            else:
                first_event, first_time, first_price = 'sl', sl_hit_time, sl_hit_price
        except (ValueError, TypeError):
            first_event, first_time, first_price = 'tp', tp_hit_time, tp_hit_price
    elif tp_hit_time is not None:
        first_event, first_time, first_price = 'tp', tp_hit_time, tp_hit_price
    elif sl_hit_time is not None:
        first_event, first_time, first_price = 'sl', sl_hit_time, sl_hit_price

    if first_event == 'tp':
        if _is_resolution_jump_tp(first_price, tp_price):
            # Price jumped straight to ~$1 — market resolved, not a genuine TP hit
            # Trader gets $1.00 payout
            return _make_resolution(
                'market_resolved', 1.0, peak_price, trough_price,
                entry_price, first_time
            )
        else:
            return _make_resolution(
                'tp_hit', tp_price, peak_price, trough_price,
                entry_price, first_time
            )

    if first_event == 'sl':
        if _is_resolution_jump_sl(first_price, sl_price):
            # Price crashed to ~$0 — market resolved against us
            # Trader gets $0.00 payout
            return _make_resolution(
                'market_resolved', 0.0, peak_price, trough_price,
                entry_price, first_time
            )
        else:
            return _make_resolution(
                'sl_hit', sl_price, peak_price, trough_price,
                entry_price, first_time
            )

    # Neither hit yet
    return {
        'resolved': False,
        'peak_price': round(peak_price, 4),
        'trough_price': round(trough_price, 4),
    }


def _make_resolution(
    resolution_type: str,
    final_price: float,
    peak_price: float,
    trough_price: float,
    entry_price: float,
    resolved_at
) -> dict:
    """Build a resolution result dict."""
    pnl_pct = ((final_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
    return {
        'resolved': True,
        'resolution_type': resolution_type,
        'final_price': round(final_price, 4),
        'peak_price': round(peak_price, 4),
        'trough_price': round(trough_price, 4),
        'hypothetical_pnl_pct': round(pnl_pct, 2),
        'resolved_at': resolved_at,
    }


def verify_signals(
    signals_state: dict,
    fetch_price_history: Callable[[str], List[dict]],
    current_prices: dict = None,
    active_market_ids: set = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Verify all active signals against price history.

    Resolution priority (in order):
      1. Price history TP/SL: did price cross TP or SL?
         → _check_price_history_for_resolution handles this, INCLUDING
           detecting resolution jumps (price leaping to ~$1 or ~$0 in one
           candle, which is a market settlement, not a genuine TP/SL hit).
      2. Market resolution fallback: if TP/SL wasn't hit but market has
         disappeared from active list with extreme price → settled.
      No artificial expiry — every Polymarket contract resolves at $1 or $0.

    Args:
        signals_state: Full signals state dict with 'active' list
        fetch_price_history: Callable(token_id) → List[{timestamp, price}]
        current_prices: Optional {token_id: price} fallback when history is empty
        active_market_ids: Optional set of currently active/trading market IDs.
            If a signal's market is not in this set, we check if it resolved.

    Returns:
        (newly_resolved, still_active) — lists of signal dicts
    """
    current_prices = current_prices or {}
    active_market_ids = active_market_ids or set()
    active = signals_state.get('active', [])
    newly_resolved = []
    still_active = []
    now = datetime.now(timezone.utc)

    for signal in active:
        token_id = signal['token_id']
        entry = signal['entry_price']
        market_gone = active_market_ids and signal['market_id'] not in active_market_ids

        # 1. Fetch price history and check TP/SL
        #    Every Polymarket contract resolves — no artificial expiry needed.
        try:
            history = fetch_price_history(token_id)
        except Exception as e:
            logger.warning(f"Price history failed for signal {signal['signal_id']}: {e}")
            history = []

        if history:
            logger.info(
                f"Signal {signal['signal_id']}: got {len(history)} price points, "
                f"checking TP={signal['tp_price']:.4f} SL={signal['sl_price']:.4f}"
            )

            result = _check_price_history_for_resolution(signal, history)

            if result.get('resolved'):
                signal['status'] = result['resolution_type']
                try:
                    signal['resolved_at'] = _parse_ts(result['resolved_at']).isoformat()
                except (ValueError, TypeError):
                    signal['resolved_at'] = now.isoformat()
                signal['resolution_type'] = result['resolution_type']
                signal['final_price'] = result['final_price']
                signal['peak_price'] = result['peak_price']
                signal['trough_price'] = result['trough_price']
                signal['hypothetical_pnl_pct'] = result['hypothetical_pnl_pct']
                newly_resolved.append(signal)

                rt = result['resolution_type']
                if rt == 'market_resolved':
                    won = result['hypothetical_pnl_pct'] > 0
                    label = f"MARKET SETTLED ({'WIN' if won else 'LOSS'})"
                else:
                    label = 'WIN' if rt == 'tp_hit' else 'LOSS'
                logger.info(
                    f"SIGNAL {label}: {signal['signal_id']} "
                    f"{rt} {result['hypothetical_pnl_pct']:+.1f}% "
                    f"(peak={result['peak_price']:.4f} trough={result['trough_price']:.4f}) "
                    f"({signal['question'][:40]}...)"
                )
                continue

            # Still active — update tracking from price history
            signal['peak_price'] = result.get('peak_price', signal.get('peak_price'))
            signal['trough_price'] = result.get('trough_price', signal.get('trough_price'))

            lp = safe_float(history[-1].get('price', 0))
            if lp > 0:
                signal['current_price'] = round(lp, 4)
                signal['live_pnl_pct'] = round(
                    ((lp - entry) / entry * 100) if entry > 0 else 0, 2
                )

            # 3. If TP/SL wasn't hit but market is gone → market resolved
            #    (e.g., no price history near extremes, but market disappeared)
            if market_gone:
                cp = current_prices.get(token_id, 0) or safe_float(signal.get('current_price', 0))
                if cp >= 0.95 or cp <= 0.05:
                    payout = 1.0 if cp >= 0.50 else 0.0
                    pnl_pct = ((payout - entry) / entry * 100) if entry > 0 else 0
                    signal['status'] = 'market_resolved'
                    signal['resolved_at'] = now.isoformat()
                    signal['resolution_type'] = 'market_resolved'
                    signal['final_price'] = payout
                    signal['hypothetical_pnl_pct'] = round(pnl_pct, 2)
                    newly_resolved.append(signal)
                    won = payout > entry
                    logger.info(
                        f"MARKET RESOLVED ({'WIN' if won else 'LOSS'}): {signal['signal_id']} "
                        f"entry=${entry:.4f} → payout=${payout:.2f} "
                        f"pnl={pnl_pct:+.1f}% "
                        f"({signal['question'][:40]}...)"
                    )
                    continue

            still_active.append(signal)
            continue

        # No price history available
        logger.info(
            f"No price history for signal {signal['signal_id']} "
            f"token={token_id[:20]}..."
        )

        # 5. Fallback: no history, but market might have resolved
        if market_gone:
            cp = current_prices.get(token_id, 0) or safe_float(signal.get('current_price', 0))
            if cp >= 0.95 or cp <= 0.05:
                payout = 1.0 if cp >= 0.50 else 0.0
                pnl_pct = ((payout - entry) / entry * 100) if entry > 0 else 0
                signal['status'] = 'market_resolved'
                signal['resolved_at'] = now.isoformat()
                signal['resolution_type'] = 'market_resolved'
                signal['final_price'] = payout
                signal['peak_price'] = signal.get('peak_price') or max(entry, cp)
                signal['trough_price'] = signal.get('trough_price') or min(entry, cp)
                signal['hypothetical_pnl_pct'] = round(pnl_pct, 2)
                newly_resolved.append(signal)
                won = payout > entry
                logger.info(
                    f"MARKET RESOLVED ({'WIN' if won else 'LOSS'}): {signal['signal_id']} "
                    f"entry=${entry:.4f} → payout=${payout:.2f} (no price history) "
                    f"({signal['question'][:40]}...)"
                )
                continue

        # Update from current_prices map if available
        cp = current_prices.get(token_id, 0)
        if cp > 0 and cp != signal.get('current_price'):
            signal['current_price'] = round(cp, 4)
            signal['live_pnl_pct'] = round(
                ((cp - entry) / entry * 100) if entry > 0 else 0, 2
            )
        still_active.append(signal)

    return newly_resolved, still_active


# ─── Signal State Persistence ───────────────────────────────────────────────

def load_signals(filepath: str) -> dict:
    """Load signals state from file or create new."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            if 'active' in data and 'resolved' in data:
                return data
            logger.warning("Invalid signals file structure, creating new")
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to load signals, creating new")

    return _create_empty_state()


def _create_empty_state() -> dict:
    return {
        'active': [],
        'resolved': [],
        'cooldowns': {},   # market_id → timestamp of last SL_HIT
        'stats': {
            'total_signals': 0,
            'total_wins': 0,
            'total_losses': 0,
        },
        'created_at': datetime.now(timezone.utc).isoformat(),
        'last_updated': datetime.now(timezone.utc).isoformat(),
    }


def save_signals(signals_state: dict, filepath: str):
    """Save signals state to file."""
    signals_state['last_updated'] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(signals_state, f, indent=2, default=str)


def is_signal_on_cooldown(signals_state: dict, market_id: str, cooldown_hours: int = 24) -> bool:
    """Check if a market is on signal cooldown (recently had a SL_HIT)."""
    cooldowns = signals_state.get('cooldowns', {})
    ts_str = cooldowns.get(market_id)
    if not ts_str:
        return False
    try:
        ts = _parse_ts(ts_str)
        hours_since = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        return hours_since < cooldown_hours
    except (ValueError, TypeError):
        return False


def update_signal_stats(signals_state: dict, newly_resolved: List[dict]):
    """Update stats and cooldowns after verification resolves signals."""
    stats = signals_state.setdefault('stats', {
        'total_signals': 0, 'total_wins': 0, 'total_losses': 0,
    })
    cooldowns = signals_state.setdefault('cooldowns', {})

    for signal in newly_resolved:
        rt = signal.get('resolution_type', '')
        if rt == 'tp_hit':
            stats['total_wins'] = stats.get('total_wins', 0) + 1
        elif rt == 'sl_hit':
            stats['total_losses'] = stats.get('total_losses', 0) + 1
            # Set cooldown for this market
            cooldowns[signal['market_id']] = datetime.now(timezone.utc).isoformat()
        elif rt == 'market_resolved':
            # Market settled at $1 or $0 — classify as win or loss
            pnl = signal.get('hypothetical_pnl_pct', 0)
            if pnl > 0:
                stats['total_wins'] = stats.get('total_wins', 0) + 1
            else:
                stats['total_losses'] = stats.get('total_losses', 0) + 1
                cooldowns[signal['market_id']] = datetime.now(timezone.utc).isoformat()
            stats['total_market_resolved'] = stats.get('total_market_resolved', 0) + 1
        # No 'expired' type — every contract resolves at $1 or $0

    # Clean old cooldowns (>48h)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    signals_state['cooldowns'] = {
        mid: ts for mid, ts in cooldowns.items()
        if _is_recent(ts, cutoff)
    }


def _is_recent(ts_str: str, cutoff: datetime) -> bool:
    try:
        ts = _parse_ts(ts_str)
        return ts > cutoff
    except (ValueError, TypeError):
        return False


# ─── Signal Metrics for Dashboard ───────────────────────────────────────────

def signal_metrics(signals_state: dict) -> dict:
    """
    Compute signal performance metrics for the dashboard.
    Returns accuracy stats, average win/loss, and per-tier breakdown.
    """
    resolved = signals_state.get('resolved', [])
    active = signals_state.get('active', [])

    # Classify resolved signals: market_resolved counts as win or loss based on P&L
    tp_hits = [s for s in resolved if s.get('resolution_type') == 'tp_hit']
    sl_hits = [s for s in resolved if s.get('resolution_type') == 'sl_hit']
    market_resolved = [s for s in resolved if s.get('resolution_type') == 'market_resolved']
    # Market-resolved signals split into wins (pnl > 0) and losses
    mr_wins = [s for s in market_resolved if s.get('hypothetical_pnl_pct', 0) > 0]
    mr_losses = [s for s in market_resolved if s.get('hypothetical_pnl_pct', 0) <= 0]

    wins = tp_hits + mr_wins
    losses = sl_hits + mr_losses

    total_resolved = len(resolved)
    win_count = len(wins)
    loss_count = len(losses)

    decisive = win_count + loss_count
    win_rate = win_count / decisive if decisive > 0 else 0

    avg_win_pct = (
        sum(s.get('hypothetical_pnl_pct', 0) for s in wins) / win_count
        if win_count > 0 else 0
    )
    avg_loss_pct = (
        sum(s.get('hypothetical_pnl_pct', 0) for s in losses) / loss_count
        if loss_count > 0 else 0
    )

    # Expected value per signal
    ev_per_signal = (
        (win_rate * avg_win_pct) + ((1 - win_rate) * avg_loss_pct)
        if decisive > 0 else 0
    )

    # Per-tier breakdown
    tier_stats = {}
    for tier_name in ['high', 'medium', 'low']:
        tier_resolved = [s for s in resolved if s.get('edge_tier') == tier_name]
        tier_wins = [s for s in tier_resolved if s.get('resolution_type') == 'tp_hit'
                     or (s.get('resolution_type') == 'market_resolved' and s.get('hypothetical_pnl_pct', 0) > 0)]
        tier_losses = [s for s in tier_resolved if s.get('resolution_type') == 'sl_hit'
                       or (s.get('resolution_type') == 'market_resolved' and s.get('hypothetical_pnl_pct', 0) <= 0)]
        tier_decisive = len(tier_wins) + len(tier_losses)
        tier_stats[tier_name] = {
            'total': len(tier_resolved),
            'wins': len(tier_wins),
            'losses': len(tier_losses),
            'win_rate': round(len(tier_wins) / tier_decisive, 3) if tier_decisive > 0 else 0,
            'avg_pnl': round(
                sum(s.get('hypothetical_pnl_pct', 0) for s in tier_resolved) / len(tier_resolved), 2
            ) if tier_resolved else 0,
        }

    return {
        'total_signals_issued': signals_state.get('stats', {}).get('total_signals', 0),
        'active_signals': len(active),
        'total_resolved': total_resolved,
        'wins': win_count,
        'losses': loss_count,
        'market_resolved': len(market_resolved),
        'win_rate': round(win_rate, 3),
        'avg_win_pct': round(avg_win_pct, 2),
        'avg_loss_pct': round(avg_loss_pct, 2),
        'ev_per_signal': round(ev_per_signal, 2),
        'by_tier': tier_stats,
    }
