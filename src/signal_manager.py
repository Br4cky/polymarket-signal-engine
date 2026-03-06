"""
Signal Manager — Replaces paper trading with signal emission + retroactive verification.

Instead of pretending to trade (and getting inaccurate P&L from 15-min scan gaps),
we emit signals with clear parameters (entry, TP, SL, max entry) and verify them
retroactively by checking actual price history between scans.

Signal lifecycle:
  ACTIVE      → Signal issued, awaiting resolution
  TP_HIT      → Price reached take-profit target (win)
  SL_HIT      → Price hit stop-loss level (loss)
  EXPIRED     → Max hold duration exceeded without TP or SL
  INVALIDATED → Entry price moved past max_entry before anyone could act
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
    Compute take-profit, stop-loss, and max-entry prices using actual
    mispricing data from the scoring layers — not just fixed percentages.

    TP logic (where we think fair value actually is):
      1. Cross-platform divergence: if Kalshi/Manifold says 40¢ and we're at 30¢,
         TP should be near 40¢ (the implied fair value from another market).
      2. Structural mispricing: combinatorial (YES+NO != $1) gives an implied
         fair price; TP moves toward the corrected price.
      3. Dislocation: if the score is velocity-driven (overreaction), we target
         a reversion proportional to the overreaction magnitude.
      4. Fallback: tier-based percentage if no layer gives a clear target.

    SL logic:
      - Base from tier config, but tightened for cheap tokens (gap risk)
      - Widened slightly when mispricing evidence is very strong (more room)

    Max entry:
      - Wider buffer for cheap tokens (thin books, wide spreads)

    Returns dict with tp_price, sl_price, max_entry_price and percentage values.
    """
    opp = opportunity or {}
    structural = opp.get('structural_detail', {})
    dislocation = opp.get('dislocation_detail', {})
    layer_scores = opp.get('layer_scores', {})
    edge_score = opp.get('edge_score', 0)

    # ── Take-Profit: based on where we think price SHOULD be ──
    # Collect implied fair-value estimates from each layer
    fair_value_estimates = []

    # 1. Cross-platform: Kalshi says price should be X
    kalshi = structural.get('cross_platform', {})
    kalshi_div = kalshi.get('divergence_pct', 0)  # in percentage points
    if kalshi_div > 3 and kalshi.get('direction') == 'kalshi_higher':
        # Kalshi is higher → Poly is underpriced → fair value is above entry
        implied_fv = entry_price + (kalshi_div / 100.0)
        fair_value_estimates.append(('kalshi', implied_fv, kalshi_div))

    # 2. Manifold cross-reference
    manifold = structural.get('manifold', {})
    manifold_div = manifold.get('divergence_pct', 0)
    if manifold_div > 3 and manifold.get('direction') in ('manifold_higher', 'kalshi_higher'):
        implied_fv = entry_price + (manifold_div / 100.0)
        fair_value_estimates.append(('manifold', implied_fv, manifold_div))

    # 3. Combinatorial: YES+NO gap means one side is overpriced
    combo = structural.get('combinatorial', {})
    combo_gap = combo.get('mispricing_pct', 0)
    if combo_gap > 2:
        # Implied correction: half the gap (conservative)
        implied_fv = entry_price + (combo_gap / 200.0)
        fair_value_estimates.append(('combo', implied_fv, combo_gap))

    # 4. Dislocation: price velocity overreaction → mean reversion target
    velocity_score = dislocation.get('price_velocity', 0)
    if velocity_score > 3:
        # Strong overreaction → expect partial reversion
        # Use the velocity magnitude as a proxy for how far to revert
        reversion_pct = min(0.40, velocity_score * 0.04)  # 3pts → 12%, 8pts → 32%
        implied_fv = entry_price * (1.0 + reversion_pct)
        fair_value_estimates.append(('velocity', implied_fv, reversion_pct * 100))

    # ── Determine TP from mispricing estimates ──
    tier_tp_pct = edge_tier.get('profit_target_pct', 25) / 100.0  # fallback

    if fair_value_estimates:
        # Weight by source reliability: cross-platform > combo > velocity
        weights = {'kalshi': 3.0, 'manifold': 2.0, 'combo': 2.0, 'velocity': 1.0}
        weighted_fv = sum(fv * weights.get(src, 1) for src, fv, _ in fair_value_estimates)
        total_weight = sum(weights.get(src, 1) for src, _, _ in fair_value_estimates)
        target_fv = weighted_fv / total_weight

        # TP = partway to fair value (80% of the way — leave some on the table)
        tp_price = entry_price + (target_fv - entry_price) * 0.80

        # But never less than the tier minimum
        tier_min_tp = entry_price * (1.0 + tier_tp_pct * 0.5)  # at least half of tier %
        tp_price = max(tp_price, tier_min_tp)

        # And never more than 2x the tier target (don't be unrealistic)
        tier_max_tp = entry_price * (1.0 + tier_tp_pct * 2.0)
        tp_price = min(tp_price, tier_max_tp)

        tp_pct = (tp_price - entry_price) / entry_price if entry_price > 0 else tier_tp_pct
    else:
        # No layer gave a clear target — use tier-based fallback
        tp_pct = tier_tp_pct
        tp_price = entry_price * (1.0 + tp_pct)

    # ── Stop-Loss ──
    sl_pct = abs(edge_tier.get('stop_loss_pct', -20)) / 100.0

    # Cheap token adjustment: tighter SL for thin order books
    if entry_price < 0.03:
        sl_pct = min(sl_pct, 0.12)   # Max 12% SL for penny tokens
    elif entry_price < 0.08:
        sl_pct = min(sl_pct, 0.15)   # Max 15% for cheap tokens

    # Widen SL slightly when mispricing evidence is very strong
    # (more conviction = give it more room to work)
    if edge_score >= 50 and len(fair_value_estimates) >= 2:
        sl_pct = min(sl_pct * 1.15, 0.30)  # Up to 15% wider, max 30%

    sl_price = entry_price * (1.0 - sl_pct)

    # ── Max Entry: don't chase ──
    if entry_price < 0.05:
        max_entry_buffer = 0.20   # 20% buffer for very cheap
    elif entry_price < 0.10:
        max_entry_buffer = 0.15
    else:
        max_entry_buffer = 0.10

    max_entry_price = entry_price * (1.0 + max_entry_buffer)

    # Clamp to valid range
    tp_price = min(tp_price, 0.99)
    sl_price = max(sl_price, 0.001)
    max_entry_price = min(max_entry_price, 0.99)

    return {
        'tp_price': round(tp_price, 4),
        'sl_price': round(sl_price, 4),
        'max_entry_price': round(max_entry_price, 4),
        'tp_pct': round(tp_pct * 100, 1),
        'sl_pct': round(-sl_pct * 100, 1),
        'tp_sources': [src for src, _, _ in fair_value_estimates],  # which layers drove the TP
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


def emit_signal(opportunity: dict, config: dict) -> dict:
    """
    Create a signal from a scored opportunity.
    Returns a fully-formed signal dict ready for persistence and messaging.
    """
    from src.portfolio import get_edge_tier, get_edge_tier_name

    edge_score = opportunity['edge_score']
    entry_price = opportunity['current_price']
    tier = get_edge_tier(edge_score, config)
    tier_name = get_edge_tier_name(edge_score, config)

    # Compute TP/SL/max entry — using actual mispricing data from layers
    targets = compute_tp_sl(entry_price, tier, tier_name, opportunity=opportunity)

    # Compute expiry: min of resolution date and now + max_hold_days
    # But NEVER set expiry in the past (can happen if resolution_date has passed)
    max_hold_days = tier.get('max_hold_days', 5)
    now = datetime.now(timezone.utc)
    hold_expiry = now + timedelta(days=max_hold_days)

    resolution_date = opportunity.get('resolution_date', '')
    if resolution_date:
        try:
            res = datetime.fromisoformat(str(resolution_date))
            if res.tzinfo is None:
                res = res.replace(tzinfo=timezone.utc)
            # Only use resolution date if it's in the future
            if res > now:
                expiry = min(hold_expiry, res)
            else:
                expiry = hold_expiry
        except (ValueError, TypeError):
            expiry = hold_expiry
    else:
        expiry = hold_expiry

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
        'max_entry_price': targets['max_entry_price'],
        'tp_price': targets['tp_price'],
        'sl_price': targets['sl_price'],
        'tp_pct': targets['tp_pct'],
        'sl_pct': targets['sl_pct'],
        'expiry': expiry.isoformat(),

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
        'tp_sources': targets.get('tp_sources', []),

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
        f"TP ${targets['tp_price']:.4f} ({targets['tp_pct']:+.0f}%) "
        f"SL ${targets['sl_price']:.4f} ({targets['sl_pct']:.0f}%) "
        f"Max entry ${targets['max_entry_price']:.4f} "
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
      1. Price history TP/SL: did price cross TP or SL during the signal's
         lifetime? Always checked FIRST, even if the signal has expired.
         → _check_price_history_for_resolution handles this, INCLUDING
           detecting resolution jumps (price leaping to ~$1 or ~$0 in one
           candle, which is a market settlement, not a genuine TP/SL hit).
      2. Market resolution fallback: if TP/SL wasn't hit but market has
         disappeared from active list with extreme price → settled.
      3. Expiry: ONLY if neither TP/SL was hit NOR market resolved,
         and past expiry datetime → close with last known price.

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

        # Check if signal has expired (used later as fallback)
        is_expired = False
        try:
            expiry = _parse_ts(signal.get('expiry', ''))
            is_expired = now >= expiry
        except (ValueError, TypeError):
            pass

        # 1. Fetch price history and check TP/SL FIRST (even if expired)
        #    A signal that expired but hit TP during its lifetime is a WIN, not an expiry.
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

            # 4. Expiry fallback: TP/SL wasn't hit in price history, market not gone
            if is_expired:
                last_price = signal.get('current_price', entry)
                lp = safe_float(history[-1].get('price', 0))
                if lp > 0:
                    last_price = lp
                signal['status'] = 'expired'
                signal['resolved_at'] = now.isoformat()
                signal['resolution_type'] = 'expired'
                signal['final_price'] = round(last_price, 4)
                signal['hypothetical_pnl_pct'] = round(
                    ((last_price - entry) / entry * 100) if entry > 0 else 0, 2
                )
                newly_resolved.append(signal)
                logger.info(
                    f"SIGNAL EXPIRED: {signal['signal_id']} "
                    f"pnl={signal['hypothetical_pnl_pct']:+.1f}% "
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

        # 6. Expiry fallback (no price history path)
        if is_expired:
            last_price = signal.get('current_price', entry)
            cp = current_prices.get(token_id, 0)
            if cp > 0:
                last_price = cp
            signal['status'] = 'expired'
            signal['resolved_at'] = now.isoformat()
            signal['resolution_type'] = 'expired'
            signal['final_price'] = round(last_price, 4)
            signal['hypothetical_pnl_pct'] = round(
                ((last_price - entry) / entry * 100) if entry > 0 else 0, 2
            )
            newly_resolved.append(signal)
            logger.info(
                f"SIGNAL EXPIRED: {signal['signal_id']} "
                f"pnl={signal['hypothetical_pnl_pct']:+.1f}% "
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
            'total_expired': 0,
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
        'total_signals': 0, 'total_wins': 0, 'total_losses': 0, 'total_expired': 0,
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
        elif rt == 'expired':
            stats['total_expired'] = stats.get('total_expired', 0) + 1

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
    expired = [s for s in resolved if s.get('resolution_type') == 'expired']

    # Market-resolved signals split into wins (pnl > 0) and losses
    mr_wins = [s for s in market_resolved if s.get('hypothetical_pnl_pct', 0) > 0]
    mr_losses = [s for s in market_resolved if s.get('hypothetical_pnl_pct', 0) <= 0]

    wins = tp_hits + mr_wins
    losses = sl_hits + mr_losses

    total_resolved = len(resolved)
    win_count = len(wins)
    loss_count = len(losses)

    # Win rate excludes expired (they're inconclusive)
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
        'expired': len(expired),
        'market_resolved': len(market_resolved),
        'win_rate': round(win_rate, 3),
        'avg_win_pct': round(avg_win_pct, 2),
        'avg_loss_pct': round(avg_loss_pct, 2),
        'ev_per_signal': round(ev_per_signal, 2),
        'by_tier': tier_stats,
    }
