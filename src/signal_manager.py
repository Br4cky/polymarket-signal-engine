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
) -> dict:
    """
    Compute take-profit, stop-loss, and max-entry prices from edge tier config.

    For cheap tokens, tighter SL to account for gap risk on thin order books.
    Max entry = entry + buffer so users don't chase into moved markets.

    Returns dict with tp_price, sl_price, max_entry_price and percentage values.
    """
    tp_pct = edge_tier.get('profit_target_pct', 25) / 100.0
    sl_pct = abs(edge_tier.get('stop_loss_pct', -20)) / 100.0

    # Cheap token adjustment (mirrors portfolio.py _smart_stop_loss)
    if entry_price < 0.03:
        sl_pct = min(sl_pct, 0.12)   # Max 12% SL for penny tokens
    elif entry_price < 0.08:
        sl_pct = min(sl_pct, 0.15)   # Max 15% for cheap tokens

    tp_price = entry_price * (1.0 + tp_pct)
    sl_price = entry_price * (1.0 - sl_pct)

    # Max entry: don't chase — wider buffer for cheap tokens (wider spreads)
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
        if d_detail.get('time_decay', 0) > 2:
            sub.append("time decay")
        if d_detail.get('order_book', 0) > 2:
            sub.append("book imbalance")
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

    # Compute TP/SL/max entry
    targets = compute_tp_sl(entry_price, tier, tier_name)

    # Compute expiry: min of resolution date and now + max_hold_days
    max_hold_days = tier.get('max_hold_days', 5)
    hold_expiry = datetime.now(timezone.utc) + timedelta(days=max_hold_days)

    resolution_date = opportunity.get('resolution_date', '')
    if resolution_date:
        try:
            res = datetime.fromisoformat(str(resolution_date))
            if res.tzinfo is None:
                res = res.replace(tzinfo=timezone.utc)
            expiry = min(hold_expiry, res)
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
    """Parse a timestamp (int/float epoch or ISO string) to datetime."""
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    ts_str = str(ts).strip()
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
    sl_hit_time = None
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
                continue

        # Track extremes
        peak_price = max(peak_price, price)
        trough_price = min(trough_price, price)

        # Check TP hit (price >= tp_price)
        if tp_hit_time is None and price >= tp_price:
            tp_hit_time = ts

        # Check SL hit (price <= sl_price)
        if sl_hit_time is None and price <= sl_price:
            sl_hit_time = ts

    # Determine resolution
    if tp_hit_time is not None and sl_hit_time is not None:
        # Both hit — which came first?
        try:
            tp_t = _parse_ts(tp_hit_time)
            sl_t = _parse_ts(sl_hit_time)
            if tp_t <= sl_t:
                return _make_resolution('tp_hit', tp_price, peak_price, trough_price, entry_price, tp_hit_time)
            else:
                return _make_resolution('sl_hit', sl_price, peak_price, trough_price, entry_price, sl_hit_time)
        except (ValueError, TypeError):
            # Can't determine order — take TP (benefit of the doubt)
            return _make_resolution('tp_hit', tp_price, peak_price, trough_price, entry_price, tp_hit_time)

    if tp_hit_time is not None:
        return _make_resolution('tp_hit', tp_price, peak_price, trough_price, entry_price, tp_hit_time)

    if sl_hit_time is not None:
        return _make_resolution('sl_hit', sl_price, peak_price, trough_price, entry_price, sl_hit_time)

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
) -> Tuple[List[dict], List[dict]]:
    """
    Verify all active signals against price history.

    For each active signal:
      1. Check if signal expired (past expiry datetime)
      2. Fetch price history (CLOB 60s candles) since signal was issued
      3. Check if TP or SL was hit at any point
      4. Update peak/trough tracking if still active

    Args:
        signals_state: Full signals state dict with 'active' list
        fetch_price_history: Callable(token_id) → List[{timestamp, price}]

    Returns:
        (newly_resolved, still_active) — lists of signal dicts
    """
    active = signals_state.get('active', [])
    newly_resolved = []
    still_active = []
    now = datetime.now(timezone.utc)

    for signal in active:
        token_id = signal['token_id']

        # 1. Check expiry
        try:
            expiry = _parse_ts(signal.get('expiry', ''))
            if now >= expiry:
                # Expired — fetch last known price for final P&L
                history = []
                try:
                    history = fetch_price_history(token_id)
                except Exception:
                    pass

                last_price = signal.get('current_price', signal['entry_price'])
                if history:
                    lp = safe_float(history[-1].get('price', 0))
                    if lp > 0:
                        last_price = lp

                entry = signal['entry_price']
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
        except (ValueError, TypeError):
            pass

        # 2. Fetch price history and check TP/SL
        try:
            history = fetch_price_history(token_id)
        except Exception as e:
            logger.warning(f"Price history failed for signal {signal['signal_id']}: {e}")
            still_active.append(signal)
            continue

        if not history:
            still_active.append(signal)
            continue

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

            won = 'WIN' if result['resolution_type'] == 'tp_hit' else 'LOSS'
            logger.info(
                f"SIGNAL {won}: {signal['signal_id']} "
                f"{result['resolution_type']} {result['hypothetical_pnl_pct']:+.1f}% "
                f"(peak={result['peak_price']:.4f} trough={result['trough_price']:.4f}) "
                f"({signal['question'][:40]}...)"
            )
        else:
            # Still active — update tracking
            signal['peak_price'] = result.get('peak_price', signal.get('peak_price'))
            signal['trough_price'] = result.get('trough_price', signal.get('trough_price'))

            # Update current price from latest history point
            if history:
                lp = safe_float(history[-1].get('price', 0))
                if lp > 0:
                    signal['current_price'] = round(lp, 4)
                    entry = signal['entry_price']
                    signal['live_pnl_pct'] = round(
                        ((lp - entry) / entry * 100) if entry > 0 else 0, 2
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

    wins = [s for s in resolved if s.get('resolution_type') == 'tp_hit']
    losses = [s for s in resolved if s.get('resolution_type') == 'sl_hit']
    expired = [s for s in resolved if s.get('resolution_type') == 'expired']

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
        tier_wins = [s for s in tier_resolved if s.get('resolution_type') == 'tp_hit']
        tier_losses = [s for s in tier_resolved if s.get('resolution_type') == 'sl_hit']
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
        'win_rate': round(win_rate, 3),
        'avg_win_pct': round(avg_win_pct, 2),
        'avg_loss_pct': round(avg_loss_pct, 2),
        'ev_per_signal': round(ev_per_signal, 2),
        'by_tier': tier_stats,
    }
