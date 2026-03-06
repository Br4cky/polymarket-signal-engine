"""
Single-Pool Repricing Portfolio
One unified capital pool trading all mispricings. Position sizing, profit targets,
stop losses, and hold duration driven by edge score conviction tiers.
"""

import json
import math
import os
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

from src.utils import safe_float

logger = logging.getLogger(__name__)


# ─── Edge Tier Helpers ─────────────────────────────────────────────────────

def get_edge_tier(edge_score: float, config: dict) -> dict:
    """Return the matching edge tier config for a given edge score."""
    tiers = config.get('trading', {}).get('edge_tiers', {})
    # Check tiers from highest to lowest
    for tier_name in ['high', 'medium', 'low']:
        tier = tiers.get(tier_name, {})
        if edge_score >= tier.get('min_edge', 999):
            return tier
    # Default fallback (low tier values)
    return {
        'min_edge': 20, 'size_multiplier': 0.6,
        'profit_target_pct': 15, 'stop_loss_pct': -15
    }


def get_edge_tier_name(edge_score: float, config: dict) -> str:
    """Return the tier name (high/medium/low) for a given edge score."""
    tiers = config.get('trading', {}).get('edge_tiers', {})
    for tier_name in ['high', 'medium', 'low']:
        tier = tiers.get(tier_name, {})
        if edge_score >= tier.get('min_edge', 999):
            return tier_name
    return 'low'


# ─── Fund & Portfolio Creation ──────────────────────────────────────────────

def create_fund(name: str, capital: float) -> dict:
    """Create a fresh fund state."""
    return {
        'name': name,
        'capital': capital,
        'available_cash': capital,
        'positions': [],
        'realized_trades': [],
        'win_count': 0,
        'loss_count': 0,
        'peak_equity': capital,
        'max_drawdown': 0.0,
        'equity_history': [
            {'timestamp': datetime.now(timezone.utc).isoformat(), 'equity': capital}
        ]
    }


def create_portfolio(config: dict) -> dict:
    """Create a fresh single-pool portfolio."""
    pool_cfg = config.get('funds', {}).get('main_pool', {})
    capital = pool_cfg.get('capital', config.get('portfolio', {}).get('total_capital', 5000))

    return {
        'main_pool': create_fund(
            name=pool_cfg.get('name', 'Repricing Engine'),
            capital=capital
        ),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'last_updated': datetime.now(timezone.utc).isoformat()
    }


def _migrate_dual_fund(data: dict, config: dict) -> dict:
    """Migrate old dual-fund (fund_a/fund_b) portfolio to single main_pool."""
    fund_a = data.get('fund_a', {})
    fund_b = data.get('fund_b', {})

    # Merge positions
    all_positions = list(fund_a.get('positions', [])) + list(fund_b.get('positions', []))
    # Backfill edge_score_at_entry if missing
    for pos in all_positions:
        if 'edge_score_at_entry' not in pos or not pos['edge_score_at_entry']:
            pos['edge_score_at_entry'] = 25  # Conservative default

    # Merge realized trades
    all_trades = list(fund_a.get('realized_trades', [])) + list(fund_b.get('realized_trades', []))
    all_trades.sort(key=lambda t: t.get('exit_timestamp', ''))

    # Merge equity
    capital_a = fund_a.get('capital', 0)
    capital_b = fund_b.get('capital', 0)
    total_capital = capital_a + capital_b
    cash_a = fund_a.get('available_cash', 0)
    cash_b = fund_b.get('available_cash', 0)

    # Merge equity history (interleave by timestamp, keep last 500)
    hist_a = fund_a.get('equity_history', [])
    hist_b = fund_b.get('equity_history', [])
    # Create combined equity snapshots — sum at each timestamp
    combined_hist = []
    for h in hist_a[-250:]:
        combined_hist.append(h)
    if not combined_hist:
        combined_hist.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'equity': total_capital
        })

    pool = {
        'name': 'Repricing Engine',
        'capital': total_capital,
        'available_cash': round(cash_a + cash_b, 2),
        'positions': all_positions,
        'realized_trades': all_trades,
        'win_count': fund_a.get('win_count', 0) + fund_b.get('win_count', 0),
        'loss_count': fund_a.get('loss_count', 0) + fund_b.get('loss_count', 0),
        'peak_equity': total_capital,  # Reset peak — new structure
        'max_drawdown': 0.0,
        'equity_history': combined_hist[-500:]
    }

    migrated = {
        'main_pool': pool,
        'created_at': data.get('created_at', datetime.now(timezone.utc).isoformat()),
        'last_updated': datetime.now(timezone.utc).isoformat()
    }

    logger.info(
        f"MIGRATION: Merged fund_a ({len(fund_a.get('positions', []))} pos) + "
        f"fund_b ({len(fund_b.get('positions', []))} pos) → main_pool "
        f"({len(all_positions)} pos, {len(all_trades)} trades, "
        f"${pool['available_cash']:.2f} cash)"
    )
    return migrated


def load_portfolio(filepath: str, config: dict) -> dict:
    """Load portfolio from file or create new. Auto-migrates old dual-fund format."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Check for old dual-fund structure → migrate
            if 'fund_a' in data and 'fund_b' in data and 'main_pool' not in data:
                logger.info("Detected old dual-fund portfolio, migrating to single pool...")
                migrated = _migrate_dual_fund(data, config)
                save_portfolio(migrated, filepath)
                return migrated

            # New single-pool structure
            if 'main_pool' in data and 'available_cash' in data.get('main_pool', {}):
                # Check if config capital changed
                cfg_cap = config.get('funds', {}).get('main_pool', {}).get(
                    'capital', config.get('portfolio', {}).get('total_capital', 5000))
                if data['main_pool'].get('capital') != cfg_cap:
                    logger.warning(
                        f"Capital changed (config: {cfg_cap}, "
                        f"portfolio: {data['main_pool'].get('capital')}). "
                        f"Resetting portfolio."
                    )
                    return create_portfolio(config)
                return data
            else:
                logger.warning("Portfolio file has invalid structure, creating new")
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to load portfolio, creating new")
    return create_portfolio(config)


def save_portfolio(portfolio: dict, filepath: str):
    """Save portfolio state to file."""
    portfolio['last_updated'] = datetime.now(timezone.utc).isoformat()
    with open(filepath, 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)


# ─── Position Sizing (Half-Kelly) ───────────────────────────────────────────

def _hold_duration_decay(entry_timestamp: str) -> float:
    """
    Adjust stop-loss tightness based on how long a position has been held.

    Repricing strategy: if the market hasn't corrected quickly, the thesis
    is weakening. Tighten stops progressively.
    - 0-24h: 1.0 (fresh signal, full stop width)
    - 24-48h: 0.85 (should have repriced by now)
    - 48-72h: 0.60 (thesis weakening)
    - 72h+: 0.30 (approaching forced exit)
    """
    try:
        entry = datetime.fromisoformat(str(entry_timestamp))
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        hours_held = (datetime.now(timezone.utc) - entry).total_seconds() / 3600.0
    except (ValueError, TypeError):
        hours_held = 0

    if hours_held <= 24:
        return 1.0
    elif hours_held <= 48:
        return 0.85
    elif hours_held <= 72:
        return 0.60
    else:
        return 0.30


def _conviction_multiplier(layer_scores: dict) -> float:
    """
    Scale position size by signal conviction quality.

    Multi-layer confirmation = higher conviction = larger size.
    Single-layer signal = lower conviction = smaller size.
    """
    active_layers = 0
    if layer_scores.get('structural', 0) > 2:
        active_layers += 1
    if layer_scores.get('smart_money', 0) > 2:
        active_layers += 1
    if layer_scores.get('dislocation', 0) > 3:
        active_layers += 1
    if layer_scores.get('external', 0) > 1:
        active_layers += 1

    conviction_map = {0: 0.3, 1: 0.5, 2: 0.8, 3: 1.1, 4: 1.3}
    return conviction_map.get(active_layers, 1.0)


def kelly_position_size(
    edge_score: float,
    current_price: float,
    available_cash: float,
    num_positions: int,
    config: dict,
    layer_scores: dict = None
) -> float:
    """
    Calculate position size using half-Kelly criterion.
    Edge score drives the tier multiplier (replaces old band-based sizing).

    Returns position size in USD (0 if no trade warranted).
    """
    max_pct = config.get('portfolio', {}).get('max_position_size_pct', 4) / 100.0
    kelly_frac = config.get('portfolio', {}).get('kelly_fraction', 0.4)
    max_positions = config.get('portfolio', {}).get('max_positions', 50)

    if edge_score < config.get('scoring', {}).get('edge_threshold', 20):
        return 0.0

    # Map edge score to estimated win probability
    estimated_win_prob = 0.50 + (edge_score - 50) / 100.0 * 0.40

    # Payoff odds from price
    if current_price <= 0.001 or current_price >= 0.999:
        return 0.0
    payoff_odds = (1.0 - current_price) / current_price

    # Kelly formula: f* = (p * b - q) / b
    q = 1.0 - estimated_win_prob
    kelly_full = (estimated_win_prob * payoff_odds - q) / payoff_odds

    if kelly_full <= 0:
        return 0.0

    kelly_adjusted = kelly_full * kelly_frac
    position_size = available_cash * kelly_adjusted

    # Cap at max percentage
    max_size = available_cash * max_pct
    position_size = min(position_size, max_size)

    # Diversification: reduce as positions grow
    diversification = 1.0 - (num_positions / (max_positions * 1.5))
    position_size *= max(0.3, diversification)

    # Edge tier sizing: higher edge → bigger bet
    tier = get_edge_tier(edge_score, config)
    position_size *= tier.get('size_multiplier', 1.0)

    # Conviction sizing: multi-layer confirmation = larger bet
    if layer_scores:
        position_size *= _conviction_multiplier(layer_scores)

    # Thin market haircut: cheap tokens gap through stops, size down
    if current_price < 0.03:
        position_size *= 0.4  # Penny tokens: 40% of normal size
    elif current_price < 0.08:
        position_size *= 0.65  # Cheap tokens: 65% of normal size

    position_size = min(position_size, available_cash)

    return max(0.0, round(position_size, 2))


# ─── Trade Execution (Paper) ────────────────────────────────────────────────

def execute_paper_trade(
    fund: dict,
    opportunity: dict,
    config: dict
) -> Tuple[dict, Optional[dict]]:
    """Execute a simulated buy trade."""
    slippage = config.get('trading', {}).get('slippage_pct', 0.5) / 100.0

    size = kelly_position_size(
        edge_score=opportunity['edge_score'],
        current_price=opportunity['current_price'],
        available_cash=fund['available_cash'],
        num_positions=len(fund['positions']),
        config=config,
        layer_scores=opportunity.get('layer_scores', {})
    )

    if size < 5.0:
        return fund, None

    entry_price = opportunity['current_price'] * (1.0 + slippage)
    entry_price = min(entry_price, 0.99)

    if entry_price <= 0.001:
        return fund, None

    shares = size / entry_price

    position = {
        'position_id': str(uuid.uuid4())[:8],
        'market_id': opportunity['market_id'],
        'question': opportunity['question'],
        'token_id': opportunity['token_id'],
        'outcome': opportunity['outcome'],
        'convexity_band': opportunity.get('convexity_band', ''),  # metadata only
        'entry_price': round(entry_price, 4),
        'entry_timestamp': datetime.now(timezone.utc).isoformat(),
        'shares': round(shares, 2),
        'entry_usd': round(size, 2),
        'current_price': opportunity['current_price'],
        'current_value': round(shares * opportunity['current_price'], 2),
        'unrealized_pnl': 0.0,
        'unrealized_pnl_pct': 0.0,
        'edge_score_at_entry': opportunity['edge_score'],
        'edge_tier_at_entry': get_edge_tier_name(opportunity['edge_score'], config),
        'layer_scores_at_entry': opportunity.get('layer_scores', {}),
        'days_to_close': opportunity.get('days_to_close', 0),
        'resolution_date': opportunity.get('resolution_date', ''),
        'potential_multiple': opportunity.get('potential_multiple', 0),
        'slug': opportunity.get('slug', '')
    }

    fund['available_cash'] -= size
    fund['available_cash'] = round(fund['available_cash'], 2)
    fund['positions'].append(position)

    logger.info(
        f"Paper trade: BUY {opportunity['outcome']} "
        f"${size:.2f} @ {entry_price:.4f} "
        f"[edge={opportunity['edge_score']:.0f}, tier={position['edge_tier_at_entry']}] "
        f"({opportunity['question'][:50]}...)"
    )

    return fund, position


# ─── Mark to Market ─────────────────────────────────────────────────────────

def update_fund_positions(fund: dict, current_prices: Dict[str, float]):
    """Update all positions with current market prices."""
    now = datetime.now(timezone.utc)

    for pos in fund['positions']:
        token_id = pos['token_id']
        if token_id in current_prices:
            new_price = current_prices[token_id]
            pos['current_price'] = new_price
            pos['current_value'] = round(pos['shares'] * new_price, 2)
            pos['unrealized_pnl'] = round(pos['current_value'] - pos['entry_usd'], 2)
            if pos['entry_usd'] > 0:
                pos['unrealized_pnl_pct'] = round(
                    (pos['unrealized_pnl'] / pos['entry_usd']) * 100, 2
                )

        # Recalculate days_to_close from resolution_date (live countdown)
        res_date = pos.get('resolution_date', '')
        if res_date:
            try:
                res = datetime.fromisoformat(str(res_date))
                if res.tzinfo is None:
                    res = res.replace(tzinfo=timezone.utc)
                pos['days_to_close'] = max(0, (res - now).days)
            except (ValueError, TypeError):
                pass

    _recalculate_fund_metrics(fund)


def _recalculate_fund_metrics(fund: dict):
    """Recalculate fund-level metrics."""
    total_pos_value = sum(p['current_value'] for p in fund['positions'])
    equity = fund['available_cash'] + total_pos_value

    if equity > fund.get('peak_equity', fund['capital']):
        fund['peak_equity'] = equity

    peak = fund.get('peak_equity', fund['capital'])
    if peak > 0:
        dd = ((equity - peak) / peak) * 100
        if dd < fund.get('max_drawdown', 0):
            fund['max_drawdown'] = round(dd, 2)

    fund.setdefault('equity_history', [])
    fund['equity_history'].append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'equity': round(equity, 2)
    })
    if len(fund['equity_history']) > 500:
        fund['equity_history'] = fund['equity_history'][-500:]


def update_portfolio(portfolio: dict, current_prices: Dict[str, float]):
    """Update main pool with current prices."""
    if 'main_pool' in portfolio:
        update_fund_positions(portfolio['main_pool'], current_prices)


# ─── Cooldown Check ──────────────────────────────────────────────────────────

def is_on_cooldown(fund: dict, market_id: str, cooldown_hours: int = 24) -> bool:
    """
    Check if a market is on stop-out cooldown.
    Returns True if the market was stopped out within the last `cooldown_hours`.
    """
    cooldowns = fund.get('stop_cooldowns', {})
    ts_str = cooldowns.get(market_id)
    if not ts_str:
        return False
    try:
        ts = datetime.fromisoformat(str(ts_str))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hours_since = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        return hours_since < cooldown_hours
    except (ValueError, TypeError):
        return False


# ─── Position Closing ───────────────────────────────────────────────────────

def close_position(fund: dict, position_id: str, exit_price: float, reason: str = 'manual') -> Optional[dict]:
    """Close a position and return the trade record."""
    pos_index = None
    for i, pos in enumerate(fund['positions']):
        if pos['position_id'] == position_id:
            pos_index = i
            break

    if pos_index is None:
        return None

    pos = fund['positions'].pop(pos_index)
    exit_value = pos['shares'] * exit_price
    pnl = exit_value - pos['entry_usd']
    pnl_pct = (pnl / pos['entry_usd'] * 100) if pos['entry_usd'] > 0 else 0

    trade = {
        'position_id': pos['position_id'],
        'market_id': pos['market_id'],
        'question': pos.get('question', ''),
        'outcome': pos['outcome'],
        'convexity_band': pos.get('convexity_band', ''),
        'entry_price': pos['entry_price'],
        'exit_price': round(exit_price, 4),
        'entry_timestamp': pos['entry_timestamp'],
        'exit_timestamp': datetime.now(timezone.utc).isoformat(),
        'shares': pos['shares'],
        'entry_usd': pos['entry_usd'],
        'exit_usd': round(exit_value, 2),
        'pnl_usd': round(pnl, 2),
        'pnl_pct': round(pnl_pct, 2),
        'win': pnl > 0,
        'reason': reason,
        'edge_score_at_entry': pos.get('edge_score_at_entry', 0),
        'edge_tier_at_entry': pos.get('edge_tier_at_entry', 'low'),
        'layer_scores_at_entry': pos.get('layer_scores_at_entry', {}),
    }

    fund['available_cash'] += exit_value
    fund['available_cash'] = round(fund['available_cash'], 2)

    if pnl > 0:
        fund['win_count'] = fund.get('win_count', 0) + 1
    else:
        fund['loss_count'] = fund.get('loss_count', 0) + 1

    fund.setdefault('realized_trades', []).append(trade)

    # Track stop-out cooldowns — prevent re-entering same market after a loss
    if not trade['win'] and 'stop_loss' in reason:
        cooldowns = fund.setdefault('stop_cooldowns', {})
        cooldowns[trade['market_id']] = datetime.now(timezone.utc).isoformat()
        # Clean old cooldowns (>48h) to prevent unbounded growth
        cutoff = datetime.now(timezone.utc).timestamp() - 48 * 3600
        cooldowns_clean = {}
        for mid, ts in cooldowns.items():
            try:
                t = datetime.fromisoformat(str(ts))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if t.timestamp() > cutoff:
                    cooldowns_clean[mid] = ts
            except (ValueError, TypeError):
                pass
        fund['stop_cooldowns'] = cooldowns_clean

    _recalculate_fund_metrics(fund)

    logger.info(
        f"Closed {position_id}: "
        f"{'WIN' if pnl > 0 else 'LOSS'} ${pnl:+.2f} ({pnl_pct:+.1f}%) [{reason}]"
    )
    return trade


def _smart_stop_loss(pos: dict, config: dict) -> float:
    """
    Compute dynamic stop-loss based on edge score at entry.
    Higher edge = wider stop (more confident). Lower edge = tighter stop.
    Hold-duration decay tightens stops as positions age.

    For very cheap tokens (< $0.05), use tighter stops since these
    can gap 50%+ between scans on thin order books.
    """
    edge = pos.get('edge_score_at_entry', 25)
    tier = get_edge_tier(edge, config)
    base_stop = tier.get('stop_loss_pct', -20)

    # Cheap token adjustment — these gap massively, tighter stop triggers sooner
    entry_price = pos.get('entry_price', 0.5)
    if entry_price < 0.03:
        base_stop = max(base_stop, -12)  # Tighter: -12% max for penny tokens
    elif entry_price < 0.08:
        base_stop = max(base_stop, -15)  # -15% for cheap tokens

    # Hold-duration tightening
    entry_ts = pos.get('entry_timestamp', '')
    decay = _hold_duration_decay(entry_ts)
    stop = base_stop * decay

    return round(stop, 1)


def _repricing_profit_target(pos: dict, config: dict) -> float:
    """
    Profit target based on edge score at entry.
    Higher edge = higher target (more confident in the mispricing).
    """
    edge = pos.get('edge_score_at_entry', 25)
    tier = get_edge_tier(edge, config)
    return tier.get('profit_target_pct', 25)


def auto_close_positions(fund: dict, current_prices: Dict[str, float], config: dict) -> List[dict]:
    """
    Repricing auto-close: edge-tier profit targets, aggressive trailing stops,
    and hold-duration-aware stop losses.
    """
    to_close = []

    for pos in fund['positions']:
        pnl_pct = pos.get('unrealized_pnl_pct', 0)
        edge = pos.get('edge_score_at_entry', 25)
        tier = get_edge_tier(edge, config)

        # ── 1. Repricing profit target ──
        profit_target = _repricing_profit_target(pos, config)
        if pnl_pct >= profit_target:
            to_close.append((pos['position_id'], f'profit_target_{profit_target:.0f}pct'))
            continue

        # ── 3. Aggressive trailing stop ──
        peak = pos.get('peak_pnl_pct', pnl_pct)
        if pnl_pct > peak:
            pos['peak_pnl_pct'] = pnl_pct
            peak = pnl_pct

        trailing_floor = None
        if peak >= 80:
            trailing_floor = peak * 0.50
        elif peak >= 50:
            trailing_floor = peak * 0.30
        elif peak >= 30:
            trailing_floor = peak * 0.15

        if trailing_floor is not None and pnl_pct < trailing_floor:
            to_close.append((pos['position_id'], f'trailing_stop_from_{peak:.0f}pct'))
            continue

        # ── 4. Smart stop loss: edge-tier + hold-duration tightening ──
        stop_loss = _smart_stop_loss(pos, config)
        if pnl_pct <= stop_loss:
            to_close.append((pos['position_id'], f'stop_loss_{stop_loss:.0f}pct'))
            continue

        # ── 5. Gap-risk early exit for thin markets ──
        # Penny tokens can gap 10-20% between 15-min scans on thin order books.
        # If we're within a gap-risk buffer of the stop, close early to avoid
        # getting gapped through and taking a much larger loss.
        entry_price = pos.get('entry_price', 0.5)
        if entry_price < 0.03:
            gap_buffer = 3.0  # Close if within 3% of stop for penny tokens
        elif entry_price < 0.08:
            gap_buffer = 2.0  # Close if within 2% of stop for cheap tokens
        else:
            gap_buffer = 0.0  # No early exit for liquid markets

        if gap_buffer > 0 and pnl_pct <= (stop_loss + gap_buffer):
            to_close.append((pos['position_id'], f'gap_risk_exit_{pnl_pct:.0f}pct_near_stop_{stop_loss:.0f}pct'))
            continue

    closed = []
    for pid, reason in to_close:
        try:
            pos = next((p for p in fund['positions'] if p['position_id'] == pid), None)
            if pos:
                exit_price = current_prices.get(pos['token_id'], pos['current_price'])
                trade = close_position(fund, pid, exit_price, reason)
                if trade:
                    closed.append(trade)
        except Exception as e:
            logger.error(f"Failed to close position {pid}: {e}")

    return closed


# ─── Portfolio Summary for Dashboard ──────────────────────────────────────

def fund_summary(fund: dict) -> dict:
    """Generate a clean summary of the pool for the dashboard."""
    total_pos_value = sum(p.get('current_value', 0) for p in fund.get('positions', []))
    equity = fund.get('available_cash', 0) + total_pos_value
    total_realized = sum(t.get('pnl_usd', 0) for t in fund.get('realized_trades', []))
    total_unrealized = sum(p.get('unrealized_pnl', 0) for p in fund.get('positions', []))
    total_pnl = total_realized + total_unrealized
    total_trades = fund.get('win_count', 0) + fund.get('loss_count', 0)

    capital = fund.get('capital', 5000) or 5000
    win_rate = fund.get('win_count', 0) / total_trades if total_trades > 0 else 0

    return {
        'name': fund.get('name', 'Repricing Engine'),
        'capital': capital,
        'cash': round(fund.get('available_cash', 0), 2),
        'equity': round(equity, 2),
        'pnl': round(total_pnl, 2),
        'total_pnl': round(total_pnl, 2),
        'win_rate': round(win_rate, 3),
        'win_count': fund.get('win_count', 0),
        'loss_count': fund.get('loss_count', 0),
        'open_positions': len(fund.get('positions', [])),
        'closed_trades': total_trades,
        'max_drawdown': fund.get('max_drawdown', 0),
        'peak_equity': fund.get('peak_equity', 5000),
        'equity_history': fund.get('equity_history', [])[-100:]
    }


def portfolio_summary(portfolio: dict) -> dict:
    """Generate portfolio summary from single pool."""
    pool = fund_summary(portfolio.get('main_pool', {}))

    return {
        'total_capital': pool['capital'],
        'combined_equity': pool['equity'],
        'combined_pnl': pool['pnl'],
        'combined_pnl_pct': round((pool['pnl'] / pool['capital']) * 100, 2) if pool['capital'] > 0 else 0,
        'combined_win_rate': pool['win_rate'],
        'combined_total_trades': pool['closed_trades'],
        'open_positions': pool['open_positions'],
        'pool': pool
    }
