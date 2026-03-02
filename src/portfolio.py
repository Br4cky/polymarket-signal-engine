"""
Dual-Fund Portfolio Simulator
Two virtual funds (5x Hunter and 10x Hunter) running simultaneously.
Half-Kelly position sizing, automatic profit-target/stop-loss, full P&L tracking.
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


# ─── Fund & Portfolio Creation ──────────────────────────────────────────────

def create_fund(name: str, band: str, capital: float) -> dict:
    """Create a fresh fund state."""
    return {
        'name': name,
        'band': band,
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
    """Create a fresh dual-fund portfolio."""
    funds_config = config.get('funds', {})

    fund_a_cfg = funds_config.get('fund_a', {})
    fund_b_cfg = funds_config.get('fund_b', {})

    return {
        'fund_a': create_fund(
            name=fund_a_cfg.get('name', '5x Hunter'),
            band=fund_a_cfg.get('band', '5x'),
            capital=fund_a_cfg.get('capital', 250)
        ),
        'fund_b': create_fund(
            name=fund_b_cfg.get('name', '10x Hunter'),
            band=fund_b_cfg.get('band', '10x'),
            capital=fund_b_cfg.get('capital', 250)
        ),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'last_updated': datetime.now(timezone.utc).isoformat()
    }


def load_portfolio(filepath: str, config: dict) -> dict:
    """Load portfolio from file or create new. Validates structure and capital."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Validate it has the expected structure
            if ('fund_a' in data and 'fund_b' in data
                    and 'available_cash' in data.get('fund_a', {})):
                # Check if config capital changed — if so, reset
                cfg_cap_a = config.get('funds', {}).get('fund_a', {}).get('capital', 250)
                cfg_cap_b = config.get('funds', {}).get('fund_b', {}).get('capital', 250)
                if (data['fund_a'].get('capital') != cfg_cap_a
                        or data['fund_b'].get('capital') != cfg_cap_b):
                    logger.warning(
                        f"Capital changed (config: {cfg_cap_a}/{cfg_cap_b}, "
                        f"portfolio: {data['fund_a'].get('capital')}/{data['fund_b'].get('capital')}). "
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

def _time_decay_multiplier(days_to_close: int) -> float:
    """
    Adjust position size based on resolution proximity.

    Near-resolution markets have binary risk — either they resolve in your
    favour (big win) or against (total loss). Size accordingly:
    - Far out: full size, plenty of time for information to work
    - Mid-range: slight reduction, increasing vol
    - Close: aggressive reduction, binary outcome imminent
    - Very close: minimal size, pure gamble territory
    """
    if days_to_close >= 30:
        return 1.0
    elif days_to_close >= 14:
        return 0.85
    elif days_to_close >= 7:
        return 0.65
    elif days_to_close >= 3:
        return 0.40
    else:
        return 0.20


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

    # 1 layer = 0.6x, 2 = 0.85x, 3 = 1.0x, 4 = 1.15x
    conviction_map = {0: 0.4, 1: 0.6, 2: 0.85, 3: 1.0, 4: 1.15}
    return conviction_map.get(active_layers, 1.0)


def kelly_position_size(
    edge_score: float,
    current_price: float,
    convexity_band: str,
    available_cash: float,
    num_positions: int,
    config: dict,
    days_to_close: int = 30,
    layer_scores: dict = None
) -> float:
    """
    Calculate position size using half-Kelly criterion with time-aware
    and conviction-aware adjustments.

    Returns position size in USD (0 if no trade warranted).
    """
    max_pct = config.get('portfolio', {}).get('max_position_size_pct', 3) / 100.0
    kelly_frac = config.get('portfolio', {}).get('kelly_fraction', 0.5)
    max_positions = config.get('portfolio', {}).get('max_positions_per_fund', 15)

    if edge_score < config.get('scoring', {}).get('edge_threshold', 50):
        return 0.0

    if num_positions >= max_positions:
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

    # Band adjustment: smaller bets on riskier tails
    multipliers = {'20x': 0.5, '10x': 0.7, '5x': 1.0, '2x': 1.0, 'yield': 1.2}
    position_size *= multipliers.get(convexity_band, 1.0)

    # Time-aware sizing: reduce near resolution (binary risk increases)
    position_size *= _time_decay_multiplier(days_to_close)

    # Conviction sizing: multi-layer confirmation = larger bet
    if layer_scores:
        position_size *= _conviction_multiplier(layer_scores)

    position_size = min(position_size, available_cash)

    return max(0.0, round(position_size, 2))


# ─── Trade Execution (Paper) ────────────────────────────────────────────────

def execute_paper_trade(
    fund: dict,
    opportunity: dict,
    config: dict
) -> Tuple[dict, Optional[dict]]:
    """Execute a simulated buy trade on a specific fund."""
    slippage = config.get('trading', {}).get('slippage_pct', 0.5) / 100.0

    size = kelly_position_size(
        edge_score=opportunity['edge_score'],
        current_price=opportunity['current_price'],
        convexity_band=opportunity['convexity_band'],
        available_cash=fund['available_cash'],
        num_positions=len(fund['positions']),
        config=config,
        days_to_close=opportunity.get('days_to_close', 30),
        layer_scores=opportunity.get('layer_scores', {})
    )

    if size < 5.0:
        return fund, None

    # Paper trade: simulate limit order fill near current price.
    # Do NOT use the ask price — in thin prediction markets the best ask
    # can be 0.99 even when current_price is 0.15 (no active sellers).
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
        'convexity_band': opportunity['convexity_band'],
        'entry_price': round(entry_price, 4),
        'entry_timestamp': datetime.now(timezone.utc).isoformat(),
        'shares': round(shares, 2),
        'entry_usd': round(size, 2),
        'current_price': opportunity['current_price'],
        'current_value': round(shares * opportunity['current_price'], 2),
        'unrealized_pnl': 0.0,
        'unrealized_pnl_pct': 0.0,
        'edge_score_at_entry': opportunity['edge_score'],
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
        f"[{fund['name']}] Paper trade: BUY {opportunity['outcome']} "
        f"${size:.2f} @ {entry_price:.4f} ({opportunity['question'][:50]}...)"
    )

    return fund, position


# ─── Mark to Market ─────────────────────────────────────────────────────────

def update_fund_positions(fund: dict, current_prices: Dict[str, float]):
    """Update all positions in a fund with current market prices."""
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
    """Update both funds with current prices."""
    for fund_key in ['fund_a', 'fund_b']:
        if fund_key in portfolio:
            update_fund_positions(portfolio[fund_key], current_prices)


# ─── Position Closing ───────────────────────────────────────────────────────

def close_position(fund: dict, position_id: str, exit_price: float, reason: str = 'manual') -> Optional[dict]:
    """Close a position in a fund and return the trade record."""
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
        'layer_scores_at_entry': pos.get('layer_scores_at_entry', {}),
        'days_to_close_at_entry': pos.get('days_to_close', 0),
    }

    fund['available_cash'] += exit_value
    fund['available_cash'] = round(fund['available_cash'], 2)

    if pnl > 0:
        fund['win_count'] = fund.get('win_count', 0) + 1
    else:
        fund['loss_count'] = fund.get('loss_count', 0) + 1

    fund.setdefault('realized_trades', []).append(trade)
    _recalculate_fund_metrics(fund)

    logger.info(
        f"[{fund['name']}] Closed {position_id}: "
        f"{'WIN' if pnl > 0 else 'LOSS'} ${pnl:+.2f} ({pnl_pct:+.1f}%)"
    )
    return trade


def _smart_stop_loss(pos: dict, config: dict) -> float:
    """
    Compute dynamic stop-loss threshold based on convexity band and conviction.

    Key insight: a 20x token can absorb an -85% drawdown because the math
    still works — you paid $5 for a chance at $100, losing $4.25 is fine
    if the thesis holds. A 5x token losing -60% means the thesis is broken.

    High conviction (multi-layer confirmation at entry) → wider stops.
    Low conviction → tighter stops, cut losses faster.
    """
    band = pos.get('convexity_band', '5x')
    entry_layers = pos.get('layer_scores_at_entry', {})
    days = pos.get('days_to_close', 30)

    # Base stop loss by band (respects convexity math)
    base_stops = {
        '20x': -85,
        '10x': -75,
        '5x': -60,
        '2x': -40,
        'yield': -25,
    }
    stop = base_stops.get(band, -60)

    # Conviction adjustment: count active layers at entry
    active_layers = 0
    if entry_layers.get('structural', 0) > 2:
        active_layers += 1
    if entry_layers.get('smart_money', 0) > 2:
        active_layers += 1
    if entry_layers.get('dislocation', 0) > 3:
        active_layers += 1
    if entry_layers.get('external', 0) > 1:
        active_layers += 1

    # Low conviction → tighten stop by 25%
    # High conviction → keep full stop width
    if active_layers <= 1:
        stop *= 0.75  # e.g., -75% → -56%
    elif active_layers >= 3:
        stop *= 1.05  # Slightly wider for high conviction

    # Tighten near resolution (less time for thesis to play out)
    if days < 5:
        stop *= 0.80
    elif days < 10:
        stop *= 0.90

    return round(stop, 1)


def _trailing_profit_target(pos: dict, config: dict) -> float:
    """
    Dynamic profit target that scales with convexity band.

    20x tokens should target much higher exits than 5x tokens.
    Also implements a trailing mechanism: once up 100%+, lock in
    at least 50% of peak gain.
    """
    band = pos.get('convexity_band', '5x')

    # Base profit targets by band
    base_targets = {
        '20x': 500,   # 5x initial investment (not full 20x — take some off the table)
        '10x': 400,   # 4x
        '5x': 300,    # 3x (current default)
        '2x': 150,    # 1.5x
        'yield': 80,
    }
    return base_targets.get(band, 300)


def auto_close_positions(fund: dict, current_prices: Dict[str, float], config: dict) -> List[dict]:
    """
    Smart auto-close: band-aware stop-losses, conviction-weighted stops,
    and dynamic profit targets.
    """
    to_close = []

    for pos in fund['positions']:
        pnl_pct = pos.get('unrealized_pnl_pct', 0)

        # Dynamic profit target based on band
        profit_target = _trailing_profit_target(pos, config)
        if pnl_pct >= profit_target:
            to_close.append((pos['position_id'], f'profit_target_{profit_target:.0f}pct'))
            continue

        # Trailing stop: if position was ever up 100%+, lock in at least
        # half the peak gain. Track peak_pnl_pct on position.
        peak = pos.get('peak_pnl_pct', pnl_pct)
        if pnl_pct > peak:
            pos['peak_pnl_pct'] = pnl_pct
            peak = pnl_pct

        if peak >= 100 and pnl_pct < peak * 0.50:
            to_close.append((pos['position_id'], f'trailing_stop_from_{peak:.0f}pct'))
            continue

        # Smart stop loss: band + conviction + time aware
        stop_loss = _smart_stop_loss(pos, config)
        if pnl_pct <= stop_loss:
            to_close.append((pos['position_id'], f'stop_loss_{stop_loss:.0f}pct'))
            continue

    closed = []
    for pid, reason in to_close:
        pos = next((p for p in fund['positions'] if p['position_id'] == pid), None)
        if pos:
            exit_price = current_prices.get(pos['token_id'], pos['current_price'])
            trade = close_position(fund, pid, exit_price, reason)
            if trade:
                closed.append(trade)

    return closed


# ─── Fund Summary for Dashboard ─────────────────────────────────────────────

def fund_summary(fund: dict) -> dict:
    """Generate a clean summary of a fund for the dashboard."""
    total_pos_value = sum(p['current_value'] for p in fund.get('positions', []))
    equity = fund['available_cash'] + total_pos_value
    total_realized = sum(t.get('pnl_usd', 0) for t in fund.get('realized_trades', []))
    total_unrealized = sum(p.get('unrealized_pnl', 0) for p in fund.get('positions', []))
    total_pnl = total_realized + total_unrealized
    total_trades = fund.get('win_count', 0) + fund.get('loss_count', 0)

    capital = fund.get('capital', 250)
    win_rate = fund.get('win_count', 0) / total_trades if total_trades > 0 else 0

    return {
        'name': fund.get('name', ''),
        'band': fund.get('band', ''),
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
        'peak_equity': fund.get('peak_equity', 250),
        'equity_history': fund.get('equity_history', [])[-100:]
    }


def portfolio_summary(portfolio: dict) -> dict:
    """Generate combined portfolio summary."""
    fund_a = fund_summary(portfolio.get('fund_a', {}))
    fund_b = fund_summary(portfolio.get('fund_b', {}))

    combined_equity = fund_a['equity'] + fund_b['equity']
    combined_pnl = fund_a['pnl'] + fund_b['pnl']
    total_capital = fund_a['capital'] + fund_b['capital']
    total_wins = fund_a['win_count'] + fund_b['win_count']
    total_trades = fund_a['closed_trades'] + fund_b['closed_trades']

    return {
        'total_capital': total_capital,
        'combined_equity': round(combined_equity, 2),
        'combined_pnl': round(combined_pnl, 2),
        'combined_pnl_pct': round((combined_pnl / total_capital) * 100, 2) if total_capital > 0 else 0,
        'combined_win_rate': round(total_wins / total_trades, 3) if total_trades > 0 else 0,
        'combined_total_trades': total_trades,
        'fund_a': fund_a,
        'fund_b': fund_b
    }
