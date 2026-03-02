#!/usr/bin/env python3
"""Generate realistic sample data matching the dashboard's expected format."""

import json
import os
import random
from datetime import datetime, timezone, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(42)
now = datetime.now(timezone.utc)

# ── Sample markets ──
MARKETS = [
    ("Will Trump win the 2028 Republican primary?", "politics", "trump-2028-primary", "5x"),
    ("Will Bitcoin exceed $150k by July 2026?", "crypto", "btc-150k-july", "10x"),
    ("Will the Fed cut rates in Q2 2026?", "economics", "fed-rate-cut-q2", "5x"),
    ("Will Real Madrid win Champions League 2026?", "sports", "real-madrid-ucl-2026", "10x"),
    ("Will OpenAI IPO before 2027?", "tech", "openai-ipo-2027", "5x"),
    ("Will UK inflation drop below 2% by June 2026?", "economics", "uk-inflation-target", "10x"),
    ("Will Nvidia market cap exceed $5T in 2026?", "tech", "nvidia-5t-2026", "5x"),
    ("Will there be a US government shutdown in 2026?", "politics", "us-shutdown-2026", "10x"),
    ("Will Ethereum hit $10k in 2026?", "crypto", "eth-10k-2026", "10x"),
    ("Will a Category 5 hurricane hit US in 2026?", "science", "cat5-hurricane-2026", "5x"),
]

def gen_opportunities():
    opps = []
    for i, (question, category, slug, band) in enumerate(MARKETS):
        if band == '5x':
            price = round(random.uniform(0.15, 0.25), 3)
            multiple = round(1.0 / price, 1)
        else:
            price = round(random.uniform(0.05, 0.12), 3)
            multiple = round(1.0 / price, 1)

        structural = round(random.uniform(5, 25), 1)
        smart_money = round(random.uniform(3, 20), 1)
        dislocation = round(random.uniform(4, 25), 1)
        external = round(random.uniform(1, 12), 1)

        # Composite edge (weighted, normalised to 100)
        edge = round(min(100, (
            (structural / 30) * 0.30 +
            (smart_money / 25) * 0.25 +
            (dislocation / 30) * 0.30 +
            (external / 15) * 0.15
        ) * 100), 1)

        opps.append({
            'market_id': f'0x{random.randint(0x100000, 0xffffff):06x}',
            'question': question,
            'category': category,
            'slug': slug,
            'token_id': f'tok_{i:03d}',
            'outcome': 'YES',
            'current_price': price,
            'bid': round(price - random.uniform(0.005, 0.02), 3),
            'ask': round(price + random.uniform(0.005, 0.02), 3),
            'spread_pct': round(random.uniform(1, 8), 1),
            'edge_score': edge,
            'layer_scores': {
                'structural': structural,
                'smart_money': smart_money,
                'dislocation': dislocation,
                'external': external
            },
            'structural_detail': {
                'combinatorial_score': round(random.uniform(0, 12), 1),
                'cross_platform_score': round(random.uniform(0, 12), 1),
                'structural_total': structural
            },
            'dislocation_detail': {
                'velocity_score': round(random.uniform(0, 7), 1),
                'volume_score': round(random.uniform(0, 6), 1),
                'orderbook_score': round(random.uniform(0, 4), 1),
                'trajectory_score': round(random.uniform(0, 4), 1),
                'time_decay_score': round(random.uniform(0, 5), 1),
                'dislocation_total': dislocation
            },
            'smart_money_detail': {
                'whale_score': round(random.uniform(0, 8), 1),
                'leaderboard_score': round(random.uniform(0, 7), 1),
                'concentration_score': round(random.uniform(0, 6), 1),
                'smart_money_total': smart_money
            },
            'external_detail': {
                'news_score': round(random.uniform(0, 7), 1),
                'trends_score': round(random.uniform(0, 6), 1),
                'external_total': external
            },
            'convexity_band': band,
            'potential_multiple': multiple,
            'days_to_close': random.randint(7, 180),
            'liquidity_usd': round(random.uniform(5000, 500000), 0),
            'volume_24h': round(random.uniform(1000, 100000), 0),
            'fund_assignment': 'fund_a' if band == '5x' else 'fund_b',
            'recommended_action': 'BUY',
            'timestamp': now.isoformat()
        })

    opps.sort(key=lambda x: x['edge_score'], reverse=True)
    return opps

def gen_equity_history(start_equity, days=30):
    history = [start_equity]
    eq = start_equity
    for _ in range(days):
        change = eq * random.uniform(-0.03, 0.04)
        eq = max(eq * 0.8, eq + change)
        history.append(round(eq, 2))
    return history

def gen_positions(fund_key, count, band):
    positions = []
    for i in range(count):
        if band == '5x':
            entry = round(random.uniform(0.15, 0.25), 3)
        else:
            entry = round(random.uniform(0.05, 0.12), 3)
        current = round(entry + random.uniform(-0.05, 0.10), 3)
        current = max(0.01, min(0.99, current))
        qty = random.randint(20, 200)
        cost = round(entry * qty, 2)
        value = round(current * qty, 2)

        positions.append({
            'token_id': f'{fund_key}_pos_{i}',
            'market_id': f'0x{random.randint(0x100000, 0xffffff):06x}',
            'question': random.choice(MARKETS)[0],
            'outcome': 'YES',
            'entry_price': entry,
            'current_price': current,
            'quantity': qty,
            'cost_basis': cost,
            'current_value': value,
            'unrealized_pnl': round(value - cost, 2),
            'edge_score_at_entry': round(random.uniform(52, 85), 1),
            'opened_at': (now - timedelta(days=random.randint(1, 20))).isoformat()
        })
    return positions

def gen_realized_trades(fund_key, count, band):
    trades = []
    for i in range(count):
        if band == '5x':
            entry = round(random.uniform(0.15, 0.25), 3)
        else:
            entry = round(random.uniform(0.05, 0.12), 3)

        # ~40% win rate with occasional big wins
        if random.random() < 0.4:
            exit_p = round(entry + random.uniform(0.10, 0.60), 3)
        else:
            exit_p = round(entry * random.uniform(0.1, 0.8), 3)

        exit_p = max(0.01, min(0.99, exit_p))
        qty = random.randint(20, 150)
        pnl = round((exit_p - entry) * qty, 2)
        ret = round(((exit_p - entry) / entry) * 100, 1)

        trades.append({
            'token_id': f'{fund_key}_trade_{i}',
            'market_id': f'0x{random.randint(0x100000, 0xffffff):06x}',
            'question': random.choice(MARKETS)[0],
            'outcome': 'YES',
            'entry_price': entry,
            'exit_price': exit_p,
            'quantity': qty,
            'pnl': pnl,
            'return_pct': ret,
            'closed_at': (now - timedelta(days=random.randint(0, 30))).isoformat(),
            'close_reason': random.choice(['profit_target', 'stop_loss', 'manual', 'resolution'])
        })
    trades.sort(key=lambda x: x['closed_at'], reverse=True)
    return trades

def main():
    opps = gen_opportunities()

    # Fund A (5x)
    fa_positions = gen_positions('fund_a', 3, '5x')
    fa_trades = gen_realized_trades('fund_a', 8, '5x')
    fa_wins = sum(1 for t in fa_trades if t['pnl'] > 0)
    fa_equity_hist = gen_equity_history(250, 30)
    fa_cash = round(250 - sum(p['cost_basis'] for p in fa_positions), 2)
    fa_equity = round(fa_cash + sum(p['current_value'] for p in fa_positions), 2)
    fa_total_pnl = round(fa_equity - 250, 2)

    # Fund B (10x)
    fb_positions = gen_positions('fund_b', 4, '10x')
    fb_trades = gen_realized_trades('fund_b', 6, '10x')
    fb_wins = sum(1 for t in fb_trades if t['pnl'] > 0)
    fb_equity_hist = gen_equity_history(250, 30)
    fb_cash = round(250 - sum(p['cost_basis'] for p in fb_positions), 2)
    fb_equity = round(fb_cash + sum(p['current_value'] for p in fb_positions), 2)
    fb_total_pnl = round(fb_equity - 250, 2)

    combined_equity = round(fa_equity + fb_equity, 2)
    combined_pnl = round(fa_total_pnl + fb_total_pnl, 2)
    total_trades = len(fa_trades) + len(fb_trades)
    total_wins = fa_wins + fb_wins

    # Statistics
    by_category = {}
    by_convexity = {}
    for opp in opps:
        cat = opp['category']
        band = opp['convexity_band']
        by_category.setdefault(cat, {'count': 0, '_total': 0})
        by_category[cat]['count'] += 1
        by_category[cat]['_total'] += opp['edge_score']
        by_convexity.setdefault(band, {'count': 0, '_total': 0, 'total_liquidity': 0})
        by_convexity[band]['count'] += 1
        by_convexity[band]['_total'] += opp['edge_score']
        by_convexity[band]['total_liquidity'] += opp['liquidity_usd']
    for v in by_category.values():
        v['avg_edge'] = round(v['_total'] / v['count'], 1) if v['count'] else 0
        del v['_total']
    for v in by_convexity.values():
        v['avg_edge'] = round(v['_total'] / v['count'], 1) if v['count'] else 0
        del v['_total']

    signal_data = {
        'metadata': {
            'timestamp': now.isoformat(),
            'markets_scanned': 247,
            'tokens_scored': len(opps),
            'engine_version': '2.0.0'
        },
        'portfolio': {
            'combined_equity': combined_equity,
            'combined_pnl': combined_pnl,
            'combined_win_rate': round(total_wins / total_trades, 3) if total_trades else 0,
            'combined_total_trades': total_trades,
            'fund_a': {
                'name': 'Moderate Hunter',
                'band': '5x',
                'equity': fa_equity,
                'pnl': fa_total_pnl,
                'win_rate': round(fa_wins / len(fa_trades), 3) if fa_trades else 0,
                'open_positions': len(fa_positions),
                'closed_trades': len(fa_trades),
                'cash': fa_cash,
                'equity_history': fa_equity_hist
            },
            'fund_b': {
                'name': 'Moonshot Hunter',
                'band': '10x',
                'equity': fb_equity,
                'pnl': fb_total_pnl,
                'win_rate': round(fb_wins / len(fb_trades), 3) if fb_trades else 0,
                'open_positions': len(fb_positions),
                'closed_trades': len(fb_trades),
                'cash': fb_cash,
                'equity_history': fb_equity_hist
            }
        },
        'opportunities': opps,
        'positions': {
            'fund_a': fa_positions,
            'fund_b': fb_positions
        },
        'realized_trades': {
            'fund_a': fa_trades,
            'fund_b': fb_trades
        },
        'statistics': {
            'by_category': by_category,
            'by_convexity': by_convexity
        }
    }

    path = os.path.join(DATA_DIR, 'signal_data.json')
    with open(path, 'w') as f:
        json.dump(signal_data, f, indent=2, default=str)
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")

    print(f"\nSample data summary:")
    print(f"  Opportunities: {len(opps)}")
    print(f"  Edge scores: {min(o['edge_score'] for o in opps):.1f} - {max(o['edge_score'] for o in opps):.1f}")
    print(f"  Fund A: equity={fa_equity:.2f}, PnL={fa_total_pnl:+.2f}, positions={len(fa_positions)}, trades={len(fa_trades)}")
    print(f"  Fund B: equity={fb_equity:.2f}, PnL={fb_total_pnl:+.2f}, positions={len(fb_positions)}, trades={len(fb_trades)}")
    print(f"  Combined: {combined_equity:.2f} ({combined_pnl:+.2f})")

if __name__ == '__main__':
    main()
