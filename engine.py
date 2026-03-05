#!/usr/bin/env python3
"""
Polymarket Signal Engine v2 — Main Orchestrator

4-layer mispricing detection with single-pool paper trading.

Usage:
  python engine.py --once              # Single scan, update dashboard
  python engine.py --loop              # Continuous (every 15 min)
  python engine.py --execute --once    # Scan + auto-trade (paper)
  python engine.py --init              # Reset portfolio
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Setup paths so src/ imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.utils import CacheManager, safe_float
from src.scraper import PolymarketClient
from src.kalshi_scraper import KalshiClient
from src.whale_tracker import WhaleTracker
from src.news_signals import NewsSignals
from src.manifold_client import ManifoldClient
from src.scorer import compute_edge_score, rank_opportunities
from src.event_clustering import extract_base_event, are_conflicting
from src.calibration import compute_calibration, suggest_parameter_adjustments
from src.signal_log import SignalLogger
from src.portfolio import (
    load_portfolio, save_portfolio, create_portfolio,
    execute_paper_trade, update_portfolio, auto_close_positions,
    portfolio_summary, is_on_cooldown
)

# ─── Logging ────────────────────────────────────────────────────────────────

log_path = os.path.join(SCRIPT_DIR, 'data', 'engine.log')
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path)
    ]
)
logger = logging.getLogger('engine')


# ─── Config ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(os.path.join(SCRIPT_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
    return config


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(config: dict, execute_trades: bool = False):
    """
    Full signal engine pipeline:
    1. Fetch market data (Polymarket + Kalshi)
    2. Run all 4 signal layers
    3. Score and rank opportunities
    4. Update portfolio positions
    5. (Optional) Execute paper trades
    6. Write dashboard data
    """
    start = time.time()
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Signal Engine v2 — Pipeline starting")
    logger.info("=" * 60)

    # ── Initialize components ──
    cache = CacheManager(data_dir)
    poly_client = PolymarketClient(config.get('polymarket', {}), cache)
    kalshi_client = KalshiClient(config.get('kalshi', {}), cache)
    whale_config = config.get('whale_tracking', {})
    whale_config['data_api_base'] = config.get('polymarket', {}).get(
        'data_api_base', 'https://data-api.polymarket.com'
    )
    whale_tracker = WhaleTracker(whale_config, cache)
    news_signals = NewsSignals(config.get('news', {}), cache)
    manifold_client = ManifoldClient(config.get('manifold', {}), cache)

    signal_logger = SignalLogger(data_dir)

    portfolio_path = os.path.join(data_dir, 'portfolio_state.json')
    portfolio = load_portfolio(portfolio_path, config)

    # ── Step 1: Fetch Polymarket data ──
    logger.info("Step 1: Fetching Polymarket data...")
    markets = poly_client.fetch_enriched_markets(max_markets=300)
    logger.info(f"Got {len(markets)} markets")

    if not markets:
        logger.warning("No markets fetched. Check API connectivity.")
        _write_dashboard([], portfolio, config, data_dir)
        return

    # ── Step 2: Fetch cross-platform data ──
    logger.info("Step 2: Fetching Kalshi data for cross-platform comparison...")
    kalshi_events = []
    if config.get('kalshi', {}).get('enabled', False):
        kalshi_events = kalshi_client.fetch_events()
        logger.info(f"Got {len(kalshi_events)} Kalshi events")

    # ── Step 3: External signals (GDELT + Wikipedia + Fear/Greed + comments) ──
    logger.info("Step 3: External signals enabled (GDELT + Wikipedia + Fear/Greed + comments)")

    # ── Step 3a: Build Manifold cross-reference index ──
    logger.info("Step 3a: Building Manifold Markets cross-reference index...")
    manifold_client.build_manifold_index()
    logger.info(f"Manifold index: {len(manifold_client._manifold_markets)} binary markets")

    # ── Step 3b: Build whale index (one batch of API calls) ──
    logger.info("Step 3b: Building whale position index...")
    whale_tracker.build_whale_index()
    logger.info(f"Whale index covers {len(whale_tracker._whale_market_index)} markets")

    # ── Step 4: Score all market tokens across all 4 layers ──
    logger.info("Step 4: Computing edge scores (all 4 layers)...")
    scored_items = []
    markets_with_history = 0

    # Scan all price levels — repricing opportunities exist at any price
    MIN_PRICE = 0.01
    MAX_PRICE = 0.95
    MAX_HISTORY_FETCHES = 50

    history_fetches = 0
    _external_cache = {}

    for market in markets:
        for token in market.get('tokens', []):
            current_price = safe_float(token.get('current_price', 0))

            if current_price < MIN_PRICE or current_price > MAX_PRICE:
                continue

            # Fetch price history only for candidates with decent volume (capped)
            price_history = []
            if (safe_float(market.get('volume_24h', 0)) > 100
                    and history_fetches < MAX_HISTORY_FETCHES):
                price_history = poly_client.fetch_price_history(token['token_id'])
                history_fetches += 1
                if price_history:
                    markets_with_history += 1

            # Layer 1 partial: cross-platform match
            kalshi_price = None
            if kalshi_events:
                match = kalshi_client.find_matching_markets(
                    market.get('question', ''), kalshi_events
                )
                if match:
                    kalshi_markets = kalshi_client.fetch_markets_for_event(match.get('ticker', ''))
                    if kalshi_markets:
                        kalshi_price = safe_float(kalshi_markets[0].get('yes_bid', 0)) / 100.0

            # Layer 2: smart money
            smart_money = whale_tracker.compute_smart_money_score(
                market_id=market.get('condition_id', market.get('market_id', '')),
                token_id=token.get('token_id', ''),
                market_slug=market.get('slug', ''),
            )

            # Layer 1 extension: Manifold cross-reference
            mkt_question = market.get('question', '')
            if mkt_question not in _external_cache:
                _manifold_cache = _external_cache.get('__manifold__', {})
                if mkt_question not in _manifold_cache:
                    _manifold_cache[mkt_question] = manifold_client.find_matching_probability(mkt_question)
                    _external_cache['__manifold__'] = _manifold_cache

                _external_cache[mkt_question] = news_signals.compute_external_score(
                    market_question=mkt_question,
                    market_category=market.get('category', ''),
                    comment_count=market.get('comment_count', 0),
                    volume_24h=safe_float(market.get('volume_24h', 0)),
                    volume_total=safe_float(market.get('volume_total', 0)),
                )
            external = _external_cache[mkt_question]
            manifold_prob = _external_cache.get('__manifold__', {}).get(mkt_question)

            scores = compute_edge_score(
                token=token,
                market=market,
                price_history=price_history,
                kalshi_price=kalshi_price,
                smart_money=smart_money,
                external=external,
                config=config,
                manifold_prob=manifold_prob,
                created_at=market.get('created_at', market.get('startDate', '')),
            )

            scored_items.append({
                'market': market,
                'token': token,
                'scores': scores
            })

    # Log score distribution
    if scored_items:
        all_edges = [s['scores']['edge_score'] for s in scored_items if 'edge_score' in s.get('scores', {})]
        if all_edges:
            all_edges.sort(reverse=True)
            logger.info(
                f"Scored {len(scored_items)} tokens "
                f"({markets_with_history} with price history) — "
                f"Edge scores: top5={all_edges[:5]}, "
                f"min={min(all_edges):.1f}, max={max(all_edges):.1f}, "
                f"median={all_edges[len(all_edges)//2]:.1f}"
            )
        else:
            logger.info(f"Scored {len(scored_items)} tokens but no edge scores computed")
    else:
        logger.info("No tokens scored")

    # ── Step 5: Rank opportunities ──
    opportunities = rank_opportunities(scored_items, config)
    logger.info(f"Found {len(opportunities)} opportunities above threshold")

    for i, opp in enumerate(opportunities[:5]):
        layers = opp.get('layer_scores', {})
        logger.info(
            f"  #{i+1}: Edge={opp['edge_score']:.1f} "
            f"[S:{layers.get('structural',0):.0f} M:{layers.get('smart_money',0):.0f} "
            f"D:{layers.get('dislocation',0):.0f} E:{layers.get('external',0):.0f}] "
            f"@{opp['current_price']:.2f} {opp['question'][:50]}"
        )

    signal_logger.log_signals(opportunities, top_n=10)

    # ── Step 6: Update existing positions ──
    current_prices = {}
    for item in scored_items:
        tok = item['token']
        current_prices[tok['token_id']] = safe_float(tok['current_price'])

    update_portfolio(portfolio, current_prices)

    # Auto-close positions hitting targets/stops
    fund = portfolio.get('main_pool')
    if fund:
        closed = auto_close_positions(fund, current_prices, config)
        if closed:
            logger.info(f"Auto-closed {len(closed)} positions")
            for trade in closed:
                signal_logger.log_exit(fund['name'], trade)

    # ── Step 7: Execute trades (if enabled) ──
    if execute_trades and config.get('trading', {}).get('auto_trade_enabled', False):
        max_exposure_pct = config['trading'].get('max_exposure_pct', 60) / 100.0
        fund = portfolio.get('main_pool')

        if fund:
            existing_tokens = {p['token_id'] for p in fund['positions']}
            existing_markets = {p['market_id'] for p in fund['positions']}
            trades_done = 0
            cooldown_skips = 0
            cluster_skips = 0

            # Build cluster exposure map (base_event → total $USD deployed)
            cluster_exposure = {}
            for p in fund['positions']:
                base = extract_base_event(p.get('question', ''), p.get('slug', ''))
                cluster_exposure[base] = cluster_exposure.get(base, 0) + p.get('entry_usd', 0)

            max_cluster_pct = config['trading'].get('max_cluster_pct', 12) / 100.0
            max_cluster_usd = fund['capital'] * max_cluster_pct

            for opp in opportunities:
                # Dynamic exposure check
                total_deployed = sum(p['entry_usd'] for p in fund['positions'])
                exposure_pct = total_deployed / fund['capital'] if fund['capital'] > 0 else 1.0
                if exposure_pct >= max_exposure_pct:
                    break

                if opp['token_id'] in existing_tokens:
                    continue

                if opp['market_id'] in existing_markets:
                    continue

                # Stop-out cooldown: don't re-enter markets we were stopped out of
                cooldown_hours = config['trading'].get('stop_cooldown_hours', 24)
                if is_on_cooldown(fund, opp['market_id'], cooldown_hours=cooldown_hours):
                    cooldown_skips += 1
                    continue

                # Cluster cap: don't overload correlated positions
                opp_base = extract_base_event(opp.get('question', ''), opp.get('slug', ''))
                if cluster_exposure.get(opp_base, 0) >= max_cluster_usd:
                    cluster_skips += 1
                    continue

                # Conflict check: don't take opposing directional bets
                opp_question = opp.get('question', '')
                has_conflict = any(
                    are_conflicting(opp_question, p.get('question', ''))
                    for p in fund['positions']
                )
                if has_conflict:
                    continue

                fund, position = execute_paper_trade(fund, opp, config)
                if position:
                    trades_done += 1
                    signal_logger.log_entry(fund['name'], position, opp)
                    # Update cluster exposure map
                    cluster_exposure[opp_base] = cluster_exposure.get(opp_base, 0) + position.get('entry_usd', 0)

            if cooldown_skips:
                logger.info(f"Cooldown filter: skipped {cooldown_skips} re-entry attempts")
            if cluster_skips:
                logger.info(f"Cluster cap: skipped {cluster_skips} correlated opportunities")

            portfolio['main_pool'] = fund
            if trades_done:
                logger.info(
                    f"Executed {trades_done} paper trades "
                    f"(exposure: {sum(p['entry_usd'] for p in fund['positions']):.0f}/"
                    f"{fund['capital']:.0f})"
                )

    # ── Step 8: Calibration analysis ──
    all_realized = portfolio.get('main_pool', {}).get('realized_trades', [])
    calibration = compute_calibration(all_realized)
    if calibration.get('total_trades', 0) >= 20:
        cal_score = calibration.get('calibration_score', 50)
        logger.info(
            f"Calibration: score={cal_score}, "
            f"win_rate={calibration.get('overall_win_rate', 0):.1%}, "
            f"trades={calibration['total_trades']}"
        )
        for rec in calibration.get('recommendations', []):
            logger.info(f"  Recommendation: {rec}")

        adjustments = suggest_parameter_adjustments(calibration, config)
        if adjustments:
            logger.warning(f"Suggested parameter changes: {adjustments}")

    # ── Step 9: Save and write dashboard ──
    save_portfolio(portfolio, portfolio_path)

    layer_health = _compute_layer_health(scored_items, whale_tracker)
    _write_dashboard(opportunities, portfolio, config, data_dir, layer_health, calibration)

    elapsed = time.time() - start
    summary = portfolio_summary(portfolio)
    signal_logger.log_summary(summary)
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s — "
        f"Equity: ${summary['combined_equity']:.2f} "
        f"(PnL: ${summary['combined_pnl']:+.2f})"
    )


def _compute_layer_health(scored_items: list, whale_tracker) -> dict:
    """Compute health stats for each signal layer."""
    layers = {'structural': [], 'smart_money': [], 'dislocation': [], 'external': []}

    for item in scored_items:
        scores = item.get('scores', {})
        ls = scores.get('layer_scores', {})
        for layer in layers:
            val = ls.get(layer, 0)
            layers[layer].append(val)

    health = {}
    for layer, values in layers.items():
        non_zero = [v for v in values if v > 0]
        health[layer] = {
            'active': len(non_zero) > 0,
            'tokens_scored': len(values),
            'tokens_with_signal': len(non_zero),
            'coverage_pct': round(len(non_zero) / max(1, len(values)) * 100, 1),
            'avg_score': round(sum(non_zero) / max(1, len(non_zero)), 2) if non_zero else 0,
            'max_score': round(max(non_zero), 2) if non_zero else 0,
        }

    health['smart_money']['quality_traders'] = len(getattr(whale_tracker, '_quality_traders', []))
    health['smart_money']['markets_indexed'] = len(getattr(whale_tracker, '_whale_market_index', {}))

    return health


def _write_dashboard(opportunities: list, portfolio: dict, config: dict, data_dir: str, layer_health: dict = None, calibration: dict = None):
    """Write signal_data.json for the dashboard."""
    by_category = {}
    by_convexity = {}

    for opp in opportunities:
        cat = opp.get('category', 'other')
        band = opp.get('convexity_band', 'other')

        by_category.setdefault(cat, {'count': 0, 'total_edge': 0})
        by_category[cat]['count'] += 1
        by_category[cat]['total_edge'] += opp['edge_score']

        by_convexity.setdefault(band, {'count': 0, 'total_edge': 0, 'total_liquidity': 0})
        by_convexity[band]['count'] += 1
        by_convexity[band]['total_edge'] += opp['edge_score']
        by_convexity[band]['total_liquidity'] += opp.get('liquidity_usd', 0)

    for v in by_category.values():
        v['avg_edge'] = round(v['total_edge'] / v['count'], 1) if v['count'] > 0 else 0
        del v['total_edge']
    for v in by_convexity.values():
        v['avg_edge'] = round(v['total_edge'] / v['count'], 1) if v['count'] > 0 else 0
        del v['total_edge']

    top_n = config.get('dashboard', {}).get('top_opportunities_count', 25)

    pool = portfolio.get('main_pool', {})

    data = {
        'metadata': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'markets_scanned': len(set(o['market_id'] for o in opportunities)) if opportunities else 0,
            'tokens_scored': len(opportunities),
            'engine_version': '3.0.0'
        },
        'portfolio': portfolio_summary(portfolio),
        'opportunities': opportunities[:top_n],
        'positions': pool.get('positions', []),
        'realized_trades': pool.get('realized_trades', []),
        'statistics': {
            'by_category': by_category,
            'by_convexity': by_convexity
        },
        'layer_health': layer_health or {},
        'calibration': calibration or {}
    }

    path = os.path.join(data_dir, 'signal_data.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Dashboard data written to {path}")


# ─── Entry Point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Polymarket Signal Engine v2')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--loop', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=900, help='Loop interval (seconds)')
    parser.add_argument('--execute', action='store_true', help='Enable paper trade execution')
    parser.add_argument('--init', action='store_true', help='Reset portfolio')
    args = parser.parse_args()

    config = load_config()

    if args.execute:
        config['trading']['auto_trade_enabled'] = True

    if args.init:
        data_dir = os.path.join(SCRIPT_DIR, 'data')
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, 'portfolio_state.json')
        portfolio = create_portfolio(config)
        save_portfolio(portfolio, path)
        logger.info("Portfolio reset to initial state")
        if not args.once and not args.loop:
            return

    if args.loop:
        logger.info(f"Starting continuous mode (interval={args.interval}s)")
        while True:
            try:
                run_pipeline(config, execute_trades=args.execute)
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
            logger.info(f"Sleeping {args.interval}s...")
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
    else:
        try:
            run_pipeline(config, execute_trades=args.execute)
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            sys.exit(1)


if __name__ == '__main__':
    main()
