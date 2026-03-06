#!/usr/bin/env python3
"""
Polymarket Signal Engine v4 — Main Orchestrator

4-layer mispricing detection → signal emission + retroactive verification.

Instead of paper trading, the engine emits signals with entry/TP/SL parameters,
then verifies them retroactively using price history on subsequent scans.

Usage:
  python engine.py --once              # Single scan, emit signals, verify active
  python engine.py --loop              # Continuous (every 15 min)
  python engine.py --init              # Reset signals state
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
from src.calibration import compute_calibration
from src.signal_log import SignalLogger
from src.signal_manager import (
    load_signals, save_signals, emit_signal, verify_signals,
    is_signal_on_cooldown, update_signal_stats, signal_metrics
)
from src.telegram import send_signal, send_resolution

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
    logger.info("Signal Engine v4 — Pipeline starting")
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

    # ── Step 1: Fetch Polymarket data ──
    logger.info("Step 1: Fetching Polymarket data...")
    markets = poly_client.fetch_enriched_markets(max_markets=300)
    logger.info(f"Got {len(markets)} markets")

    if not markets:
        logger.warning("No markets fetched. Check API connectivity.")
        _write_dashboard([], config, data_dir)
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
    whale_index_size = len(whale_tracker._whale_market_index)
    logger.info(f"Whale index covers {whale_index_size} markets")

    # Diagnostic: log sample market condition_ids vs whale index keys
    if whale_index_size > 0:
        sample_idx_keys = [k for k in list(whale_tracker._whale_market_index.keys())[:5] if not k.startswith('slug:')]
        sample_mkt_cids = []
        for m in markets[:5]:
            cid = m.get('condition_id', '')
            mid = m.get('market_id', '')
            slug = m.get('slug', '')
            sample_mkt_cids.append(f"cid={cid[:20]}.. mid={mid} slug={slug[:30]}")
        logger.info(f"WHALE_DIAG: Index keys sample: {sample_idx_keys}")
        logger.info(f"WHALE_DIAG: Market lookup IDs: {sample_mkt_cids}")

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

    # ── Step 6: Build current prices map ──
    current_prices = {}
    for item in scored_items:
        tok = item['token']
        current_prices[tok['token_id']] = safe_float(tok['current_price'])

    # ── Step 7: Verify active signals against price history ──
    signals_path = os.path.join(data_dir, 'signals_state.json')
    signals_state = load_signals(signals_path)

    logger.info(f"Step 7: Verifying {len(signals_state.get('active', []))} active signals...")
    newly_resolved, still_active = verify_signals(
        signals_state,
        fetch_price_history=poly_client.fetch_price_history,
        current_prices=current_prices,
    )

    if newly_resolved:
        signals_state['resolved'].extend(newly_resolved)
        # Keep last 200 resolved signals
        signals_state['resolved'] = signals_state['resolved'][-200:]
        update_signal_stats(signals_state, newly_resolved)
        wins = sum(1 for s in newly_resolved if s.get('resolution_type') == 'tp_hit')
        losses = sum(1 for s in newly_resolved if s.get('resolution_type') == 'sl_hit')
        expired = sum(1 for s in newly_resolved if s.get('resolution_type') == 'expired')
        logger.info(f"Signal verification: {len(newly_resolved)} resolved ({wins} wins, {losses} losses, {expired} expired)")

        # Send Telegram notifications for resolved signals
        for resolved_sig in newly_resolved:
            send_resolution(resolved_sig, config)

    signals_state['active'] = still_active

    # ── Step 8: Emit new signals ──
    signals_cfg = config.get('signals', {})
    if signals_cfg.get('enabled', True):
        max_active = signals_cfg.get('max_active_signals', 30)
        cooldown_hours = signals_cfg.get('cooldown_after_loss_hours', 24)
        max_per_cluster = signals_cfg.get('max_signals_per_cluster', 4)

        active_tokens = {s['token_id'] for s in signals_state['active']}
        active_markets = {s['market_id'] for s in signals_state['active']}

        # Build cluster count map (base_event → active signal count)
        cluster_counts = {}
        for s in signals_state['active']:
            base = extract_base_event(s.get('question', ''), s.get('slug', ''))
            cluster_counts[base] = cluster_counts.get(base, 0) + 1

        signals_emitted = 0
        cooldown_skips = 0
        cluster_skips = 0
        conflict_skips = 0

        for opp in opportunities:
            # Max active signals cap
            if len(signals_state['active']) >= max_active:
                break

            # Already have signal for this token or market
            if opp['token_id'] in active_tokens:
                continue
            if opp['market_id'] in active_markets:
                continue

            # Cooldown: don't re-signal markets where we recently got stopped out
            if is_signal_on_cooldown(signals_state, opp['market_id'], cooldown_hours):
                cooldown_skips += 1
                continue

            # Cluster cap: don't flood with correlated signals
            opp_base = extract_base_event(opp.get('question', ''), opp.get('slug', ''))
            if cluster_counts.get(opp_base, 0) >= max_per_cluster:
                cluster_skips += 1
                continue

            # Conflict check: don't signal opposing directions
            all_active_questions = [s.get('question', '') for s in signals_state['active']]
            opp_question = opp.get('question', '')
            has_conflict = any(are_conflicting(opp_question, q) for q in all_active_questions)
            if has_conflict:
                conflict_skips += 1
                continue

            # Emit the signal
            signal = emit_signal(opp, config)
            signals_state['active'].append(signal)
            signals_state['stats']['total_signals'] = signals_state['stats'].get('total_signals', 0) + 1
            active_tokens.add(opp['token_id'])
            active_markets.add(opp['market_id'])
            cluster_counts[opp_base] = cluster_counts.get(opp_base, 0) + 1
            signals_emitted += 1

            signal_logger.log_signal_call(signal)
            send_signal(signal, config)

        if cooldown_skips:
            logger.info(f"Signal cooldown: skipped {cooldown_skips} re-signal attempts")
        if cluster_skips:
            logger.info(f"Signal cluster cap: skipped {cluster_skips} correlated signals")
        if conflict_skips:
            logger.info(f"Signal conflict: skipped {conflict_skips} opposing directions")
        if signals_emitted:
            logger.info(f"Emitted {signals_emitted} new signals (total active: {len(signals_state['active'])})")
    else:
        logger.info("Signal emission disabled in config")

    # Save signals state
    save_signals(signals_state, signals_path)

    # ── Step 8: Calibration from resolved signals ──
    resolved_signals = signals_state.get('resolved', [])
    calibration = compute_calibration(resolved_signals) if len(resolved_signals) >= 10 else {}
    if calibration.get('total_trades', 0) >= 20:
        cal_score = calibration.get('calibration_score', 50)
        logger.info(
            f"Calibration: score={cal_score}, "
            f"win_rate={calibration.get('overall_win_rate', 0):.1%}, "
            f"signals={calibration['total_trades']}"
        )

    # ── Step 9: Write dashboard ──
    layer_health = _compute_layer_health(scored_items, whale_tracker)
    sig_metrics = signal_metrics(signals_state)
    _write_dashboard(opportunities, config, data_dir, layer_health, calibration,
                     signals_state=signals_state, sig_metrics=sig_metrics,
                     markets_scanned=len(markets), tokens_scored=len(scored_items))

    elapsed = time.time() - start
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s — "
        f"Signals: {sig_metrics['active_signals']} active, "
        f"{sig_metrics['total_resolved']} resolved "
        f"(win rate: {sig_metrics['win_rate']:.0%}, "
        f"EV: {sig_metrics['ev_per_signal']:+.1f}%/signal)"
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


def _write_dashboard(opportunities: list, config: dict, data_dir: str,
                     layer_health: dict = None, calibration: dict = None,
                     signals_state: dict = None, sig_metrics: dict = None,
                     markets_scanned: int = 0, tokens_scored: int = 0):
    """Write signal_data.json for the dashboard."""
    data = {
        'metadata': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'markets_scanned': markets_scanned,
            'tokens_scored': tokens_scored,
            'opportunities_found': len(opportunities),
            'engine_version': '4.0.0'
        },
        'signals': {
            'active': (signals_state or {}).get('active', []),
            'recent_resolved': (signals_state or {}).get('resolved', [])[-50:],
            'metrics': sig_metrics or {},
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
    parser = argparse.ArgumentParser(description='Polymarket Signal Engine v4')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--loop', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=900, help='Loop interval (seconds)')
    parser.add_argument('--execute', action='store_true', help='(Ignored, kept for GH Actions compat)')
    parser.add_argument('--init', action='store_true', help='Reset signals state')
    args = parser.parse_args()

    config = load_config()

    if args.init:
        data_dir = os.path.join(SCRIPT_DIR, 'data')
        os.makedirs(data_dir, exist_ok=True)
        from src.signal_manager import _create_empty_state, save_signals as _save_signals
        signals_path = os.path.join(data_dir, 'signals_state.json')
        _save_signals(_create_empty_state(), signals_path)
        logger.info("Signals state reset")
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
