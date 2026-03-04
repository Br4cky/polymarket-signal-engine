"""
Signal Log — Append-only JSONL log of every signal and trade event.

Designed as the data layer for a future Telegram/Discord notification bot.
Each line is a self-contained JSON object with everything needed to generate
a human-readable alert message.

Event types:
  - signal:   New opportunity detected above threshold
  - entry:    Paper trade executed (position opened)
  - exit:     Position closed (profit target, stop loss, trailing stop)
  - summary:  End-of-run portfolio snapshot

File: data/signal_log.jsonl (append-only, one JSON object per line)
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SignalLogger:
    """Append-only signal log for bot-ready event history."""

    def __init__(self, data_dir: str):
        self.log_path = os.path.join(data_dir, 'signal_log.jsonl')
        os.makedirs(data_dir, exist_ok=True)

    def _append(self, event: dict):
        """Append a single event to the JSONL log."""
        event['logged_at'] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(event, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to write signal log: {e}")

    def log_signals(self, opportunities: List[dict], top_n: int = 10):
        """
        Log top opportunities from this run.

        Each signal contains everything a notification bot would need:
        question, price, band, edge score, layer breakdown, and a
        pre-formatted message string.
        """
        for opp in opportunities[:top_n]:
            layers = opp.get('layer_scores', {})
            price = opp.get('current_price', 0)
            multiple = opp.get('potential_multiple', 0)

            # Pre-format the bot message so it's ready to send
            message = (
                f"🎯 Signal: {opp.get('convexity_band', '?')} opportunity\n"
                f"📊 {opp.get('question', '')[:100]}\n"
                f"💰 {opp.get('outcome', '')} @ {price:.2f} → {multiple:.1f}x potential\n"
                f"⚡ Edge: {opp.get('edge_score', 0):.1f} "
                f"[S:{layers.get('structural', 0):.0f} "
                f"W:{layers.get('smart_money', 0):.0f} "
                f"D:{layers.get('dislocation', 0):.0f} "
                f"E:{layers.get('external', 0):.0f}]\n"
                f"🔗 https://polymarket.com/event/{opp.get('slug', '')}"
            )

            self._append({
                'type': 'signal',
                'market_id': opp.get('market_id', ''),
                'token_id': opp.get('token_id', ''),
                'question': opp.get('question', ''),
                'outcome': opp.get('outcome', ''),
                'slug': opp.get('slug', ''),
                'current_price': price,
                'convexity_band': opp.get('convexity_band', ''),
                'potential_multiple': multiple,
                'edge_score': opp.get('edge_score', 0),
                'layer_scores': layers,
                'volume_24h': opp.get('volume_24h', 0),
                'category': opp.get('category', ''),
                'resolution_date': opp.get('resolution_date', ''),
                'message': message,
            })

    def log_entry(self, fund_name: str, position: dict, opportunity: dict):
        """Log a new paper trade entry."""
        layers = opportunity.get('layer_scores', {})

        message = (
            f"📥 Entry: [{fund_name}]\n"
            f"📊 {opportunity.get('question', '')[:100]}\n"
            f"💰 BUY {position.get('outcome', '')} "
            f"${position.get('entry_usd', 0):.2f} @ {position.get('entry_price', 0):.4f}\n"
            f"🎯 Edge: {position.get('edge_score_at_entry', 0):.1f} "
            f"[{position.get('edge_tier_at_entry', 'low')}]"
        )

        self._append({
            'type': 'entry',
            'fund': fund_name,
            'position_id': position.get('position_id', ''),
            'market_id': position.get('market_id', ''),
            'question': position.get('question', ''),
            'outcome': position.get('outcome', ''),
            'slug': position.get('slug', ''),
            'entry_price': position.get('entry_price', 0),
            'entry_usd': position.get('entry_usd', 0),
            'shares': position.get('shares', 0),
            'convexity_band': position.get('convexity_band', ''),
            'edge_score': position.get('edge_score_at_entry', 0),
            'layer_scores': position.get('layer_scores_at_entry', {}),
            'message': message,
        })

    def log_exit(self, fund_name: str, trade: dict):
        """Log a position close (stop loss, profit target, trailing stop, etc.)."""
        pnl = trade.get('pnl_usd', 0)
        pnl_pct = trade.get('pnl_pct', 0)
        won = '✅' if trade.get('win', False) else '❌'

        message = (
            f"{won} Exit: [{fund_name}]\n"
            f"📊 {trade.get('question', '')[:100]}\n"
            f"💰 {trade.get('outcome', '')} "
            f"${trade.get('entry_usd', 0):.2f} → ${trade.get('exit_usd', 0):.2f} "
            f"({pnl_pct:+.1f}%)\n"
            f"📝 Reason: {trade.get('reason', 'unknown')}"
        )

        self._append({
            'type': 'exit',
            'fund': fund_name,
            'position_id': trade.get('position_id', ''),
            'market_id': trade.get('market_id', ''),
            'question': trade.get('question', ''),
            'outcome': trade.get('outcome', ''),
            'entry_price': trade.get('entry_price', 0),
            'exit_price': trade.get('exit_price', 0),
            'entry_usd': trade.get('entry_usd', 0),
            'exit_usd': trade.get('exit_usd', 0),
            'pnl_usd': pnl,
            'pnl_pct': pnl_pct,
            'win': trade.get('win', False),
            'reason': trade.get('reason', ''),
            'convexity_band': trade.get('convexity_band', ''),
            'edge_score_at_entry': trade.get('edge_score_at_entry', 0),
            'message': message,
        })

    def log_summary(self, portfolio_sum: dict):
        """Log end-of-run portfolio snapshot."""
        pool = portfolio_sum.get('pool', {})

        message = (
            f"📈 Portfolio Update\n"
            f"Equity: ${portfolio_sum.get('combined_equity', 0):.2f} "
            f"({portfolio_sum.get('combined_pnl_pct', 0):+.1f}%)\n"
            f"Open: {portfolio_sum.get('open_positions', 0)} | "
            f"Win rate: {portfolio_sum.get('combined_win_rate', 0):.0%}"
        )

        self._append({
            'type': 'summary',
            'combined_equity': portfolio_sum.get('combined_equity', 0),
            'combined_pnl': portfolio_sum.get('combined_pnl', 0),
            'combined_pnl_pct': portfolio_sum.get('combined_pnl_pct', 0),
            'combined_win_rate': portfolio_sum.get('combined_win_rate', 0),
            'open_positions': portfolio_sum.get('open_positions', 0),
            'closed_trades': portfolio_sum.get('combined_total_trades', 0),
            'message': message,
        })
