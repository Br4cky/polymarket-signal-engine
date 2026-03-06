"""
Telegram Bot — Signal notification scaffold.

Formats signal calls and resolutions into Telegram-ready messages.
Currently just logs what would be sent; flip config telegram.enabled = true
and add bot_token + channel_id to go live.

Setup:
  1. Create a bot via @BotFather on Telegram
  2. Get the bot token
  3. Create a channel and add the bot as admin
  4. Get the channel ID (e.g. @your_channel or -100xxxxx)
  5. Set in config.json: telegram.enabled=true, bot_token, channel_id
"""

import json
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


def format_signal_message(signal: dict) -> str:
    """
    Format a new signal call for Telegram.

    Includes everything a user needs:
    - What to buy and the safe entry range
    - TP and SL levels
    - Edge score and conviction breakdown
    - Link to the market
    """
    layers = signal.get('layer_scores', {})
    edge = signal.get('edge_score', 0)
    tier = signal.get('edge_tier', 'low').upper()

    # Conviction emoji
    tier_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(tier, '⚪')

    min_entry = signal.get('min_entry_price') or signal.get('entry_price', 0)
    max_entry = signal.get('max_entry_price', 0)

    msg = (
        f"{tier_emoji} NEW SIGNAL [{tier}]\n"
        f"\n"
        f"📊 {signal.get('question', '')}\n"
        f"➡️ {signal.get('outcome', '')}\n"
        f"\n"
        f"📍 Entry Range: ${min_entry:.4f} – ${max_entry:.4f}\n"
        f"✅ Take Profit: ${signal.get('tp_price', 0):.4f} ({signal.get('tp_pct', 0):+.0f}%)\n"
        f"🛑 Stop Loss: ${signal.get('sl_price', 0):.4f} ({signal.get('sl_pct', 0):.0f}%)\n"
        f"\n"
        f"Edge: {edge:.0f}/100 "
        f"[S:{layers.get('structural', 0):.0f} "
        f"M:{layers.get('smart_money', 0):.0f} "
        f"D:{layers.get('dislocation', 0):.0f} "
        f"E:{layers.get('external', 0):.0f}]\n"
        f"Band: {signal.get('convexity_band', '?')} "
        f"({signal.get('potential_multiple', 0):.0f}x potential)\n"
        f"\n"
        f"💡 {signal.get('rationale', '')}\n"
        f"\n"
        f"🔗 polymarket.com/event/{signal.get('slug', '')}"
    )
    return msg


def format_resolution_message(signal: dict) -> str:
    """
    Format a signal resolution (TP hit, SL hit, market resolved) for Telegram.
    """
    rt = signal.get('resolution_type', 'unknown')
    pnl = signal.get('hypothetical_pnl_pct') or 0
    entry = signal.get('entry_price') or 0
    final = signal.get('final_price') or 0
    peak = signal.get('peak_price') or 0
    trough = signal.get('trough_price') or 0
    edge = signal.get('edge_score') or 0
    tier = signal.get('edge_tier') or ''

    if rt == 'tp_hit':
        header = f"✅ SIGNAL WIN +{pnl:.1f}%"
    elif rt == 'sl_hit':
        header = f"❌ SIGNAL LOSS {pnl:.1f}%"
    elif rt == 'market_resolved':
        won = pnl > 0
        header = f"{'✅' if won else '❌'} SETTLED {'WIN' if won else 'LOSS'} {pnl:+.1f}%"
    else:
        header = f"📋 SIGNAL CLOSED {pnl:+.1f}%"

    msg = (
        f"{header}\n"
        f"\n"
        f"📊 {signal.get('question', '')[:80]}\n"
        f"Entry: ${entry:.4f} → Final: ${final:.4f}\n"
        f"Peak: ${peak:.4f} | Trough: ${trough:.4f}\n"
        f"\n"
        f"Edge at call: {edge:.0f} [{tier}]"
    )
    return msg


def format_daily_summary(metrics: dict) -> str:
    """Format a daily summary of signal performance."""
    msg = (
        f"📈 Signal Engine Daily Summary\n"
        f"\n"
        f"Active signals: {metrics.get('active_signals', 0)}\n"
        f"Total resolved: {metrics.get('total_resolved', 0)}\n"
        f"Win rate: {metrics.get('win_rate', 0):.0%}\n"
        f"Avg win: {metrics.get('avg_win_pct', 0):+.1f}%\n"
        f"Avg loss: {metrics.get('avg_loss_pct', 0):.1f}%\n"
        f"EV per signal: {metrics.get('ev_per_signal', 0):+.1f}%\n"
    )

    by_tier = metrics.get('by_tier', {})
    for tier_name in ['high', 'medium', 'low']:
        t = by_tier.get(tier_name, {})
        if t.get('total', 0) > 0:
            msg += f"\n{tier_name.upper()}: {t['wins']}/{t['total']} wins ({t['win_rate']:.0%})"

    return msg


def send_message(text: str, config: dict) -> bool:
    """
    Send a message to the configured Telegram channel.

    Returns True if sent successfully, False otherwise.
    Currently logs only — flip telegram.enabled to go live.
    """
    tg_config = config.get('telegram', {})

    if not tg_config.get('enabled', False):
        logger.debug(f"Telegram disabled — would send: {text[:100]}...")
        return False

    bot_token = tg_config.get('bot_token', '')
    channel_id = tg_config.get('channel_id', '')

    if not bot_token or not channel_id:
        logger.warning("Telegram enabled but bot_token or channel_id missing")
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                'chat_id': channel_id,
                'text': text,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True,
            },
            timeout=10
        )
        resp.raise_for_status()
        logger.info(f"Telegram message sent to {channel_id}")
        return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def send_signal(signal: dict, config: dict) -> bool:
    """Format and send a new signal call to Telegram."""
    msg = format_signal_message(signal)
    return send_message(msg, config)


def send_resolution(signal: dict, config: dict) -> bool:
    """Format and send a signal resolution to Telegram."""
    msg = format_resolution_message(signal)
    return send_message(msg, config)
