"""
Calibration Module — Tracks prediction accuracy and adjusts parameters.

The core problem: our Kelly formula maps edge_score → estimated_win_prob,
but we never check if that mapping is accurate. If 70-score trades only
win 35% of the time (vs the 58% we assume), we're systematically overbetting.

This module:
  1. Groups realized trades by edge score quintile
  2. Compares actual win rates to estimated win rates
  3. Detects calibration drift
  4. Recommends parameter adjustments (kelly_fraction, edge_threshold)
  5. Tracks per-layer signal quality (which layers actually predict wins?)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_calibration(realized_trades: List[dict]) -> dict:
    """
    Analyse realized trades to check if our scoring is well-calibrated.

    Groups trades into edge score bands and compares actual win rates
    to the estimated win probability the Kelly formula used.

    Returns: {
        total_trades: int,
        overall_win_rate: float,
        quintiles: [{band, trades, actual_win_rate, estimated_win_rate, drift}],
        calibration_score: float (0-100, higher=better calibrated),
        recommendations: [str],
        layer_accuracy: {layer_name: {trades, avg_score_winners, avg_score_losers, predictive_power}}
    }
    """
    if not realized_trades:
        return _empty_calibration()

    # Filter to trades with edge score data
    trades_with_scores = [
        t for t in realized_trades
        if t.get('edge_score_at_entry', 0) > 0
    ]

    if len(trades_with_scores) < 5:
        return _empty_calibration(len(realized_trades))

    # ── Overall stats ──
    total = len(trades_with_scores)
    wins = sum(1 for t in trades_with_scores if t.get('win', False))
    overall_wr = wins / total

    # ── Edge score quintile analysis ──
    quintiles = _compute_quintile_accuracy(trades_with_scores)

    # ── Per-layer signal quality ──
    layer_accuracy = _compute_layer_accuracy(trades_with_scores)

    # ── Calibration score (0-100) ──
    # Measures how well edge scores predict actual outcomes
    # Perfect calibration = 100, random = ~50, inverse = 0
    calibration_score = _compute_calibration_score(quintiles)

    # ── Recommendations ──
    recommendations = _generate_recommendations(
        quintiles, layer_accuracy, overall_wr, calibration_score, total
    )

    return {
        'total_trades': total,
        'overall_win_rate': round(overall_wr, 3),
        'overall_avg_pnl_pct': round(
            sum(t.get('pnl_pct', 0) for t in trades_with_scores) / total, 1
        ),
        'quintiles': quintiles,
        'calibration_score': calibration_score,
        'layer_accuracy': layer_accuracy,
        'recommendations': recommendations,
    }


def _empty_calibration(total: int = 0) -> dict:
    """Return empty calibration when insufficient data."""
    return {
        'total_trades': total,
        'overall_win_rate': 0,
        'overall_avg_pnl_pct': 0,
        'quintiles': [],
        'calibration_score': 50,  # Neutral — no data
        'layer_accuracy': {},
        'recommendations': ['Need 5+ realized trades with edge scores for calibration'],
    }


def _compute_quintile_accuracy(trades: List[dict]) -> List[dict]:
    """Group trades by edge score range and compute accuracy per group."""
    # Use bands: 15-25, 25-35, 35-50, 50-65, 65+
    bands = [
        (15, 25, 'low'),
        (25, 35, 'below_avg'),
        (35, 50, 'average'),
        (50, 65, 'above_avg'),
        (65, 101, 'high'),
    ]

    quintiles = []
    for low, high, label in bands:
        group = [t for t in trades if low <= t.get('edge_score_at_entry', 0) < high]
        if not group:
            continue

        wins = sum(1 for t in group if t.get('win', False))
        actual_wr = wins / len(group)

        # What did our Kelly formula estimate for this range?
        mid_edge = (low + high) / 2
        estimated_wr = 0.50 + (mid_edge - 50) / 100.0 * 0.40

        drift = actual_wr - estimated_wr
        avg_pnl = sum(t.get('pnl_pct', 0) for t in group) / len(group)

        quintiles.append({
            'band': f'{low}-{high}',
            'label': label,
            'trades': len(group),
            'wins': wins,
            'actual_win_rate': round(actual_wr, 3),
            'estimated_win_rate': round(estimated_wr, 3),
            'drift': round(drift, 3),
            'avg_pnl_pct': round(avg_pnl, 1),
        })

    return quintiles


def _compute_layer_accuracy(trades: List[dict]) -> dict:
    """
    Analyse which signal layers actually predict winning trades.

    For each layer, compare average score on winners vs losers.
    A good predictive layer has significantly higher scores on winners.
    """
    layers = ['structural', 'smart_money', 'dislocation', 'external']
    result = {}

    for layer in layers:
        winners = [
            t.get('layer_scores_at_entry', {}).get(layer, 0)
            for t in trades if t.get('win', False)
            and t.get('layer_scores_at_entry', {}).get(layer, 0) > 0
        ]
        losers = [
            t.get('layer_scores_at_entry', {}).get(layer, 0)
            for t in trades if not t.get('win', False)
            and t.get('layer_scores_at_entry', {}).get(layer, 0) > 0
        ]

        avg_win = sum(winners) / max(1, len(winners)) if winners else 0
        avg_loss = sum(losers) / max(1, len(losers)) if losers else 0

        # Predictive power: how much higher are winner scores?
        # > 1.3 = good predictor, < 1.0 = inverse predictor (bad)
        if avg_loss > 0:
            predictive_power = avg_win / avg_loss
        elif avg_win > 0:
            predictive_power = 2.0  # Only winners have signal = good
        else:
            predictive_power = 1.0  # No data

        result[layer] = {
            'trades_with_signal': len(winners) + len(losers),
            'avg_score_winners': round(avg_win, 2),
            'avg_score_losers': round(avg_loss, 2),
            'predictive_power': round(predictive_power, 2),
        }

    return result


def _compute_calibration_score(quintiles: List[dict]) -> float:
    """
    Score overall calibration quality (0-100).

    Perfect: higher edge quintiles have monotonically higher win rates.
    Good: general upward trend with some noise.
    Bad: no correlation or inverse correlation.
    """
    if len(quintiles) < 2:
        return 50.0  # Insufficient data

    # Check if win rates increase with edge score
    win_rates = [q['actual_win_rate'] for q in quintiles]

    # Count concordant pairs (higher edge → higher win rate)
    concordant = 0
    discordant = 0
    for i in range(len(win_rates)):
        for j in range(i + 1, len(win_rates)):
            if win_rates[j] > win_rates[i]:
                concordant += 1
            elif win_rates[j] < win_rates[i]:
                discordant += 1

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 50.0

    # Kendall's tau scaled to 0-100
    tau = (concordant - discordant) / total_pairs
    calibration = round((tau + 1) / 2 * 100, 1)

    # Also penalise large drift in any quintile
    max_drift = max(abs(q['drift']) for q in quintiles) if quintiles else 0
    if max_drift > 0.30:
        calibration *= 0.7  # Severe miscalibration
    elif max_drift > 0.20:
        calibration *= 0.85

    return round(min(100, max(0, calibration)), 1)


def _generate_recommendations(
    quintiles: List[dict],
    layer_accuracy: dict,
    overall_wr: float,
    calibration_score: float,
    total_trades: int
) -> List[str]:
    """Generate actionable parameter recommendations."""
    recs = []

    if total_trades < 20:
        recs.append(f'Only {total_trades} trades — need 20+ for reliable calibration')
        return recs

    # Check overall win rate vs expectation
    if overall_wr < 0.25:
        recs.append('Win rate critically low (<25%) — consider raising edge_threshold by 5-10 points')
    elif overall_wr < 0.35:
        recs.append('Win rate below target (<35%) — consider raising edge_threshold by 3-5 points')

    # Check calibration score
    if calibration_score < 40:
        recs.append('Edge scores poorly predict outcomes — scoring model needs rethinking')
    elif calibration_score < 60:
        recs.append('Edge scores weakly predict outcomes — review layer weights')

    # Check per-quintile drift
    for q in quintiles:
        if q['drift'] < -0.20 and q['trades'] >= 5:
            recs.append(
                f"Edge {q['band']}: estimated {q['estimated_win_rate']:.0%} win rate "
                f"but actual {q['actual_win_rate']:.0%} — overestimating this range"
            )

    # Check layer quality
    for layer, stats in layer_accuracy.items():
        if stats['trades_with_signal'] >= 10:
            if stats['predictive_power'] < 0.8:
                recs.append(
                    f"Layer '{layer}' is counter-predictive "
                    f"(power={stats['predictive_power']:.1f}) — consider reducing its weight"
                )
            elif stats['predictive_power'] > 1.5:
                recs.append(
                    f"Layer '{layer}' is strongly predictive "
                    f"(power={stats['predictive_power']:.1f}) — consider increasing its weight"
                )

    if not recs:
        recs.append('Calibration looks healthy — no adjustments needed')

    return recs


def suggest_parameter_adjustments(calibration: dict, current_config: dict) -> dict:
    """
    Based on calibration analysis, suggest specific config changes.

    Returns dict of suggested parameter changes (empty if no changes needed).
    """
    suggestions = {}
    cal_score = calibration.get('calibration_score', 50)
    overall_wr = calibration.get('overall_win_rate', 0)
    total = calibration.get('total_trades', 0)

    if total < 20:
        return suggestions  # Not enough data

    # Kelly fraction adjustment
    if overall_wr < 0.30:
        current_kelly = current_config.get('portfolio', {}).get('kelly_fraction', 0.5)
        suggestions['kelly_fraction'] = round(current_kelly * 0.75, 2)

    # Edge threshold adjustment
    quintiles = calibration.get('quintiles', [])
    for q in quintiles:
        if q.get('label') == 'low' and q.get('actual_win_rate', 0) < 0.20 and q.get('trades', 0) >= 5:
            current_threshold = current_config.get('scoring', {}).get('edge_threshold', 15)
            suggestions['edge_threshold'] = current_threshold + 5

    # Layer weight adjustments
    layer_acc = calibration.get('layer_accuracy', {})
    weight_changes = {}
    for layer, stats in layer_acc.items():
        if stats.get('trades_with_signal', 0) >= 10:
            power = stats.get('predictive_power', 1.0)
            if power < 0.7:
                weight_changes[layer] = 'reduce'
            elif power > 1.8:
                weight_changes[layer] = 'increase'

    if weight_changes:
        suggestions['layer_weight_changes'] = weight_changes

    return suggestions
