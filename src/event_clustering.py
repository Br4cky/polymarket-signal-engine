"""
Event Clustering — Detect conflicting directional bets on the same asset.

The problem isn't concentration per se — if Iran markets are genuinely mispriced,
load up. The problem is CONFLICTING bets:
  - "Bitcoin reach $85k" + "Bitcoin dip to $50k" = opposing directions
  - One wins when the other loses, net is always negative after spread

Same-direction bets are fine:
  - "Iran successor by March 4" + "Iran successor by March 15" = same direction
  - "Bitcoin reach $85k" + "Bitcoin reach $90k" = same direction (correlated)
"""

import re
import logging

logger = logging.getLogger(__name__)

# Month names for stripping date patterns
_MONTHS = r'(?:january|february|march|april|may|june|july|august|september|october|november|december)'

# Patterns to extract the underlying asset/event (strip dates, thresholds, specifics)
_STRIP_PATTERNS = [
    re.compile(rf'-?(?:on|by|in|before|after|during)-?{_MONTHS}-?\d*-?(?:2\d{{3}})?-?', re.IGNORECASE),
    re.compile(r'-20\d{2}(?:-|$)'),
    re.compile(r'-?\$?\d[\d,]*(?:k|m|b)?(?=-|$)', re.IGNORECASE),
    re.compile(r'-?\d+(?:st|nd|rd|th)(?=-|$)', re.IGNORECASE),
    re.compile(r'-?\d+-\d+(?=-|$)'),
    re.compile(r'-\d+$'),
    re.compile(r'-{2,}'),
    re.compile(r'^-+|-+$'),
]

# Words that indicate directional intent
_BULLISH_WORDS = {'reach', 'hit', 'above', 'over', 'exceed', 'break', 'surge', 'rise', 'invade', 'strike', 'launch', 'win', 'fall', 'name'}
_BEARISH_WORDS = {'dip', 'below', 'under', 'drop', 'crash', 'decline', 'lose'}


def _slugify(text: str) -> str:
    """Convert free text to a slug (lowercase, hyphens, no special chars)."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'-{2,}', '-', text)
    return text.strip('-')


def extract_base_event(question: str, slug: str = '') -> str:
    """
    Extract the base event/asset from a market question.

    Strips dates, numeric thresholds, and filler words so that
    date-variant and threshold-variant markets collapse to the same key.
    """
    raw = slug if slug else _slugify(question)
    raw = raw.lower().strip('-')

    for pattern in _STRIP_PATTERNS:
        raw = pattern.sub('', raw)

    # Strip filler words but keep directional words
    raw = re.sub(r'-(?:the|a|an|of|to|at|by|on|in|for|from|and|or|will|be|high|low)(?=-|$)', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'-{2,}', '-', raw)
    raw = raw.strip('-')

    if len(raw) < 3:
        raw = slug if slug else _slugify(question)

    return raw


def extract_direction(question: str) -> str:
    """
    Detect the directional intent of a market question.

    Returns:
        'bullish':  "Bitcoin reach $85k", "Will Iran strike..."
        'bearish':  "Bitcoin dip to $50k", "Will price drop below..."
        'neutral':  "Will there be a ceasefire...", "Iran successor..."
    """
    words = set(question.lower().split())
    # Also check slug-style hyphenated words
    words.update(question.lower().replace('-', ' ').split())

    has_bull = bool(words & _BULLISH_WORDS)
    has_bear = bool(words & _BEARISH_WORDS)

    if has_bull and not has_bear:
        return 'bullish'
    elif has_bear and not has_bull:
        return 'bearish'
    else:
        return 'neutral'


def extract_underlying_asset(question: str) -> str:
    """
    Extract the broad underlying asset/topic from a question.

    "Will Bitcoin reach $85k?" → "bitcoin"
    "Will Bitcoin dip to $50k?" → "bitcoin"
    "Will Crude Oil hit $120?"  → "crude-oil"
    "Will Iran name successor?" → "iran-khamenei" (or similar)

    This is coarser than extract_base_event — it groups ALL Bitcoin
    markets together regardless of direction, so we can detect conflicts.
    """
    q = question.lower()

    # Known asset patterns (order matters — check specific before general)
    asset_patterns = [
        (r'bitcoin|btc', 'bitcoin'),
        (r'ethereum|eth\b', 'ethereum'),
        (r'crude oil|oil.*cl\b', 'crude-oil'),
        (r'gold.*xau|xau.*gold', 'gold'),
        (r's&p 500|spy\b|sp500', 'sp500'),
        (r'nasdaq|qqq\b', 'nasdaq'),
    ]

    for pattern, asset_name in asset_patterns:
        if re.search(pattern, q):
            return asset_name

    # For non-financial assets, fall back to extract_base_event
    return ''


def are_conflicting(question_a: str, question_b: str) -> bool:
    """
    Detect if two market questions represent CONFLICTING bets.

    Conflicting means opposite directions on the same underlying asset:
      - "Bitcoin reach $85k" vs "Bitcoin dip to $50k" → CONFLICTING
      - "Bitcoin reach $85k" vs "Bitcoin reach $90k" → NOT conflicting (same direction)
      - "Iran strike" vs "Iran successor" → NOT conflicting (different events)

    Returns True if the positions would hedge against each other.
    """
    # Must share the same underlying asset
    asset_a = extract_underlying_asset(question_a)
    asset_b = extract_underlying_asset(question_b)

    if not asset_a or not asset_b:
        return False  # Can't determine asset — allow both

    if asset_a != asset_b:
        return False  # Different assets — no conflict

    # Same asset — check directions
    dir_a = extract_direction(question_a)
    dir_b = extract_direction(question_b)

    # Conflict only if one is bullish and the other is bearish
    if {dir_a, dir_b} == {'bullish', 'bearish'}:
        return True

    return False
