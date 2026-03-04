"""
Event Clustering — Group related markets into base events.

Prevents concentration risk by identifying that "Iran successor by March 3/4/5/6"
are all the same underlying bet, and "Bitcoin reach $85k/$90k in March" are correlated.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Month names for stripping date patterns
_MONTHS = r'(?:january|february|march|april|may|june|july|august|september|october|november|december)'

# Compiled patterns for performance (applied in order)
_STRIP_PATTERNS = [
    # Dates: "on March 6, 2026", "by March 31", "in March", "March 2026"
    re.compile(rf'-?(?:on|by|in|before|after|during)-?{_MONTHS}-?\d*-?(?:2\d{{3}})?-?', re.IGNORECASE),
    # Standalone year: "-2026", "-2025"
    re.compile(r'-20\d{2}(?:-|$)'),
    # Dollar amounts: "$85,000", "$120", "85000"
    re.compile(r'-?\$?\d[\d,]*(?:k|m|b)?(?=-|$)', re.IGNORECASE),
    # Ordinal numbers: "98th", "1st"
    re.compile(r'-?\d+(?:st|nd|rd|th)(?=-|$)', re.IGNORECASE),
    # Time ranges: "140-159", "200-219" (tweet count ranges etc)
    re.compile(r'-?\d+-\d+(?=-|$)'),
    # Standalone trailing numbers: strip any remaining digits at end of slug
    re.compile(r'-\d+$'),
    # Common filler words that create noise between variants
    re.compile(r'-(?:the|a|an|of|to|at|by|on|in|for|from|and|or|will|be|hit|dip|reach|high|low)(?=-|$)', re.IGNORECASE),
    # Clean up multiple consecutive hyphens
    re.compile(r'-{2,}'),
    # Trailing/leading hyphens
    re.compile(r'^-+|-+$'),
]


def _slugify(text: str) -> str:
    """Convert free text to a slug (lowercase, hyphens, no special chars)."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'-{2,}', '-', text)
    return text.strip('-')


def extract_base_event(question: str, slug: str = '') -> str:
    """
    Extract the base event identifier from a market question/slug.

    Strips dates, numeric thresholds, and filler words so that date-variant
    and threshold-variant markets collapse to the same key.

    Examples:
        "Will Iran name a successor to Khamenei by March 3?"  → "iran-name-successor-khamenei"
        "Will Iran name a successor to Khamenei by March 31?" → "iran-name-successor-khamenei"
        "Will Bitcoin reach $85,000 in March?"                 → "bitcoin-march"
        "Will Bitcoin reach $90,000 in March?"                 → "bitcoin-march"
        "Will US or Israel strike Iran on March 6, 2026?"      → "us-israel-strike-iran"

    Args:
        question: Market question text
        slug: URL slug (preferred if available, cleaner than question)

    Returns:
        Normalised base event string
    """
    # Prefer slug (cleaner), fall back to slugifying question
    raw = slug if slug else _slugify(question)
    raw = raw.lower().strip('-')

    # Apply stripping patterns in order
    for pattern in _STRIP_PATTERNS:
        raw = pattern.sub('', raw)

    # Final cleanup
    raw = re.sub(r'-{2,}', '-', raw)
    raw = raw.strip('-')

    # If we stripped too aggressively and nothing is left, use original
    if len(raw) < 3:
        raw = slug if slug else _slugify(question)

    return raw
