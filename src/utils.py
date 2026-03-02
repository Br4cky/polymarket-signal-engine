"""
Utility module for Polymarket signal engine.
Provides rate limiting, caching, statistical helpers, and text processing.
"""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_per_minute: int):
        """
        Initialize rate limiter.

        Args:
            max_per_minute: Maximum number of requests allowed per minute
        """
        self.max_per_minute = max_per_minute
        self.tokens = float(max_per_minute)
        self.last_update = time.time()

    def wait_if_needed(self) -> None:
        """
        Wait if necessary to maintain rate limit.
        Uses token bucket algorithm with 1-minute window.
        """
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Refill tokens based on elapsed time
        self.tokens += elapsed * (self.max_per_minute / 60.0)
        if self.tokens > self.max_per_minute:
            self.tokens = float(self.max_per_minute)

        # Wait if no tokens available
        if self.tokens < 1.0:
            sleep_time = (1.0 - self.tokens) * (60.0 / self.max_per_minute)
            logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
            self.tokens = 0.0
        else:
            self.tokens -= 1.0


class CacheManager:
    """JSON file-based cache manager."""

    def __init__(self, cache_dir: str):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory path for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "cache.json"
        self._cache: dict = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, key: str, ttl_seconds: int) -> Optional[Any]:
        """
        Get cached data if available and not expired.

        Args:
            key: Cache key
            ttl_seconds: Time-to-live in seconds

        Returns:
            Cached data or None if expired/missing
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        elapsed = time.time() - entry.get("timestamp", 0)

        if elapsed > ttl_seconds:
            del self._cache[key]
            self._save_cache()
            return None

        return entry.get("data")

    def set(self, key: str, data: Any) -> None:
        """
        Store data in cache with current timestamp.

        Args:
            key: Cache key
            data: Data to cache
        """
        self._cache[key] = {
            "data": data,
            "timestamp": time.time()
        }
        self._save_cache()


def mean(values: List[float]) -> float:
    """
    Calculate arithmetic mean of values.

    Args:
        values: List of numeric values

    Returns:
        Mean value, or 0.0 if empty list
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values: List[float]) -> float:
    """
    Calculate standard deviation of values.

    Args:
        values: List of numeric values

    Returns:
        Standard deviation, or 0.0 if fewer than 2 values
    """
    if len(values) < 2:
        return 0.0

    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def z_score(value: float, values: List[float]) -> float:
    """
    Calculate z-score of a value relative to a dataset.

    Args:
        value: Value to score
        values: List of values for reference distribution

    Returns:
        Z-score (standard deviations from mean), or 0.0 if std_dev is 0
    """
    m = mean(values)
    sd = std_dev(values)

    if sd == 0.0:
        return 0.0

    return (value - m) / sd


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert any value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Converted float or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords from market question text.
    Removes common stopwords and returns lowercased, stripped tokens.

    Args:
        text: Market question string

    Returns:
        List of keyword strings
    """
    stopwords = {
        "will", "the", "a", "an", "and", "or", "by", "in", "on", "at",
        "to", "for", "of", "is", "be", "have", "has", "do", "does",
        "did", "would", "could", "should", "can", "may", "might",
        "must", "shall", "if", "than", "as", "when", "where", "why",
        "how", "what", "which", "who", "whom", "whose", "that", "this",
        "these", "those", "with", "from", "up", "out", "about", "into",
        "through", "during", "before", "after", "above", "below", "all",
        "each", "both", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "same", "so", "than", "too", "very",
        "just", "been", "were", "was", "are", "am", "it", "its", "i"
    }

    # Split on whitespace and punctuation
    tokens = []
    current_token = []

    for char in text.lower():
        if char.isalnum():
            current_token.append(char)
        else:
            if current_token:
                tokens.append(''.join(current_token))
                current_token = []

    if current_token:
        tokens.append(''.join(current_token))

    # Filter stopwords and empty strings
    keywords = [
        t.strip() for t in tokens
        if t.strip() and t.strip() not in stopwords and len(t.strip()) > 1
    ]

    return keywords
