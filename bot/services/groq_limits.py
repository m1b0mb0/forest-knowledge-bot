"""Interpret Groq's actual quota responses without exposing credentials."""

import math
import re

from groq import RateLimitError


def retry_after(error: RateLimitError) -> float | None:
    """Read Groq's numeric Retry-After without exposing response details."""
    try:
        delay = float(error.response.headers.get("retry-after", ""))
    except (TypeError, ValueError):
        return None
    return delay if math.isfinite(delay) and delay >= 0 else None


def is_daily_limit(error: RateLimitError) -> bool:
    """Distinguish exhausted daily quotas from short minute limits."""
    remaining = error.response.headers.get("x-ratelimit-remaining-requests")
    if remaining is not None and remaining.strip() == "0":
        # Groq's requests header refers to requests per DAY, not per minute.
        return True
    body = error.body
    if not isinstance(body, dict):
        return False
    details = body.get("error", body)
    if isinstance(details, dict):
        message = " ".join(
            value for key in ("message", "type", "code")
            if isinstance(value := details.get(key), str)
        )
    elif isinstance(details, str):
        message = details
    else:
        return False
    return bool(re.search(
        r"\b(?:TPD|RPD)\b|\b(?:tokens|requests)[ _-]+per[ _-]+day\b",
        message,
        flags=re.IGNORECASE,
    ))
