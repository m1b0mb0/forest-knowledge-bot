"""Check provider limit metadata without making Groq requests."""

from typing import Any

import httpx
import pytest
from groq import RateLimitError

from bot.services import groq_limits


def rate_limit(
    headers: dict[str, str] | None = None, body: Any = None,
) -> RateLimitError:
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://example.invalid/chat"),
    )
    return RateLimitError("Provider response", response=response, body=body)


@pytest.mark.parametrize("header, expected", [
    ("0", 0.0), ("2", 2.0), ("1.75", 1.75), (" 3 ", 3.0),
    (None, None), ("", None), ("tomorrow", None), ("-1", None),
    ("NaN", None), ("inf", None), ("-Infinity", None),
])
def test_retry_after_accepts_only_finite_nonnegative_seconds(
    header: str | None, expected: float | None,
) -> None:
    headers = {} if header is None else {"retry-after": header}

    assert groq_limits.retry_after(rate_limit(headers)) == expected


def test_daily_requests_header_is_distinct_from_minute_tokens() -> None:
    assert groq_limits.is_daily_limit(rate_limit({
        "x-ratelimit-remaining-requests": "0",
    }))
    assert not groq_limits.is_daily_limit(rate_limit({
        "x-ratelimit-remaining-requests": "15",
        "x-ratelimit-remaining-tokens": "0",
    }))


@pytest.mark.parametrize("body", [
    {"error": {"message": "Limit reached on tokens per day (TPD)."}},
    {"message": "Limit reached on requests per day."},
    {"error": {"code": "TPD"}},
    {"error": {"type": "requests_per_day"}},
    {"error": "Rate limit reached: RPD."},
])
def test_daily_limit_is_recognized_in_structured_error_body(body: Any) -> None:
    assert groq_limits.is_daily_limit(rate_limit(body=body))


@pytest.mark.parametrize("body", [
    None, "unstructured TPD", {}, {"error": None},
    {"error": {"message": "Limit reached on tokens per minute (TPM)."}},
    {"error": {"code": "rate_limit_exceeded"}},
])
def test_unknown_or_minute_limit_is_not_claimed_as_daily(body: Any) -> None:
    assert not groq_limits.is_daily_limit(rate_limit(body=body))
