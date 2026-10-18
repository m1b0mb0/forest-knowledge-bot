"""Load and validate configuration from environment variables."""

import os

from aiogram.utils.token import TokenValidationError, validate_token


def load_bot_token() -> str:
    """Return the Telegram token without exposing it in validation errors."""
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("BOT_TOKEN is missing. Set it in your local .env file.")

    try:
        validate_token(token)
    except TokenValidationError:
        raise ValueError(
            "BOT_TOKEN has an invalid format. Check your local .env file."
        ) from None

    return token
