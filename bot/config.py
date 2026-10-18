"""Load and validate configuration from environment variables."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from aiogram.utils.token import TokenValidationError, validate_token


@dataclass(frozen=True)
class ArchiveSettings:
    """Public Drive archive configuration"""

    folder_id: str
    api_key: str = field(repr=False)
    data_dir: Path = Path("data")


def load_archive_settings() -> ArchiveSettings:
    """Require public-folder access and a local persistent data directory."""
    folder_id = os.getenv("DRIVE_FOLDER_ID", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id):
        raise ValueError(
            "DRIVE_FOLDER_ID is missing."
        )
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing."
        )
    data_dir = os.getenv("DATA_DIR", "data").strip()
    if not data_dir:
        raise ValueError("DATA_DIR must be a non-empty directory path.")
    return ArchiveSettings(
        folder_id=folder_id,
        api_key=api_key,
        data_dir=Path(data_dir),
    )


def load_bot_token() -> str:
    """Return the Telegram token without exposing it in validation errors."""
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise ValueError(
            "BOT_TOKEN is missing."
        )

    try:
        validate_token(token)
    except TokenValidationError:
        raise ValueError(
            "BOT_TOKEN has an invalid format."
        ) from None

    return token
