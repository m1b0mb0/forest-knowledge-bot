"""Shared, opt-in setup for offline tests."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from bot.services import llm
from bot.storage import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    instance = Store(tmp_path)
    instance.initialize()
    return instance


@pytest.fixture
def offline_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid vocabulary downloads only in tests that request this fixture."""
    monkeypatch.setattr(llm, "_count_tokens", lambda text: len(text.split()))


@pytest.fixture
def typing(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace background Telegram chat actions with an offline context."""
    context = MagicMock()
    context.__aenter__ = AsyncMock()
    context.__aexit__ = AsyncMock(return_value=False)
    sender = Mock(return_value=context)
    monkeypatch.setattr("bot.handlers.ChatActionSender.typing", sender)
    return sender
