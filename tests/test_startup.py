"""Startup metadata and backlog handling without Telegram or Groq calls."""

import asyncio
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.methods import DeleteWebhook
from aiogram.types import BotCommandScopeDefault, MenuButtonCommands

import bot.__main__ as startup
from bot.storage import Store


@dataclass
class StartupHarness:
    """Record startup boundaries while replacing all external resources."""

    events: list[str]
    bot: MagicMock
    dispatcher: Mock


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> StartupHarness:
    events: list[str] = []
    bot = MagicMock(spec=Bot)
    bot.__aenter__.return_value = bot
    bot.__aexit__.side_effect = lambda *args: events.append("bot_closed")
    for method, event in (
        ("set_my_commands", "commands"),
        ("set_my_description", "description"),
        ("set_chat_menu_button", "menu"),
        ("delete_webhook", "drop"),
    ):
        getattr(bot, method).side_effect = (
            lambda *args, name=event, **kwargs: events.append(name)
        )

    client = MagicMock()
    client.__aenter__.return_value = client
    client.__aexit__.side_effect = lambda *args: events.append("client_closed")
    pool = MagicMock()
    pool.__enter__.return_value = pool
    pool.__exit__.side_effect = lambda *args: events.append("pool_closed")
    dispatcher = Mock()
    dispatcher.start_polling = AsyncMock(
        side_effect=lambda *args, **kwargs: events.append("poll")
    )
    store = Mock(spec=Store)
    store.initialize.side_effect = lambda: events.append("initialize")

    async def import_archive(*args: object) -> int:
        events.append("import")
        return 1

    monkeypatch.setattr(startup, "load_bot_token", lambda: "test-token")
    monkeypatch.setattr(
        startup,
        "load_archive_settings",
        lambda: SimpleNamespace(data_dir=Path("unused-data")),
    )
    monkeypatch.setattr(
        startup,
        "load_llm_settings",
        lambda: SimpleNamespace(api_key="test-key", model="test-model"),
    )
    monkeypatch.setattr(startup, "Store", Mock(return_value=store))
    monkeypatch.setattr(startup, "import_archive", import_archive)
    monkeypatch.setattr(
        startup.asyncio, "to_thread", AsyncMock(side_effect=lambda fn: fn())
    )
    monkeypatch.setattr(startup, "Dispatcher", Mock(return_value=dispatcher))
    monkeypatch.setattr(startup, "get_context", Mock())
    monkeypatch.setattr(
        startup, "ProcessPoolExecutor", Mock(return_value=pool)
    )
    monkeypatch.setattr(startup, "AsyncGroq", Mock(return_value=client))
    monkeypatch.setattr(startup, "Bot", Mock(return_value=bot))
    monkeypatch.setattr(startup, "ArchiveAssistant", Mock())
    monkeypatch.setattr(startup, "UploadService", Mock())
    return StartupHarness(events, bot, dispatcher)


def test_configuration_exposes_existing_commands_and_empty_chat_description(
    harness: StartupHarness,
) -> None:
    asyncio.run(startup.configure_bot(harness.bot))

    harness.bot.set_my_commands.assert_awaited_once()
    command_options = harness.bot.set_my_commands.await_args.kwargs
    assert isinstance(command_options["scope"], BotCommandScopeDefault)
    commands = command_options["commands"]
    assert [item.command for item in commands] == [
        "start", "my_documents", "delete"
    ]
    for command in commands:
        assert 1 <= len(command.description) <= 256
        assert any("а" <= char <= "я" for char in command.description.lower())

    harness.bot.set_my_description.assert_awaited_once()
    description_options = harness.bot.set_my_description.await_args.kwargs
    description = description_options["description"]
    assert 1 <= len(description) <= 512
    assert "Лісовому архіві" in description
    assert "/start" in description
    assert all(
        extension in description for extension in ("PDF", "DOCX", "TXT")
    )

    harness.bot.set_chat_menu_button.assert_awaited_once()
    menu_options = harness.bot.set_chat_menu_button.await_args.kwargs
    assert isinstance(menu_options["menu_button"], MenuButtonCommands)
    assert "chat_id" not in menu_options
    harness.bot.delete_webhook.assert_not_awaited()


def test_main_clears_every_pending_update_after_setup_before_polling(
    harness: StartupHarness,
) -> None:
    asyncio.run(startup.main())

    assert harness.events == [
        "initialize",
        "import",
        "commands",
        "description",
        "menu",
        "drop",
        "poll",
        "bot_closed",
        "client_closed",
        "pool_closed",
    ]
    harness.bot.delete_webhook.assert_awaited_once_with(
        drop_pending_updates=True
    )
    harness.dispatcher.start_polling.assert_awaited_once()
    poll = harness.dispatcher.start_polling.await_args
    assert poll.args == (harness.bot,)
    assert poll.kwargs["close_bot_session"] is False
    assert poll.kwargs["pending_questions"] == {}


@pytest.mark.parametrize(
    ("method", "failure"),
    [
        ("set_my_commands", "api"),
        ("set_my_description", "timeout"),
        ("set_chat_menu_button", "api"),
        ("delete_webhook", "timeout"),
    ],
)
def test_failed_startup_prevents_polling_and_closes_resources(
    harness: StartupHarness,
    caplog: pytest.LogCaptureFixture,
    method: str,
    failure: str,
) -> None:
    getattr(harness.bot, method).side_effect = startup_error(failure)

    with pytest.raises(SystemExit) as caught:
        asyncio.run(startup.main())

    assert caught.value.code == 1
    assert caught.value.__suppress_context__
    error_output = caplog.text + "".join(
        traceback.format_exception(caught.type, caught.value, caught.tb)
    )
    assert "not-a-real-secret" not in error_output
    getattr(harness.bot, method).assert_awaited_once()
    if method == "delete_webhook":
        harness.bot.delete_webhook.assert_awaited_once_with(
            drop_pending_updates=True
        )
    else:
        harness.bot.delete_webhook.assert_not_awaited()
    harness.dispatcher.start_polling.assert_not_awaited()
    assert harness.events[:2] == ["initialize", "import"]
    assert harness.events[-3:] == [
        "bot_closed", "client_closed", "pool_closed"
    ]


def startup_error(kind: str) -> Exception:
    """Include a fake private URL to check that provider errors stay hidden."""
    detail = "Request failed: https://api.telegram.org/botnot-a-real-secret"
    if kind == "timeout":
        return TimeoutError(detail)
    return TelegramAPIError(
        method=DeleteWebhook(drop_pending_updates=True), message=detail
    )
