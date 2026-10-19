"""Own-document commands without Telegram, database, or Groq requests."""

import asyncio
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.filters import CommandObject
from aiogram.types import Message

from bot.handlers import delete_document_handler, my_documents_handler
from bot.storage import Document, Store


def message(
    *,
    user_id: int | None = 4_000_000_001,
    sender_chat: object | None = None,
) -> Message:
    return cast(
        Message,
        SimpleNamespace(
            chat=SimpleNamespace(id=-100_123_456_789),
            from_user=(
                SimpleNamespace(id=user_id) if user_id is not None else None
            ),
            sender_chat=sender_chat,
            answer=AsyncMock(),
        ),
    )


def document(document_id: int, name: str) -> Document:
    return Document(
        id=document_id,
        source="telegram",
        source_id=f"chat:{document_id}",
        name=name,
        relative_path=Path("documents") / f"{document_id}.bin",
        text="Приклад тексту.",
        modified_at=None,
        created_at="2026-01-02T10:30:00+00:00",
        imported_at="2026-01-02T10:30:00+00:00",
        active=True,
        owner_id=4_000_000_001,
        owner_name="@sender",
        uploaded_at="2026-01-02T10:30:00+00:00",
    )


def answers(incoming: Message) -> list[str]:
    return [call.args[0] for call in incoming.answer.await_args_list]


def test_list_uses_sender_id_and_shows_actual_document_names() -> None:
    incoming = message()
    store = Mock(spec=Store)
    store.list_telegram_documents.return_value = [
        document(71, "Поради садівникові.txt"),
        document(82, "Правила біля озера.pdf"),
    ]

    asyncio.run(my_documents_handler(incoming, store))

    store.list_telegram_documents.assert_called_once_with(4_000_000_001)
    text = "\n".join(answers(incoming))
    assert "Поради садівникові.txt" in text
    assert "Правила біля озера.pdf" in text
    assert "71" in text and "82" in text
    assert "/delete" in text
    store.delete_telegram_document.assert_not_called()


def test_empty_list_has_a_friendly_response() -> None:
    incoming = message()
    store = Mock(spec=Store)
    store.list_telegram_documents.return_value = []

    asyncio.run(my_documents_handler(incoming, store))

    incoming.answer.assert_awaited_once()
    assert answers(incoming)[0].strip()
    assert "[]" not in answers(incoming)[0]
    store.delete_telegram_document.assert_not_called()


def test_large_unicode_list_preserves_ids_within_telegram_length() -> None:
    incoming = message()
    documents = [
        document(index, f"Файл {index} " + "🌳" * 35 + ".txt")
        for index in range(100, 180)
    ]
    store = Mock(spec=Store)
    store.list_telegram_documents.return_value = documents

    asyncio.run(my_documents_handler(incoming, store))

    sent = answers(incoming)
    assert len(sent) > 1
    assert all(len(text.encode("utf-16-le")) // 2 <= 4000 for text in sent)
    combined = "\n".join(sent)
    for item in documents:
        assert item.name in combined
        assert re.search(rf"(?<!\d){item.id}(?!\d)", combined)
    assert "/delete" in combined


def test_long_multiline_filename_cannot_break_list_or_message_limit() -> None:
    incoming = message()
    name = "Звіт\r\nпідроблений рядок\t" + "🌳" * 3000 + ".txt"
    store = Mock(spec=Store)
    store.list_telegram_documents.return_value = [document(42, name)]

    asyncio.run(my_documents_handler(incoming, store))

    sent = answers(incoming)
    combined = "\n".join(sent)
    assert "42" in combined and "Звіт" in combined
    assert "Звіт\r\nпідроблений рядок" not in combined
    assert "\r" not in combined and "\t" not in combined
    assert "🌳" * 1000 not in combined
    assert all(len(text.encode("utf-16-le")) // 2 <= 4000 for text in sent)
    assert "/delete" in combined


@pytest.mark.parametrize("command_name", ["list", "delete"])
@pytest.mark.parametrize(
    ("user_id", "sender_chat"),
    [
        (None, None),
        (777_000, SimpleNamespace(id=-100_999)),
    ],
)
def test_anonymous_or_missing_sender_cannot_access_document_commands(
    command_name: str,
    user_id: int | None,
    sender_chat: object | None,
) -> None:
    incoming = message(user_id=user_id, sender_chat=sender_chat)
    store = Mock(spec=Store)
    pending = {(-100_123_456_789, 777_000): "Earlier question"}
    previous = pending.copy()

    if command_name == "list":
        asyncio.run(my_documents_handler(incoming, store))
    else:
        asyncio.run(delete_document_handler(
            incoming, CommandObject(command="delete", args="42"),
            store, pending,
        ))

    store.list_telegram_documents.assert_not_called()
    store.delete_telegram_document.assert_not_called()
    incoming.answer.assert_awaited_once()
    assert answers(incoming)[0].strip()
    assert pending == previous


@pytest.mark.parametrize(
    "args",
    [
        None, "", "   ", "abc", "1.5", "0", "-1", "1 2",
        "42 extra", "9223372036854775808", "9" * 100,
    ],
)
def test_invalid_delete_id_does_not_touch_storage(args: str | None) -> None:
    incoming = message()
    store = Mock(spec=Store)
    pending = {(-100_123_456_789, 4_000_000_001): "Earlier question"}
    previous = pending.copy()

    asyncio.run(delete_document_handler(
        incoming, CommandObject(command="delete", args=args), store, pending,
    ))

    store.delete_telegram_document.assert_not_called()
    store.list_telegram_documents.assert_not_called()
    incoming.answer.assert_awaited_once()
    assert "/delete" in answers(incoming)[0]
    assert "/my_documents" in answers(incoming)[0]
    assert pending == previous


@pytest.mark.parametrize("args", ["42", "  42  ", "9223372036854775807"])
def test_successful_delete_uses_sender_and_clears_only_own_pending(
    args: str,
) -> None:
    incoming = message()
    store = Mock(spec=Store)
    store.delete_telegram_document.return_value = True
    pending = {
        (-100_123_456_789, 4_000_000_001): "Own question",
        (-100_123_456_789, 4_000_000_002): "Another sender",
        (20, 4_000_000_001): "Another chat",
    }

    asyncio.run(delete_document_handler(
        incoming, CommandObject(command="delete", args=args), store, pending,
    ))

    store.delete_telegram_document.assert_called_once_with(
        int(args), 4_000_000_001,
    )
    assert pending == {
        (-100_123_456_789, 4_000_000_002): "Another sender",
        (20, 4_000_000_001): "Another chat",
    }
    incoming.answer.assert_awaited_once()
    assert answers(incoming)[0].strip()


def test_denied_or_repeated_delete_returns_same_safe_response() -> None:
    incoming = message()
    store = Mock(spec=Store)
    store.delete_telegram_document.return_value = False
    pending = {(-100_123_456_789, 4_000_000_001): "Earlier question"}
    previous = pending.copy()

    async def scenario() -> None:
        for identifier in (42, 52):
            await delete_document_handler(
                incoming,
                CommandObject(command="delete", args=str(identifier)),
                store, pending,
            )

    asyncio.run(scenario())

    assert store.delete_telegram_document.call_count == 2
    assert pending == previous
    assert answers(incoming)[0] == answers(incoming)[1]
    assert answers(incoming)[0].strip()


@pytest.mark.parametrize("command_name", ["list", "delete"])
@pytest.mark.parametrize("error_type", [sqlite3.OperationalError, OSError])
def test_storage_failure_is_safe_and_preserves_pending(
    command_name: str, error_type: type[Exception],
) -> None:
    incoming = message()
    store = Mock(spec=Store)
    private_details = "private/database/location.sqlite3 sensitive-value"
    store.list_telegram_documents.side_effect = error_type(private_details)
    store.delete_telegram_document.side_effect = error_type(private_details)
    pending = {(-100_123_456_789, 4_000_000_001): "Earlier question"}
    previous = pending.copy()

    if command_name == "list":
        asyncio.run(my_documents_handler(incoming, store))
    else:
        asyncio.run(delete_document_handler(
            incoming, CommandObject(command="delete", args="42"),
            store, pending,
        ))

    incoming.answer.assert_awaited_once()
    text = answers(incoming)[0]
    assert text.strip()
    assert "private/" not in text
    assert "sensitive-value" not in text
    assert pending == previous
