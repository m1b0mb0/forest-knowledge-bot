"""Telegram answer and clarification flow without Telegram or Groq requests."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.types import Message

from bot.handlers import start_handler, text_handler
from bot.services.llm import ArchiveAssistant, AssistantError, AssistantReply
from bot.storage import Store


def message(
    text: str | None,
    *,
    chat_id: int = 10,
    user_id: int | None = 20,
) -> Message:
    return cast(
        Message,
        SimpleNamespace(
            text=text,
            chat=SimpleNamespace(id=chat_id),
            from_user=(
                SimpleNamespace(id=user_id) if user_id is not None else None
            ),
            answer=AsyncMock(),
        ),
    )


def test_answer_uses_fresh_documents_and_is_sent_to_the_user() -> None:
    incoming = message("  Скільки внесок під дубом?  ")
    store = Mock(spec=Store)
    store.list_documents.side_effect = [[], ["newly imported document"]]
    assistant = Mock(spec=ArchiveAssistant)
    assistant.answer = AsyncMock(return_value=AssistantReply("Відповідь."))
    pending: dict[tuple[int, int], str] = {}

    async def scenario() -> None:
        await text_handler(incoming, assistant, store, pending)
        await text_handler(incoming, assistant, store, pending)

    asyncio.run(scenario())

    assert store.list_documents.call_count == 2
    assert assistant.answer.await_args_list[0].args == (
        "Скільки внесок під дубом?", []
    )
    assert assistant.answer.await_args_list[1].args == (
        "Скільки внесок під дубом?", ["newly imported document"]
    )
    assert incoming.answer.await_count == 2
    incoming.answer.assert_awaited_with("Відповідь.")
    assert not pending


def test_clarification_retains_only_original_question_until_answer() -> None:
    store = Mock(spec=Store)
    store.list_documents.return_value = []
    assistant = Mock(spec=ArchiveAssistant)
    assistant.answer = AsyncMock(
        side_effect=[
            AssistantReply("Для якого місця?", needs_clarification=True),
            AssistantReply("Для якого періоду?", needs_clarification=True),
            AssistantReply("4 жолуді."),
        ]
    )
    pending: dict[tuple[int, int], str] = {}

    async def scenario() -> None:
        await text_handler(
            message("Скільки внесок?"), assistant, store, pending
        )
        assert pending == {(10, 20): "Скільки внесок?"}
        await text_handler(message("Під дубом"), assistant, store, pending)
        assert pending == {(10, 20): "Скільки внесок?"}
        await text_handler(message("Щомісячного"), assistant, store, pending)

    asyncio.run(scenario())

    final_question = assistant.answer.await_args.args[0]
    assert "Скільки внесок?" in final_question
    assert "Під дубом" not in final_question
    assert "Щомісячного" in final_question
    assert not pending


@pytest.mark.parametrize(
    ("chat_id", "user_id"), [(10, 21), (11, 20), (10, None)]
)
def test_pending_question_is_isolated_by_both_chat_and_user(
    chat_id: int, user_id: int | None
) -> None:
    store = Mock(spec=Store)
    store.list_documents.return_value = []
    assistant = Mock(spec=ArchiveAssistant)
    assistant.answer = AsyncMock(
        return_value=AssistantReply("Уточни місце.", needs_clarification=True)
    )
    pending = {(10, 20): "private earlier question"}
    incoming = message("Під дубом", chat_id=chat_id, user_id=user_id)

    asyncio.run(text_handler(incoming, assistant, store, pending))

    assistant.answer.assert_awaited_once_with("Під дубом", [])
    assert pending[(10, 20)] == "private earlier question"
    if user_id is None:
        assert len(pending) == 1
    else:
        assert pending[(chat_id, user_id)] == "Під дубом"


def test_assistant_error_is_shown_and_does_not_retain_question() -> None:
    incoming = message("Під дубом")
    store = Mock(spec=Store)
    store.list_documents.return_value = []
    assistant = Mock(spec=ArchiveAssistant)
    assistant.answer = AsyncMock(
        side_effect=AssistantError("Ліміт запитів вичерпано. Спробуй пізніше.")
    )
    pending = {(10, 20): "Скільки внесок?"}

    asyncio.run(text_handler(incoming, assistant, store, pending))

    incoming.answer.assert_awaited_once_with(
        "Ліміт запитів вичерпано. Спробуй пізніше."
    )
    assert not pending


@pytest.mark.parametrize("text", [None, "", "   ", "/unknown"])
def test_empty_input_and_unknown_commands_do_not_call_groq(
    text: str | None,
) -> None:
    incoming = message(text)
    store = Mock(spec=Store)
    assistant = Mock(spec=ArchiveAssistant)
    assistant.answer = AsyncMock()
    pending = {(10, 20): "Скільки внесок?"}

    asyncio.run(text_handler(incoming, assistant, store, pending))

    assistant.answer.assert_not_awaited()
    store.list_documents.assert_not_called()
    incoming.answer.assert_awaited_once_with(
        "Напиши запитання про документи архіву."
    )
    assert not pending


def test_start_resets_only_the_senders_pending_question() -> None:
    incoming = message("/start")
    pending = {(10, 20): "own question", (10, 21): "another users question"}

    asyncio.run(start_handler(incoming, pending))

    assert pending == {(10, 21): "another users question"}
    incoming.answer.assert_awaited_once()
    assert "Лісовий архів" in incoming.answer.await_args.args[0]
