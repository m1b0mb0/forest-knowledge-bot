"""Telegram answer and clarification flow without Telegram or Groq requests."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from aiogram.types import Message

from bot.handlers import document_handler, start_handler, text_handler
from bot.services.llm import ArchiveAssistant, AssistantError, AssistantReply
from bot.services.uploads import UploadError, UploadService
from bot.storage import Document, Store


@pytest.fixture(autouse=True)
def typing(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the background Telegram action with an offline context."""
    context = MagicMock()
    context.__aenter__ = AsyncMock()
    context.__aexit__ = AsyncMock(return_value=False)
    sender = Mock(return_value=context)
    monkeypatch.setattr("bot.handlers.ChatActionSender.typing", sender)
    return sender


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
            bot=Mock(),
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


def test_typing_is_active_while_waiting_for_the_assistant(typing: Mock) -> None:
    incoming = message("Коли повертатися?")
    store = Mock(spec=Store)
    store.list_documents.return_value = []
    assistant = Mock(spec=ArchiveAssistant)
    context = typing.return_value

    async def answer(
        question: str, documents: list[Document],
    ) -> AssistantReply:
        context.__aenter__.assert_awaited_once()
        context.__aexit__.assert_not_awaited()
        return AssistantReply("До заходу сонця.")

    assistant.answer = AsyncMock(side_effect=answer)

    asyncio.run(text_handler(incoming, assistant, store, {}))

    typing.assert_called_once_with(bot=incoming.bot, chat_id=incoming.chat.id)
    context.__aexit__.assert_awaited_once()
    incoming.answer.assert_awaited_once_with("До заходу сонця.")


def test_assistant_error_preserves_original_question_for_retry(
    typing: Mock,
) -> None:
    incoming = message("Під дубом")
    store = Mock(spec=Store)
    store.list_documents.return_value = []
    assistant = Mock(spec=ArchiveAssistant)
    assistant.answer = AsyncMock(
        side_effect=[
            AssistantError("Ліміт запитів вичерпано. Спробуй пізніше."),
            AssistantReply("4 жолуді."),
        ],
    )
    pending = {(10, 20): "Скільки внесок?"}

    async def scenario() -> None:
        await text_handler(incoming, assistant, store, pending)
        assert pending == {(10, 20): "Скільки внесок?"}
        incoming.answer.assert_awaited_once_with(
            "Ліміт запитів вичерпано. Спробуй пізніше."
        )
        assert (
            typing.return_value.__aexit__.await_args.args[0]
            is AssistantError
        )
        await text_handler(message("Під дубом"), assistant, store, pending)

    asyncio.run(scenario())

    assistant.answer.assert_awaited_with(
        "Початкове запитання: Скільки внесок?\nУточнення: Під дубом", []
    )
    assert not pending


def test_assistant_error_without_pending_question_keeps_no_context() -> None:
    incoming = message("Скільки внесок під дубом?")
    store = Mock(spec=Store)
    store.list_documents.return_value = []
    assistant = Mock(spec=ArchiveAssistant)
    assistant.answer = AsyncMock(side_effect=AssistantError("Спробуй пізніше."))
    pending: dict[tuple[int, int], str] = {}

    asyncio.run(text_handler(incoming, assistant, store, pending))

    assert not pending
    incoming.answer.assert_awaited_once_with("Спробуй пізніше.")


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


def test_upload_confirms_persistence_and_resets_only_the_sender(
    typing: Mock,
) -> None:
    incoming = message(None)
    uploads = Mock(spec=UploadService)
    uploads.import_document = AsyncMock(return_value=123)
    pending = {(10, 20): "own question", (10, 21): "another question"}

    asyncio.run(document_handler(incoming, uploads, pending))

    uploads.import_document.assert_awaited_once_with(incoming)
    assert pending == {(10, 21): "another question"}
    assert "додано до спільного архіву" in (
        incoming.answer.await_args.args[0]
    )
    typing.return_value.__aenter__.assert_awaited_once()
    typing.return_value.__aexit__.assert_awaited_once()


def test_upload_error_preserves_pending_question() -> None:
    incoming = message(None)
    uploads = Mock(spec=UploadService)
    uploads.import_document = AsyncMock(side_effect=UploadError(
        "Не вдалося прочитати файл."
    ))
    pending = {(10, 20): "Скільки внесок?"}

    asyncio.run(document_handler(incoming, uploads, pending))

    incoming.answer.assert_awaited_once_with("Не вдалося прочитати файл.")
    assert pending == {(10, 20): "Скільки внесок?"}
