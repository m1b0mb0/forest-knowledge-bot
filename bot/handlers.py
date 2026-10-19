"""Telegram message handlers."""

import asyncio

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from bot.services.llm import ArchiveAssistant, AssistantError
from bot.storage import Store

router = Router(name=__name__)


@router.message(CommandStart())
async def start_handler(
    message: Message,
    pending_questions: dict[tuple[int, int], str],
) -> None:
    if message.from_user is not None:
        pending_questions.pop((message.chat.id, message.from_user.id), None)
    await message.answer(
        "Вітаю! Я — «Лісовий архів».\n"
        "Став запитання про документи архіву. "
        "Якщо запитання стосується певного місця, зазнач його назву."
    )


@router.message(F.text)
async def text_handler(
    message: Message,
    assistant: ArchiveAssistant,
    store: Store,
    pending_questions: dict[tuple[int, int], str],
) -> None:
    """Answer from current documents, retaining only an unresolved question."""
    conversation = (
        (message.chat.id, message.from_user.id)
        if message.from_user is not None
        else None
    )
    previous_question = (
        pending_questions.pop(conversation, None)
        if conversation is not None
        else None
    )
    question = (message.text or "").strip()
    if not question or question.startswith("/"):
        await message.answer("Напиши запитання про документи архіву.")
        return
    original_question = previous_question or question
    if previous_question:
        question = (
            f"Початкове запитання: {previous_question}\n"
            f"Уточнення: {question}"
        )

    documents = await asyncio.to_thread(store.list_documents)
    try:
        async with ChatActionSender.typing(
            bot=message.bot, chat_id=message.chat.id,
        ):
            reply = await assistant.answer(question, documents)
    except AssistantError as error:
        if previous_question and conversation is not None:
            pending_questions[conversation] = previous_question
        await message.answer(str(error))
        return

    if reply.needs_clarification and conversation is not None:
        pending_questions[conversation] = original_question
    await message.answer(reply.text)
