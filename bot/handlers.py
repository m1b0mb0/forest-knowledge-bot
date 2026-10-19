"""Telegram message handlers."""

import asyncio
import sqlite3

from aiogram import F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from bot.services.llm import ArchiveAssistant, AssistantError
from bot.services.uploads import UploadError, UploadService
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
        "Якщо запитання стосується певного місця, зазнач його назву.\n"
        "Також можна надіслати PDF, DOCX або TXT, "
        "щоб додати файл до спільного архіву.\n"
        "Твої файли: /my_documents. Видалення: /delete <id>."
    )


@router.message(Command("my_documents"))
async def my_documents_handler(message: Message, store: Store) -> None:
    """List only the current sender's non-deleted Telegram documents."""
    owner_id = _personal_sender_id(message)
    if owner_id is None:
        await message.answer(
            "Для керування файлами напиши від свого особистого "
            "облікового запису."
        )
        return
    try:
        documents = await asyncio.to_thread(
            store.list_telegram_documents, owner_id
        )
    except (sqlite3.Error, OSError):
        await message.answer(
            "Не вдалося отримати список файлів. Спробуй ще раз."
        )
        return
    if not documents:
        await message.answer("У тебе ще немає завантажених документів.")
        return

    text = "Твої документи. Для видалення: /delete <id>.\n"
    for document in documents:
        name = " ".join(document.name.split())
        if len(name) > 500:
            name = name[:497] + "..."
        line = f"\n{document.id} — {name}"
        if len((text + line).encode("utf-16-le")) // 2 > 4000:
            await message.answer(text)
            text = "Твої документи (продовження):\n"
        text += line
    await message.answer(text)


@router.message(Command("delete"))
async def delete_document_handler(
    message: Message,
    command: CommandObject,
    store: Store,
    pending_questions: dict[tuple[int, int], str],
) -> None:
    """Delete only a document whose owner matches the current sender."""
    owner_id = _personal_sender_id(message)
    if owner_id is None:
        await message.answer(
            "Для керування файлами напиши від свого особистого "
            "облікового запису."
        )
        return
    document_id = _document_id(command.args)
    if document_id is None:
        await message.answer(
            "Використай /delete <id>. ID свого файла знайдеш у /my_documents."
        )
        return
    try:
        deleted = await asyncio.to_thread(
            store.delete_telegram_document, document_id, owner_id
        )
    except (sqlite3.Error, OSError):
        await message.answer("Не вдалося видалити документ. Спробуй ще раз.")
        return
    if not deleted:
        await message.answer(
            "Документ недоступний для видалення. "
            "Перевір список своїх файлів: /my_documents."
        )
        return
    pending_questions.pop((message.chat.id, owner_id), None)
    await message.answer(
        "Документ видалено. Його вміст більше не використовується "
        "у відповідях."
    )


def _personal_sender_id(message: Message) -> int | None:
    """Anonymous chat senders do not have a personal file owner."""
    if message.from_user is None or message.sender_chat is not None:
        return None
    return message.from_user.id


def _document_id(arguments: str | None) -> int | None:
    value = (arguments or "").strip()
    if not value.isascii() or not value.isdecimal() or len(value) > 19:
        return None
    identifier = int(value)
    return identifier if 0 < identifier <= 2**63 - 1 else None


@router.message(F.document)
async def document_handler(
    message: Message,
    uploads: UploadService,
    pending_questions: dict[tuple[int, int], str],
) -> None:
    """Confirm an upload only after its text and original are persisted."""
    try:
        async with ChatActionSender.typing(
            bot=message.bot, chat_id=message.chat.id,
        ):
            await uploads.import_document(message)
    except UploadError as error:
        await message.answer(str(error))
        return

    if message.from_user is not None:
        pending_questions.pop((message.chat.id, message.from_user.id), None)
    await message.answer(
        "Файл додано до спільного архіву. "
        "Тепер можна ставити запитання за його вмістом."
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

    try:
        documents = await asyncio.to_thread(store.list_documents)
        async with ChatActionSender.typing(
            bot=message.bot, chat_id=message.chat.id,
        ):
            reply = await assistant.answer(question, documents)
        if reply.source_ids and not await asyncio.to_thread(
            store.are_documents_active, reply.source_ids
        ):
            raise AssistantError(
                "Архів змінився під час підготовки відповіді. "
                "Повтори запитання."
            )
    except (AssistantError, sqlite3.Error, OSError) as error:
        if previous_question and conversation is not None:
            pending_questions[conversation] = previous_question
        text = (
            str(error) if isinstance(error, AssistantError)
            else "Не вдалося прочитати архів. Спробуй ще раз."
        )
        await message.answer(text)
        return

    if reply.needs_clarification and conversation is not None:
        pending_questions[conversation] = original_question
    await message.answer(reply.text)
