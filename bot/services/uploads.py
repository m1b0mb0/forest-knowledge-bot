"""Import Telegram documents into the shared local archive."""

import asyncio
import sqlite3
from concurrent.futures import Executor
from io import BytesIO
from pathlib import PurePath

import aiohttp
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from bot.services.documents import (
    MAX_DOCUMENT_BYTES,
    SUPPORTED_EXTENSIONS,
    extract_document_text,
)
from bot.storage import Store

SIZE_ERROR = "Файл має бути непорожнім і не перевищувати 10 МіБ."
FORMAT_ERROR = "Надішли документ у форматі PDF, DOCX або TXT."
DOWNLOAD_ERROR = "Не вдалося завантажити файл із Telegram. Спробуй ще раз."
PARSE_ERROR = (
    "Не вдалося прочитати документ. Перевір його вміст; "
    "TXT має бути в кодуванні UTF-8."
)
SAVE_ERROR = "Не вдалося зберегти документ. Спробуй ще раз."


class UploadError(Exception):
    """An upload failure that is safe to show to the Telegram user."""


class _BoundedBuffer(BytesIO):
    """Stop a streamed download even when Telegram's size is inaccurate."""

    def write(self, data: bytes | bytearray) -> int:
        if self.tell() + len(data) > MAX_DOCUMENT_BYTES:
            raise UploadError(SIZE_ERROR)
        return super().write(data)


class UploadService:
    """Download, parse in a worker process, and persist a new document."""

    def __init__(self, store: Store, executor: Executor) -> None:
        self._store = store
        self._executor = executor

    async def import_document(self, message: Message) -> int:
        """Import one document owned by its current Telegram sender."""
        document = message.document
        sender = message.from_user
        if message.sender_chat is not None:
            raise UploadError(
                "Надішли документ від свого особистого облікового запису."
            )
        if document is None or sender is None or not document.file_name:
            raise UploadError(FORMAT_ERROR)

        name = " ".join(
            document.file_name.replace("\\", "/").rsplit("/", 1)[-1].split()
        )
        extension = PurePath(name).suffix.lower()
        if not name or extension not in SUPPORTED_EXTENSIONS:
            raise UploadError(FORMAT_ERROR)
        self._check_size(document.file_size)

        try:
            async with asyncio.timeout(30):
                remote_file = await message.bot.get_file(
                    document.file_id, request_timeout=30
                )
                self._check_size(remote_file.file_size)
                if not remote_file.file_path:
                    raise UploadError(DOWNLOAD_ERROR)
                with _BoundedBuffer() as destination:
                    await message.bot.download_file(
                        remote_file.file_path,
                        destination=destination,
                        timeout=30,
                        chunk_size=64 * 1024,
                    )
                    content = destination.getvalue()
        except (TelegramAPIError, aiohttp.ClientError, OSError):
            raise UploadError(DOWNLOAD_ERROR) from None

        self._check_size(len(content))
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(
                self._executor, extract_document_text, content, name
            )
        except (ValueError, RuntimeError, OSError):
            raise UploadError(PARSE_ERROR) from None

        owner_name = (
            f"@{sender.username}" if sender.username else sender.full_name
        )
        try:
            return await asyncio.to_thread(
                self._store.add_telegram_document,
                source_id=f"{message.chat.id}:{message.message_id}",
                name=name,
                content=content,
                text=text,
                owner_id=sender.id,
                owner_name=owner_name,
                uploaded_at=message.date,
            )
        except (sqlite3.Error, OSError):
            raise UploadError(SAVE_ERROR) from None

    @staticmethod
    def _check_size(size: int | None) -> None:
        if size is not None and (size <= 0 or size > MAX_DOCUMENT_BYTES):
            raise UploadError(SIZE_ERROR)
