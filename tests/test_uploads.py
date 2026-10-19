"""Telegram upload boundaries without Telegram or Groq network requests."""

import asyncio
import multiprocessing
import sqlite3
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from aiogram.exceptions import TelegramAPIError, TelegramNetworkError
from aiogram.methods import GetFile
from aiogram.types import Message

from bot.services import uploads
from bot.services.uploads import UploadError, UploadService
from bot.storage import Store

TEXT = "Правила: повертатися до заходу сонця."
CONTENT = b"\xef\xbb\xbf" + TEXT.encode("utf-8")
UPLOADED_AT = datetime(2026, 10, 19, 10, 30, tzinfo=timezone.utc)


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    with ThreadPoolExecutor(max_workers=1) as worker:
        yield worker


def message(content: bytes = CONTENT, **metadata: Any) -> Message:
    async def download(
        file_path: str, *, destination: Any, **kwargs: Any
    ) -> Any:
        for offset in range(0, len(content), 4):
            destination.write(content[offset:offset + 4])
        destination.seek(0)
        return destination

    incoming = SimpleNamespace(
        document=SimpleNamespace(
            file_id="telegram-file-id",
            file_unique_id="same-content-id",
            file_name="rules.txt",
            file_size=len(content),
            mime_type="text/plain",
        ),
        from_user=SimpleNamespace(
            id=20, username="forest_reader", full_name="Forest Reader"
        ),
        chat=SimpleNamespace(id=10),
        message_id=30,
        sender_chat=None,
        date=UPLOADED_AT,
        bot=SimpleNamespace(
            get_file=AsyncMock(
                return_value=SimpleNamespace(
                    file_path="documents/telegram-file.txt",
                    file_size=len(content),
                )
            ),
            download_file=AsyncMock(side_effect=download),
        ),
        **metadata,
    )
    return cast(Message, incoming)


def test_txt_survives_reopening_with_current_sender_metadata(
    tmp_path: Path, executor: ThreadPoolExecutor
) -> None:
    store = Store(tmp_path)
    store.initialize()
    incoming = message(
        forward_origin=SimpleNamespace(sender_user=SimpleNamespace(id=999))
    )
    incoming.document.file_name = "C:\\untrusted\\  forest   rules.TXT"

    service = UploadService(store, executor)
    result = asyncio.run(service.import_document(incoming))

    saved, = Store(tmp_path).list_documents()
    assert result == saved.id
    assert saved.source == "telegram"
    assert saved.source_id == "10:30"
    assert saved.name == "forest rules.TXT"
    assert saved.text == TEXT
    assert saved.owner_id == 20
    assert saved.owner_name == "@forest_reader"
    assert saved.uploaded_at == UPLOADED_AT.isoformat()
    assert saved.active
    assert (tmp_path / saved.relative_path).read_bytes() == CONTENT
    incoming.bot.get_file.assert_awaited_once_with(
        "telegram-file-id", request_timeout=30
    )
    download = incoming.bot.download_file.await_args
    assert download.args == ("documents/telegram-file.txt",)
    assert download.kwargs["timeout"] == 30
    assert download.kwargs["chunk_size"] == 64 * 1024


def test_same_file_from_different_users_creates_distinct_uploads(
    tmp_path: Path, executor: ThreadPoolExecutor
) -> None:
    store = Store(tmp_path)
    store.initialize()
    first = message()
    second = message()
    second.message_id = 31
    second.from_user.id = 21
    second.from_user.username = None
    second.from_user.full_name = "Інший читач"
    service = UploadService(store, executor)

    async def scenario() -> None:
        await service.import_document(first)
        await service.import_document(second)

    asyncio.run(scenario())

    saved = store.list_documents()
    assert len(saved) == 2
    assert {item.source_id for item in saved} == {"10:30", "10:31"}
    assert {item.owner_id for item in saved} == {20, 21}
    assert saved[1].owner_name == "Інший читач"


def test_anonymous_chat_sender_cannot_claim_ownership(
    executor: ThreadPoolExecutor,
) -> None:
    incoming = message()
    incoming.sender_chat = SimpleNamespace(id=-100123)
    incoming.from_user.id = 1087968824
    incoming.from_user.username = "GroupAnonymousBot"
    store = Mock(spec=Store)

    with pytest.raises(UploadError, match="особистого"):
        asyncio.run(UploadService(store, executor).import_document(incoming))

    incoming.bot.get_file.assert_not_awaited()
    incoming.bot.download_file.assert_not_awaited()
    store.add_telegram_document.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document", None),
        ("from_user", None),
        ("file_name", None),
        ("file_name", ""),
        ("file_name", "folder/"),
        ("file_name", "folder/   "),
        ("file_name", "picture.jpg"),
        ("file_name", "old.doc"),
        ("file_name", "text"),
        ("file_size", 0),
        ("file_size", -1),
        ("file_size", uploads.MAX_DOCUMENT_BYTES + 1),
    ],
)
def test_invalid_metadata_is_rejected_before_network(
    field: str, value: Any, executor: ThreadPoolExecutor
) -> None:
    incoming = message()
    if field in {"document", "from_user"}:
        setattr(incoming, field, value)
    else:
        setattr(incoming.document, field, value)
    store = Mock(spec=Store)

    with pytest.raises(UploadError):
        asyncio.run(UploadService(store, executor).import_document(incoming))

    incoming.bot.get_file.assert_not_awaited()
    incoming.bot.download_file.assert_not_awaited()
    store.add_telegram_document.assert_not_called()


@pytest.mark.parametrize("size", [0, -1, uploads.MAX_DOCUMENT_BYTES + 1])
def test_get_file_size_is_checked_before_download(
    size: int, executor: ThreadPoolExecutor
) -> None:
    incoming = message()
    incoming.bot.get_file.return_value.file_size = size
    store = Mock(spec=Store)

    with pytest.raises(UploadError, match="10 МіБ"):
        asyncio.run(UploadService(store, executor).import_document(incoming))

    incoming.bot.download_file.assert_not_awaited()
    store.add_telegram_document.assert_not_called()


@pytest.mark.parametrize("declared_size", [None, 1])
def test_actual_download_is_bounded_even_with_wrong_or_missing_size(
    declared_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
    executor: ThreadPoolExecutor,
) -> None:
    monkeypatch.setattr(uploads, "MAX_DOCUMENT_BYTES", 8)
    incoming = message(b"123456789")
    incoming.document.file_size = declared_size
    incoming.bot.get_file.return_value.file_size = declared_size
    store = Mock(spec=Store)

    with pytest.raises(UploadError, match="10 МіБ"):
        asyncio.run(UploadService(store, executor).import_document(incoming))

    call = incoming.bot.download_file.await_args
    destination = call.kwargs["destination"]
    assert destination.closed
    store.add_telegram_document.assert_not_called()


def test_missing_sizes_are_accepted_when_actual_content_is_valid(
    executor: ThreadPoolExecutor,
) -> None:
    incoming = message()
    incoming.document.file_size = None
    incoming.bot.get_file.return_value.file_size = None
    store = Mock(spec=Store)
    store.add_telegram_document.return_value = 7

    assert asyncio.run(
        UploadService(store, executor).import_document(incoming)
    ) == 7
    assert store.add_telegram_document.call_args.kwargs["text"] == TEXT


@pytest.mark.parametrize("file_path", [None, ""])
def test_missing_download_path_is_rejected(
    file_path: str | None, executor: ThreadPoolExecutor
) -> None:
    incoming = message()
    incoming.bot.get_file.return_value.file_path = file_path
    store = Mock(spec=Store)

    with pytest.raises(UploadError, match="Telegram"):
        asyncio.run(UploadService(store, executor).import_document(incoming))

    incoming.bot.download_file.assert_not_awaited()
    store.add_telegram_document.assert_not_called()


@pytest.mark.parametrize("operation", ["get_file", "download_file"])
@pytest.mark.parametrize(
    "error_type",
    [TelegramAPIError, TelegramNetworkError, TimeoutError, OSError],
)
def test_download_errors_hide_token_urls_even_in_tracebacks(
    operation: str,
    error_type: type[Exception],
    executor: ThreadPoolExecutor,
) -> None:
    secret_url = "https://api.telegram.org/file/bot123:secret-token/private"
    if issubclass(error_type, TelegramAPIError):
        error = error_type(method=GetFile(file_id="file"), message=secret_url)
    else:
        error = error_type(secret_url)
    incoming = message()
    getattr(incoming.bot, operation).side_effect = error
    store = Mock(spec=Store)

    with pytest.raises(UploadError, match="Telegram") as caught:
        asyncio.run(UploadService(store, executor).import_document(incoming))

    rendered = "".join(traceback.format_exception(caught.value))
    assert "secret-token" not in rendered
    assert "api.telegram.org" not in rendered
    store.add_telegram_document.assert_not_called()


@pytest.mark.parametrize("payload_error", [False, True])
def test_aiohttp_stream_errors_hide_token_urls_in_tracebacks(
    payload_error: bool, executor: ThreadPoolExecutor
) -> None:
    secret_url = "https://api.telegram.org/file/bot123:secret-token/private"
    if payload_error:
        error = aiohttp.ClientPayloadError(secret_url)
    else:
        error = aiohttp.ClientResponseError(
            request_info=SimpleNamespace(real_url=secret_url),
            history=(),
            status=403,
            message="Cannot download file",
        )
    incoming = message()
    incoming.bot.download_file.side_effect = error
    store = Mock(spec=Store)

    with pytest.raises(UploadError, match="Telegram") as caught:
        asyncio.run(UploadService(store, executor).import_document(incoming))

    rendered = "".join(traceback.format_exception(caught.value))
    assert "secret-token" not in rendered
    assert "api.telegram.org" not in rendered
    store.add_telegram_document.assert_not_called()


@pytest.mark.parametrize(
    ("name", "content"),
    [("bad.txt", b"\xff\xfe\xfa"), ("bad.pdf", b"broken PDF"),
     ("bad.docx", b"broken DOCX"), ("blank.txt", b"   ")],
)
def test_unreadable_documents_are_not_persisted(
    name: str, content: bytes, tmp_path: Path, executor: ThreadPoolExecutor
) -> None:
    store = Store(tmp_path)
    store.initialize()
    incoming = message(content)
    incoming.document.file_name = name

    with pytest.raises(UploadError, match="прочитати"):
        asyncio.run(UploadService(store, executor).import_document(incoming))

    assert store.list_documents() == []
    assert list((tmp_path / "documents").iterdir()) == []


def test_empty_actual_download_is_rejected(
    executor: ThreadPoolExecutor,
) -> None:
    incoming = message(b"")
    incoming.document.file_size = 1
    incoming.bot.get_file.return_value.file_size = 1
    store = Mock(spec=Store)

    with pytest.raises(UploadError, match="10 МіБ"):
        asyncio.run(UploadService(store, executor).import_document(incoming))

    store.add_telegram_document.assert_not_called()


@pytest.mark.parametrize("error_type", [sqlite3.OperationalError, OSError])
def test_storage_errors_have_safe_messages(
    error_type: type[Exception], executor: ThreadPoolExecutor
) -> None:
    incoming = message()
    store = Mock(spec=Store)
    store.add_telegram_document.side_effect = error_type(
        "private-storage-path"
    )

    with pytest.raises(UploadError, match="зберегти") as caught:
        asyncio.run(UploadService(store, executor).import_document(incoming))

    assert "private-storage-path" not in "".join(
        traceback.format_exception(caught.value)
    )


def test_spawned_worker_parses_txt_without_blocking_the_event_loop(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        service = UploadService(store, executor)

        async def scenario() -> tuple[int, int]:
            task = asyncio.create_task(service.import_document(message()))
            beats = 0
            while not task.done():
                await asyncio.sleep(0.01)
                beats += 1
            return await task, beats

        document_id, beats = asyncio.run(scenario())

    assert beats > 0
    saved, = store.list_documents()
    assert saved.id == document_id
    assert saved.text == TEXT
