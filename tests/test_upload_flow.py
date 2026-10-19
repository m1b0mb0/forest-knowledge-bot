"""Upload as one user, reopen storage, and answer another without APIs."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from aiogram.types import Document as TelegramDocument
from aiogram.types import File, Message, User

from bot.handlers import document_handler, text_handler
from bot.services import llm
from bot.services.uploads import UploadService
from bot.storage import Store


def test_uploaded_text_is_shared_persistent_and_cited_with_real_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    store.upsert_drive_document(
        "unrelated", "Догляд за садом.txt", b"garden", "Поливати сад.", None
    )
    text = "Майстерня відкрита щовівторка о 17:30."
    content = text.encode("utf-8")
    filename = "Розклад майстерні рукавички Біля клена.txt"
    context = MagicMock()
    context.__aenter__ = AsyncMock()
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(
        "bot.handlers.ChatActionSender.typing", Mock(return_value=context)
    )
    monkeypatch.setattr(llm, "_count_tokens", lambda value: len(value.split()))

    async def download(
        file_path: str, *, destination: BinaryIO, **kwargs: object,
    ) -> BinaryIO:
        destination.write(content)
        return destination

    bot = SimpleNamespace(
        get_file=AsyncMock(return_value=File(
            file_id="fictional-file", file_unique_id="fictional-unique",
            file_size=len(content), file_path="documents/example.txt",
        )),
        download_file=AsyncMock(side_effect=download),
    )
    sent_at = datetime(2026, 1, 2, 10, 30, tzinfo=timezone.utc)
    uploader = cast(Message, SimpleNamespace(
        text=None,
        message_id=12,
        chat=SimpleNamespace(id=10),
        from_user=User(
            id=1001, is_bot=False, first_name="Автор", username="test_author"
        ),
        sender_chat=None,
        date=sent_at,
        document=TelegramDocument(
            file_id="fictional-file", file_unique_id="fictional-unique",
            file_name=filename, file_size=len(content),
        ),
        bot=bot,
        answer=AsyncMock(),
    ))
    reader = cast(Message, SimpleNamespace(
        text="Коли відкрита майстерня біля клена?",
        chat=SimpleNamespace(id=20),
        from_user=User(id=2002, is_bot=False, first_name="Читач"),
        bot=bot,
        answer=AsyncMock(),
    ))
    create = AsyncMock()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    def completion(payload: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps(payload)),
            )],
            usage=None,
        )

    async def scenario() -> None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            uploads = UploadService(store, executor)
            await document_handler(uploader, uploads, {})
        reopened = Store(tmp_path)
        uploaded = next(
            item for item in reopened.list_documents()
            if item.source == "telegram"
        )
        assert uploaded.owner_id == 1001
        assert uploaded.uploaded_at == sent_at.isoformat()
        assert (tmp_path / uploaded.relative_path).read_bytes() == content
        create.side_effect = [
            completion({
                "status": "selected", "document_ids": [uploaded.id],
                "text": "",
            }),
            completion({
                "status": "answer", "text": text,
                "evidence": [{"source_id": 1, "quote": text}],
            }),
        ]
        assistant = llm.ArchiveAssistant(client, "test-model")
        await text_handler(reader, assistant, reopened, {})

    asyncio.run(scenario())

    assert "додано до спільного архіву" in (
        uploader.answer.await_args.args[0]
    )
    answer = reader.answer.await_args.args[0]
    assert text in answer
    assert "«Розклад майстерні» рукавички «Біля клена»" in answer
    assert "@test_author" in answer
    assert "02.01.2026 10:30" in answer and "UTC" in answer
    assert create.await_count == 2
    for call in create.await_args_list:
        request = json.dumps(call.kwargs, ensure_ascii=False)
        assert "@test_author" not in request
        assert sent_at.isoformat() not in request
