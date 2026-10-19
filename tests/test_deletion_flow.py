"""Deletion excludes stored, cached and in-flight answers without live APIs."""

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.types import Message

from bot.handlers import text_handler
from bot.services import llm
from bot.storage import Document, Store
from tests.helpers import completion, make_client

pytestmark = pytest.mark.usefixtures("offline_token_counts", "typing")

FACT = "Майстерня відкрита щовівторка о 17:30."
QUESTION = "Коли відкрита майстерня?"
ARCHIVE_CHANGED = (
    "Архів змінився під час підготовки відповіді. Повтори запитання."
)


def upload(store: Store, source_id: str = "10:20") -> int:
    return store.add_telegram_document(
        source_id=source_id,
        name="Розклад майстерні.txt",
        content=FACT.encode("utf-8"),
        text=FACT,
        owner_id=101,
        owner_name="@author",
        uploaded_at=datetime(2026, 1, 2, 10, 30, tzinfo=timezone.utc),
    )


def message() -> Message:
    return cast(Message, SimpleNamespace(
        text=QUESTION,
        chat=SimpleNamespace(id=99),
        from_user=SimpleNamespace(id=202),
        bot=Mock(),
        answer=AsyncMock(),
    ))


def selection(*document_ids: int) -> SimpleNamespace:
    return completion({
        "status": "selected", "document_ids": list(document_ids), "text": "",
    })


def answer(*source_ids: int) -> SimpleNamespace:
    return completion({
        "status": "answer", "text": FACT,
        "evidence": [
            {"source_id": identifier, "quote": FACT}
            for identifier in source_ids
        ],
    })


def assistant(create: AsyncMock) -> llm.ArchiveAssistant:
    return llm.ArchiveAssistant(make_client(create), "offline-test-model")


def test_reply_tracks_unique_cited_database_ids_not_selection_ids(
    store: Store,
) -> None:
    unrelated_id = store.upsert_drive_document(
        "unrelated", "Догляд за садом.txt", b"garden", "Поливати сад.", None
    )
    target_id = upload(store)
    assert target_id != 1
    create = AsyncMock(side_effect=[
        selection(target_id, unrelated_id), answer(2, 2),
    ])

    reply = asyncio.run(assistant(create).answer(
        QUESTION, store.list_documents()
    ))

    assert reply.source_ids == (target_id,)
    assert unrelated_id not in reply.source_ids
    assert "Догляд за садом" not in reply.text
    assert create.await_count == 2


@pytest.mark.parametrize("status", ["clarify", "not_found"])
def test_reply_without_verified_citations_has_no_database_ids(
    store: Store, status: str,
) -> None:
    upload(store)
    create = AsyncMock(return_value=completion({
        "status": status, "document_ids": [],
        "text": "Для якого місця?" if status == "clarify" else "",
    }))

    reply = asyncio.run(assistant(create).answer(
        QUESTION, store.list_documents()
    ))

    assert reply.source_ids == ()
    assert create.await_count == 1


def test_cached_answer_is_unavailable_after_deletion_without_new_api_call(
    store: Store,
) -> None:
    document_id = upload(store)
    create = AsyncMock(side_effect=[selection(document_id), answer(1)])
    service = assistant(create)

    async def scenario() -> None:
        reply = await service.answer(QUESTION, store.list_documents())
        assert reply.source_ids == (document_id,)
        cached = await service.answer(QUESTION, store.list_documents())
        assert cached == reply
        assert create.await_count == 2
        assert store.delete_telegram_document(document_id, 101)

        fresh = await service.answer(QUESTION, store.list_documents())

        assert fresh.text == llm.NOT_FOUND
        assert fresh.source_ids == ()
        assert create.await_count == 2

    asyncio.run(scenario())


def test_deletion_invalidates_cache_with_other_documents_remaining(
    store: Store,
) -> None:
    document_id = upload(store)
    store.upsert_drive_document(
        "garden", "Догляд за садом.txt", b"garden", "Поливати сад.", None
    )
    create = AsyncMock(side_effect=[
        selection(document_id), answer(1),
        completion({"status": "not_found", "document_ids": [], "text": ""}),
    ])
    service = assistant(create)

    async def scenario() -> None:
        assert (await service.answer(
            QUESTION, store.list_documents()
        )).source_ids == (document_id,)
        assert store.delete_telegram_document(document_id, 101)

        reply = await service.answer(QUESTION, store.list_documents())

        assert reply.text == llm.NOT_FOUND
        assert reply.source_ids == ()

    asyncio.run(scenario())

    assert create.await_count == 3
    latest_request = json.dumps(
        create.await_args.kwargs["messages"], ensure_ascii=False
    )
    assert "Розклад майстерні.txt" not in latest_request


def test_deleted_tombstone_is_excluded_even_if_marked_active(
    store: Store,
) -> None:
    upload(store)
    corrupted = replace(
        store.list_documents()[0],
        active=True, deleted_at="2026-01-03T12:00:00+00:00",
    )
    create = AsyncMock()

    reply = asyncio.run(assistant(create).answer(QUESTION, [corrupted]))

    assert reply.text == llm.NOT_FOUND
    assert reply.source_ids == ()
    create.assert_not_awaited()


@pytest.mark.parametrize("delete_cited", [True, False])
def test_deletion_during_api_call_checks_only_verified_cited_sources(
    store: Store, delete_cited: bool,
) -> None:
    document_id = upload(store)
    unrelated_id = upload(store, source_id="10:21")
    incoming = message()

    async def scenario() -> None:
        generating = asyncio.Event()
        release = asyncio.Event()

        async def create_response(**kwargs: object) -> SimpleNamespace:
            if not generating.is_set():
                generating.set()
                return selection(document_id)
            await release.wait()
            return answer(1)

        create = AsyncMock(side_effect=create_response)
        service = assistant(create)
        task = asyncio.create_task(text_handler(incoming, service, store, {}))
        await asyncio.wait_for(generating.wait(), timeout=5)
        # The assistant's snapshot already contains both documents, while its
        # answer still awaits the mocked provider response.
        deleted_id = document_id if delete_cited else unrelated_id
        assert store.delete_telegram_document(deleted_id, 101)
        release.set()
        await asyncio.wait_for(task, timeout=5)
        assert create.await_count == 2

    asyncio.run(scenario())

    sent_text = incoming.answer.await_args.args[0]
    if delete_cited:
        assert sent_text == ARCHIVE_CHANGED
        assert FACT not in sent_text
    else:
        assert FACT in sent_text
        assert "@author" in sent_text
    assert incoming.answer.await_count == 1


def test_queued_cached_snapshot_cannot_send_deleted_facts(
    store: Store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = upload(store)
    create = AsyncMock(side_effect=[selection(document_id), answer(1)])
    service = assistant(create)
    incoming = message()

    async def scenario() -> None:
        await service.answer(QUESTION, store.list_documents())
        snapshot_taken = asyncio.Event()
        original_answer = service.answer

        async def queued_answer(
            question: str, documents: Sequence[Document],
        ) -> llm.AssistantReply:
            snapshot_taken.set()
            return await original_answer(question, documents)

        monkeypatch.setattr(service, "answer", queued_answer)
        # A second user may have read the store before obtaining the shared
        # assistant lock. This snapshot can still hit the old answer cache.
        async with service._lock:
            task = asyncio.create_task(text_handler(
                incoming, service, store, {}
            ))
            await asyncio.wait_for(snapshot_taken.wait(), timeout=5)
            assert store.delete_telegram_document(document_id, 101)
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())

    incoming.answer.assert_awaited_once_with(ARCHIVE_CHANGED)
    assert create.await_count == 2


def test_source_validation_database_error_hides_facts_and_restores_question(
    store: Store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_id = upload(store)
    create = AsyncMock(side_effect=[selection(document_id), answer(1)])
    incoming = message()
    original_question = "Коли відкрита майстерня біля клена?"
    pending = {(99, 202): original_question}
    validate = Mock(side_effect=sqlite3.OperationalError("private path"))
    monkeypatch.setattr(store, "are_documents_active", validate)

    asyncio.run(text_handler(incoming, assistant(create), store, pending))

    incoming.answer.assert_awaited_once_with(
        "Не вдалося прочитати архів. Спробуй ще раз."
    )
    validate.assert_called_once_with((document_id,))
    assert pending == {(99, 202): original_question}
    assert create.await_count == 2
    assert FACT not in incoming.answer.await_args.args[0]
    assert "private path" not in incoming.answer.await_args.args[0]
