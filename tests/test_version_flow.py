"""Version changes exclude stale answers without contacting external APIs."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from aiogram.types import Message

from bot.handlers import text_handler
from bot.services import llm
from bot.storage import Store

OAK_NAME = "Членські внески рукавички Під дубом.txt"
PINE_NAME = "Членські внески рукавички У соснах.txt"
OLD_FACT = "Внесок становить 3 каштани щомісяця."
NEW_FACT = "Внесок становить 7 каштанів щомісяця."
PINE_FACT = "Внесок становить 2 шишки щомісяця."
QUESTION = "Який внесок у рукавичці Під дубом?"
ARCHIVE_CHANGED = (
    "Архів змінився під час підготовки відповіді. Повтори запитання."
)


@pytest.fixture(autouse=True)
def offline_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid tokenizer downloads and Telegram typing requests."""
    monkeypatch.setattr(llm, "_count_tokens", lambda value: len(value.split()))
    context = MagicMock()
    context.__aenter__ = AsyncMock()
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(
        "bot.handlers.ChatActionSender.typing", Mock(return_value=context)
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    instance = Store(tmp_path)
    instance.initialize()
    return instance


def add_old_advice(store: Store) -> int:
    text = f"Дата: 2026-01-01\n{OLD_FACT}"
    return store.upsert_drive_document(
        "old-oak", OAK_NAME, text.encode("utf-8"), text,
        "2026-01-01T10:00:00+00:00",
    )


def add_new_advice(store: Store, name: str = OAK_NAME) -> int:
    text = f"Дата: 2026-02-01\n{NEW_FACT}"
    return store.add_telegram_document(
        source_id="10:21", name=name,
        content=text.encode("utf-8"), text=text,
        owner_id=701, owner_name="@new_author",
        uploaded_at=datetime(2026, 2, 2, 12, 0, tzinfo=timezone.utc),
    )


def completion(payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=json.dumps(payload)),
        )],
        usage=None,
    )


def selection(document_id: int) -> SimpleNamespace:
    return completion({
        "status": "selected", "document_ids": [document_id], "text": "",
    })


def answer(fact: str) -> SimpleNamespace:
    return completion({
        "status": "answer", "text": fact,
        "evidence": [{"source_id": 1, "quote": fact}],
    })


def assistant(create: AsyncMock) -> llm.ArchiveAssistant:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return llm.ArchiveAssistant(client, "offline-test-model")


def test_new_advice_replaces_cached_answer_and_excludes_previous_source(
    store: Store,
) -> None:
    old_id = add_old_advice(store)
    create = AsyncMock(side_effect=[selection(old_id), answer(OLD_FACT)])
    service = assistant(create)

    async def scenario() -> None:
        previous = await service.answer(QUESTION, store.list_documents())
        assert previous.source_ids == (old_id,)
        assert (await service.answer(
            QUESTION, store.list_documents()
        )) == previous
        assert create.await_count == 2

        new_id = add_new_advice(store)
        create.side_effect = [selection(new_id), answer(NEW_FACT)]
        current = await service.answer(QUESTION, store.list_documents())

        assert current.source_ids == (new_id,)
        assert NEW_FACT in current.text and OLD_FACT not in current.text
        assert "@new_author" in current.text
        assert "02.02.2026 12:00 UTC" in current.text
        assert "«Членські внески» рукавички «Під дубом»" in current.text
        assert create.await_count == 4
        catalog = create.await_args_list[2].kwargs["messages"][1]["content"]
        body = create.await_args_list[3].kwargs["messages"][1]["content"]
        assert f"[D{new_id}]" in catalog
        assert f"[D{old_id}]" not in catalog
        assert NEW_FACT in body and OLD_FACT not in body

    asyncio.run(scenario())


def test_deleted_latest_advice_is_not_restored_by_replay_or_restart(
    store: Store,
) -> None:
    old_id = add_old_advice(store)
    latest_id = add_new_advice(store)
    assert store.delete_telegram_document(latest_id, 701)

    reopened = Store(store.data_dir)
    reopened.initialize()
    assert add_new_advice(reopened) == latest_id
    assert add_old_advice(reopened) == old_id
    reopened.initialize()
    documents = {item.id: item for item in reopened.list_documents()}
    assert not documents[old_id].active
    assert documents[old_id].superseded_by == latest_id
    assert not documents[latest_id].active
    assert documents[latest_id].deleted_at is not None
    create = AsyncMock()

    reply = asyncio.run(assistant(create).answer(
        QUESTION, reopened.list_documents()
    ))

    assert reply.text == llm.NOT_FOUND
    assert reply.source_ids == ()
    create.assert_not_awaited()


def test_new_oak_advice_leaves_other_place_available_for_answers(
    store: Store,
) -> None:
    old_id = add_old_advice(store)
    pine_text = f"Дата: 2026-01-01\n{PINE_FACT}"
    pine_id = store.upsert_drive_document(
        "pine", PINE_NAME, pine_text.encode("utf-8"), pine_text,
        "2026-01-01T10:00:00+00:00",
    )
    latest_id = add_new_advice(store)
    create = AsyncMock(side_effect=[selection(pine_id), answer(PINE_FACT)])

    reply = asyncio.run(assistant(create).answer(
        "Який внесок у рукавичці У соснах?", store.list_documents()
    ))

    assert reply.source_ids == (pine_id,)
    assert PINE_FACT in reply.text
    assert "«У соснах»" in reply.text
    assert "@new_author" not in reply.text
    assert NEW_FACT not in reply.text and OLD_FACT not in reply.text
    catalog = create.await_args_list[0].kwargs["messages"][1]["content"]
    assert f"[D{pine_id}]" in catalog and f"[D{latest_id}]" in catalog
    assert f"[D{old_id}]" not in catalog
    assert create.await_count == 2


def test_supersession_during_generation_blocks_stale_facts_in_telegram(
    store: Store,
) -> None:
    old_id = add_old_advice(store)
    incoming = cast(Message, SimpleNamespace(
        text=QUESTION,
        chat=SimpleNamespace(id=99),
        from_user=SimpleNamespace(id=202),
        bot=Mock(), answer=AsyncMock(),
    ))

    async def scenario() -> None:
        generating = asyncio.Event()
        release = asyncio.Event()

        async def create_response(**kwargs: object) -> SimpleNamespace:
            if not generating.is_set():
                generating.set()
                return selection(old_id)
            await release.wait()
            return answer(OLD_FACT)

        create = AsyncMock(side_effect=create_response)
        task = asyncio.create_task(text_handler(
            incoming, assistant(create), store, {}
        ))
        await asyncio.wait_for(generating.wait(), timeout=5)
        add_new_advice(store)
        release.set()
        await asyncio.wait_for(task, timeout=5)
        assert create.await_count == 2

    asyncio.run(scenario())

    incoming.answer.assert_awaited_once_with(ARCHIVE_CHANGED)
    assert OLD_FACT not in incoming.answer.await_args.args[0]


def test_uncertain_identity_keeps_both_sources_for_clarification(
    store: Store,
) -> None:
    old_text = f"Дата: 2026-01-01\n{OLD_FACT}"
    old_id = store.upsert_drive_document(
        "uncertain", "Оновлені правила.txt",
        old_text.encode("utf-8"), old_text,
        "2026-01-01T10:00:00+00:00",
    )
    new_id = add_new_advice(store, name="Оновлені правила.txt")
    clarification = "Якої саме поради стосується запитання?"
    create = AsyncMock(return_value=completion({
        "status": "clarify", "document_ids": [], "text": clarification,
    }))

    reply = asyncio.run(assistant(create).answer(
        "Який внесок?", store.list_documents()
    ))

    assert reply.text == clarification
    assert reply.needs_clarification and reply.source_ids == ()
    assert all(item.active for item in store.list_documents())
    catalog = create.await_args.kwargs["messages"][1]["content"]
    assert f"[D{old_id}]" in catalog and f"[D{new_id}]" in catalog
    assert create.await_count == 1
