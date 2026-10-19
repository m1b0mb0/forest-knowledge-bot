"""Bounded retries, answer reuse and evidence checks without API requests."""

import asyncio
import json
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import groq
import httpx
import pytest

from bot.services import llm
from bot.storage import Document


def document(text: str = "Внесок — 17 каштанів.") -> Document:
    return Document(
        id=42,
        source="drive",
        source_id="fictitious-source",
        name="Умови внеску.txt",
        relative_path=Path("documents/42.txt"),
        text=text,
        modified_at="2026-10-18T12:00:00Z",
        created_at="2026-10-18T12:00:00Z",
        imported_at="2026-10-18T12:00:00Z",
        active=True,
    )


def selected(*ids: int) -> dict[str, Any]:
    return {"status": "selected", "document_ids": list(ids), "text": ""}


def answered(
    text: str = "Внесок — 17 каштанів.",
    quote: str = "Внесок — 17 каштанів.",
    source_id: int = 1,
) -> dict[str, Any]:
    return {
        "status": "answer",
        "evidence": [{"source_id": source_id, "quote": quote}],
        "text": text,
    }


def response(value: Any) -> SimpleNamespace:
    content = value if isinstance(value, str) else json.dumps(value)
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30),
    )


def assistant(*values: Any) -> tuple[llm.ArchiveAssistant, AsyncMock]:
    create = AsyncMock(side_effect=list(values))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    service = llm.ArchiveAssistant(client, "test-model")
    return service, create


def rate_limit(
    retry_after: str | None = None, *, daily: bool = False,
) -> groq.RateLimitError:
    headers = {} if retry_after is None else {"retry-after": retry_after}
    request = httpx.Request(
        "POST", "https://example.invalid/?key=fictitious-secret",
    )
    body = {
        "error": {
            "type": "tokens",
            "code": "rate_limit_exceeded",
            "message": (
                "Rate limit reached for tokens per day (TPD)."
                if daily else "Rate limit reached for tokens per minute."
            ),
        },
    }
    return groq.RateLimitError(
        "Provider detail fictitious-secret",
        response=httpx.Response(429, headers=headers, request=request),
        body=body,
    )


@pytest.fixture(autouse=True)
def offline_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_count_tokens", lambda text: len(text.split()))


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(now=100.0, sleeps=[])

    async def sleep(seconds: float) -> None:
        state.sleeps.append(seconds)
        state.now += seconds

    monkeypatch.setattr(llm.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(llm.asyncio, "sleep", sleep)
    return state


def test_invalid_selection_retries_without_repeating_catalog() -> None:
    service, create = assistant(
        response("not JSON"), response(selected(42)), response(answered()),
    )

    reply = asyncio.run(service.answer("Який внесок?", [document()]))

    assert "17 каштанів" in reply.text
    assert create.await_count == 3
    first, repair, final = create.await_args_list
    original = first.kwargs["messages"]
    retried = repair.kwargs["messages"]
    assert retried[:-1] == original[:-1]
    assert len(retried) == len(original)
    assert original[-1]["content"] in retried[-1]["content"]
    assert retried[-1]["content"] != original[-1]["content"]
    assert "[S1]" in final.kwargs["messages"][1]["content"]


def test_invalid_quote_retries_only_answer_phase() -> None:
    service, create = assistant(
        response(selected(42)),
        response(answered(quote="Внесок — 99 горіхів.")),
        response(answered()),
    )

    reply = asyncio.run(service.answer("Який внесок?", [document()]))

    assert "17 каштанів" in reply.text
    assert create.await_count == 3
    first, repair = create.await_args_list[1:]
    assert repair.kwargs["messages"][:-1] == first.kwargs["messages"][:-1]
    assert len(repair.kwargs["messages"]) == len(first.kwargs["messages"])
    assert "Внесок — 99 горіхів." not in json.dumps(
        repair.kwargs["messages"], ensure_ascii=False,
    )


def test_one_repair_budget_is_shared_by_both_phases() -> None:
    service, create = assistant(
        response("invalid"), response(selected(42)), response("invalid"),
    )

    with pytest.raises(llm.AssistantError) as caught:
        asyncio.run(service.answer("Який внесок?", [document()]))

    assert str(caught.value) == llm.INVALID_REPLY
    assert create.await_count == 3


@pytest.mark.parametrize("quote", [
    "Внесок — 18 каштанів.",
    "Внесок — 17 горіхів.",
    "Внесок — каштанів.",
])
def test_repeated_wrong_quote_fails_after_one_repair(quote: str) -> None:
    invalid = response(answered(quote=quote))
    service, create = assistant(response(selected(42)), invalid, invalid)

    with pytest.raises(llm.AssistantError) as caught:
        asyncio.run(service.answer("Який внесок?", [document()]))

    assert str(caught.value) == llm.INVALID_REPLY
    assert quote not in "".join(traceback.format_exception(caught.value))
    assert create.await_count == 3


@pytest.mark.parametrize("source_text,quote", [
    ("Не їж ягоду.", "НЕ ЇЖ ЯГОДУ."),
    ("Не їж ягоду.", "Не і\u0308ж ягоду."),
    ("П'ять каштанів — внесок.", "П’ять каштанів - внесок."),
    ("Пʼять каштанів – внесок.", "П'ять каштанів — внесок."),
    ("Норма — １７ каштанів.", "Норма — 17 каштанів."),
])
def test_equivalent_quote_typography_does_not_require_an_api_retry(
    source_text: str, quote: str,
) -> None:
    service, create = assistant(
        response(selected(42)), response(answered(quote=quote)),
    )

    reply = asyncio.run(service.answer("Яка норма?", [document(source_text)]))

    assert reply.text.endswith("— з документа «Умови внеску»")
    assert create.await_count == 2


def test_quote_normalization_preserves_a_negation() -> None:
    invalid = response(answered(quote="Їж ягоду."))
    service, create = assistant(response(selected(42)), invalid, invalid)

    with pytest.raises(llm.AssistantError):
        asyncio.run(service.answer("Що робити з ягодою?", [document(
            "Не їж ягоду.",
        )]))

    assert create.await_count == 3


def test_repeated_identical_evidence_displays_a_source_once() -> None:
    value = answered()
    value["evidence"] *= 2
    service, create = assistant(response(selected(42)), response(value))

    reply = asyncio.run(service.answer("Який внесок?", [document()]))

    assert reply.text.count("«Умови внеску»") == 1
    assert create.await_count == 2


@pytest.mark.parametrize("phase", ["selection", "answer"])
@pytest.mark.parametrize("delay", ["2", "5"])
def test_short_rate_limit_retries_only_the_failed_phase(
    phase: str, delay: str, clock: SimpleNamespace,
) -> None:
    values = [rate_limit(delay), response(selected(42)), response(answered())]
    if phase == "answer":
        values = [
            response(selected(42)), rate_limit(delay), response(answered()),
        ]
    service, create = assistant(*values)

    reply = asyncio.run(service.answer("Який внесок?", [document()]))

    assert "17 каштанів" in reply.text
    assert create.await_count == 3
    assert sum(clock.sleeps) == pytest.approx(float(delay))
    if phase == "answer":
        original, retried = create.await_args_list[1:]
        assert retried.kwargs == original.kwargs


def test_rate_limit_and_invalid_reply_share_one_retry(
    clock: SimpleNamespace,
) -> None:
    service, create = assistant(
        rate_limit("1"), response(selected(42)), response("not JSON"),
    )

    with pytest.raises(llm.AssistantError) as caught:
        asyncio.run(service.answer("Який внесок?", [document()]))

    assert str(caught.value) == llm.INVALID_REPLY
    assert create.await_count == 3


def test_invalid_reply_leaves_no_retry_for_a_later_rate_limit(
    clock: SimpleNamespace,
) -> None:
    service, create = assistant(
        response("not JSON"), response(selected(42)), rate_limit("2"),
    )

    with pytest.raises(llm.AssistantError):
        asyncio.run(service.answer("Який внесок?", [document()]))

    assert create.await_count == 3
    assert not clock.sleeps


@pytest.mark.parametrize("retry_after", [None, "5.1", "120", "invalid"])
def test_long_or_unknown_rate_limits_block_later_requests_without_retry(
    retry_after: str | None, clock: SimpleNamespace,
) -> None:
    service, create = assistant(rate_limit(retry_after))

    for question in ("Який внесок?", "Коли платити?"):
        with pytest.raises(llm.AssistantError) as caught:
            asyncio.run(service.answer(question, [document()]))
        assert "Groq" in str(caught.value)
        assert "fictitious-secret" not in "".join(
            traceback.format_exception(caught.value),
        )

    assert create.await_count == 1
    assert not clock.sleeps


def test_retry_exhaustion_does_not_make_the_next_question_wait(
    clock: SimpleNamespace,
) -> None:
    service, create = assistant(rate_limit("2"), rate_limit("5"))

    for question in ("Який внесок?", "Коли платити?"):
        with pytest.raises(llm.AssistantError):
            asyncio.run(service.answer(question, [document()]))

    assert create.await_count == 2
    assert clock.sleeps == [2.0]


def test_long_cooldown_reports_remaining_seconds_without_sleeping(
    clock: SimpleNamespace,
) -> None:
    service, create = assistant(rate_limit("120"))

    with pytest.raises(llm.AssistantError) as first:
        asyncio.run(service.answer("Який внесок?", [document()]))
    clock.now += 10
    with pytest.raises(llm.AssistantError) as second:
        asyncio.run(service.answer("Коли платити?", [document()]))

    assert "120" in str(first.value)
    assert "110" in str(second.value)
    assert create.await_count == 1
    assert not clock.sleeps


def test_daily_limit_is_not_retried_even_with_a_short_retry_header(
    clock: SimpleNamespace,
) -> None:
    service, create = assistant(rate_limit("5", daily=True))

    for _ in range(2):
        with pytest.raises(llm.AssistantError) as caught:
            asyncio.run(service.answer("Який внесок?", [document()]))
        assert "добов" in str(caught.value).casefold()

    assert create.await_count == 1
    assert not clock.sleeps


def test_request_can_resume_after_rate_limit_cooldown(
    clock: SimpleNamespace,
) -> None:
    service, create = assistant(
        rate_limit("120"), response(selected(42)), response(answered()),
    )
    with pytest.raises(llm.AssistantError):
        asyncio.run(service.answer("Який внесок?", [document()]))
    clock.now += 121

    reply = asyncio.run(service.answer("Який внесок?", [document()]))

    assert "17 каштанів" in reply.text
    assert create.await_count == 3


def test_successful_answer_is_reused_for_case_and_whitespace() -> None:
    service, create = assistant(response(selected(42)), response(answered()))

    first = asyncio.run(service.answer("Який внесок?", [document()]))
    second = asyncio.run(service.answer("  ЯКИЙ   внесок?  ", [document()]))

    assert second == first
    assert create.await_count == 2


def test_answer_cache_expires_after_five_minutes(
    clock: SimpleNamespace,
) -> None:
    service, create = assistant(
        response(selected(42)), response(answered()),
        response(selected(42)), response(answered()),
    )
    asyncio.run(service.answer("Який внесок?", [document()]))
    clock.now += 301

    asyncio.run(service.answer("Який внесок?", [document()]))

    assert create.await_count == 4


@pytest.mark.parametrize("change", ["id", "name", "text", "active"])
def test_archive_change_invalidates_cached_answers(change: str) -> None:
    original = document()
    changes = {
        "id": 99,
        "name": "Нові умови внеску.txt",
        "text": original.text + " Додаткова умова.",
        "active": False,
    }
    updated = replace(original, **{change: changes[change]})
    next_id = updated.id if updated.active else 7
    fallback = replace(original, id=7, name="Довідник.txt")
    values = [updated] if updated.active else [updated, fallback]
    service, create = assistant(
        response(selected(42)), response(answered()),
        response(selected(next_id)), response(answered()),
    )
    asyncio.run(service.answer("Який внесок?", [original]))

    asyncio.run(service.answer("Який внесок?", values))

    assert create.await_count == 4


@pytest.mark.parametrize("status", ["clarify", "not_found"])
def test_non_answers_are_not_cached(status: str) -> None:
    value = {"status": status, "document_ids": [], "text": "Яке місце?"}
    service, create = assistant(response(value), response(value))

    for _ in range(2):
        asyncio.run(service.answer("Який внесок?", [document()]))

    assert create.await_count == 2


def test_failed_answer_does_not_poison_the_answer_cache() -> None:
    service, create = assistant(
        response(selected(42)), response("not JSON"), response("not JSON"),
        response(selected(42)), response(answered()),
    )
    with pytest.raises(llm.AssistantError):
        asyncio.run(service.answer("Який внесок?", [document()]))

    reply = asyncio.run(service.answer("Який внесок?", [document()]))

    assert "17 каштанів" in reply.text
    assert create.await_count == 5


def test_answer_cache_evicts_the_least_recently_used_entry() -> None:
    values = [
        value
        for _ in range(66)
        for value in (response(selected(42)), response(answered()))
    ]
    service, create = assistant(*values)

    async def exercise_cache() -> None:
        for index in range(64):
            await service.answer(f"Питання {index}", [document()])
        # Touch the oldest entry before inserting the 65th distinct question.
        await service.answer("Питання 0", [document()])
        await service.answer("Питання 64", [document()])
        await service.answer("Питання 0", [document()])
        assert create.await_count == 130
        await service.answer("Питання 1", [document()])

    asyncio.run(exercise_cache())

    assert create.await_count == 132


def test_distinct_questions_do_not_wait_for_a_local_token_window(
    clock: SimpleNamespace,
) -> None:
    values = [
        response(selected(42)), response(answered()),
        response(selected(42)), response(answered()),
    ]
    for value in values:
        value.usage.prompt_tokens = 2500
        value.usage.completion_tokens = 500
    service, create = assistant(*values)

    async def ask_twice() -> tuple[llm.AssistantReply, llm.AssistantReply]:
        first = await service.answer("Який внесок?", [document()])
        second = await service.answer("Скільки треба здати?", [document()])
        return first, second

    replies = asyncio.run(ask_twice())

    assert all("17 каштанів" in reply.text for reply in replies)
    assert create.await_count == 4
    assert not clock.sleeps
    assert clock.now == 100.0


def test_unique_general_calendar_is_kept_within_three_candidates() -> None:
    first = replace(document("Перший довідковий текст."), id=10)
    second = replace(document("Другий довідковий текст."), id=20)
    dropped = replace(document("Відкинутий четвертий текст."), id=30)
    calendar = replace(
        document("Ягоди збирають у травні та червні."), id=40,
        name="Календар лісових спостережень.txt",
    )
    service, create = assistant(
        response(selected(10, 20, 30)),
        response(answered(
            text=calendar.text, quote=calendar.text, source_id=3,
        )),
    )

    reply = asyncio.run(service.answer(
        "У яких саме місяцях збирають ягоди?",
        [calendar, dropped, second, first],
    ))

    assert create.await_count == 2
    contents = create.await_args.kwargs["messages"][1]["content"]
    assert "[S1]" in contents and first.text in contents
    assert "[S2]" in contents and second.text in contents
    assert "[S3] Календар лісових спостережень.txt" in contents
    assert calendar.text in contents
    assert dropped.text not in contents
    assert "[S4]" not in contents
    assert reply.text == (
        f"{calendar.text}\n\n— з документа «Календар лісових спостережень»"
    )


@pytest.mark.parametrize("selection", [
    {"status": "not_found", "document_ids": [], "text": ""},
    {"status": "clarify", "document_ids": [], "text": "Який ліс?"},
    selected(42),
])
def test_unique_calendar_is_read_once_even_when_selector_declines(
    selection: dict[str, Any],
) -> None:
    calendar = replace(
        document("Ягоди збирають у травні та червні."),
        name="Календар спостережень.pdf",
    )
    service, create = assistant(
        response(selection),
        response(answered(text=calendar.text, quote=calendar.text)),
    )

    reply = asyncio.run(service.answer(
        "Які місяці підходять для збирання ягід?", [calendar],
    ))

    assert not reply.needs_clarification
    assert create.await_count == 2
    contents = create.await_args.kwargs["messages"][1]["content"]
    assert contents.count(calendar.text) == 1
    assert "[S1] Календар спостережень.pdf" in contents
    assert "[S2]" not in contents
    assert reply.text.endswith("— з документа «Календар спостережень»")


@pytest.mark.parametrize("scenario", ["ambiguous", "singular", "local"])
def test_calendar_guard_does_not_override_other_questions(
    scenario: str,
) -> None:
    calendar = replace(document(), name="Календар спостережень.txt")
    documents = [calendar]
    question = "У які місяці збирають ягоди?"
    if scenario == "ambiguous":
        documents.append(replace(
            calendar, id=91, name="Календар збору врожаю.txt",
        ))
    elif scenario == "singular":
        question = "Який внесок на місяць?"
    else:
        documents = [replace(
            calendar, name="Календар рукавички Біля клена.txt",
        )]
    service, create = assistant(response({
        "status": "not_found", "document_ids": [], "text": "",
    }))

    reply = asyncio.run(service.answer(question, documents))

    assert reply.text == llm.NOT_FOUND
    assert create.await_count == 1


def test_month_duration_does_not_override_place_clarification() -> None:
    calendar = replace(
        document("Ягоди збирають у травні та червні."),
        name="Календар спостережень.txt",
    )
    contributions = [
        replace(
            document(), id=51,
            name="Членські внески рукавички Біля клена.txt",
        ),
        replace(
            document(), id=52,
            name="Членські внески рукавички У очереті.txt",
        ),
    ]
    clarification = "Для якої рукавички тебе цікавлять внески?"
    service, create = assistant(response({
        "status": "clarify", "document_ids": [], "text": clarification,
    }))

    reply = asyncio.run(service.answer(
        "За скільки місяців сплачувати внесок?",
        [calendar, *contributions],
    ))

    assert reply.needs_clarification
    assert reply.text == clarification
    assert create.await_count == 1
    request = create.await_args.kwargs["messages"]
    assert all(calendar.text not in message["content"] for message in request)

