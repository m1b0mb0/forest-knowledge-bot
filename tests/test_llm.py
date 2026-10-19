"""Document selection and grounded answers without external model requests."""

import asyncio
import json
import re
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

NOT_FOUND = "Не знайшов відповіді в доступних документах."


def document(document_id: int, name: str, text: str) -> Document:
    return Document(
        id=document_id,
        source="drive",
        source_id=f"source-{document_id}",
        name=name,
        relative_path=Path("documents") / f"{document_id}.bin",
        text=text,
        modified_at="2026-10-18T12:00:00Z",
        created_at="2026-10-18T12:00:00Z",
        imported_at="2026-10-18T12:00:00Z",
        active=True,
    )


def payload(
    status: str = "answer",
    document_ids: list[int] | None = None,
    text: str = "Сума внеску — 120 грн.",
) -> dict[str, Any]:
    return {
        "status": status,
        "document_ids": [1] if document_ids is None else document_ids,
        "text": text,
    }


def selection_payload(
    *documents: Document, text: str = "",
) -> dict[str, Any]:
    return {
        "status": "selected",
        "document_ids": [item.id for item in documents],
        "text": text,
    }


def response(value: Any, finish_reason: str = "stop") -> SimpleNamespace:
    content = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30),
    )


def assistant(*responses: Any) -> tuple[llm.ArchiveAssistant, AsyncMock]:
    create = AsyncMock(side_effect=list(responses))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return llm.ArchiveAssistant(client, "test-model"), create


def messages_text(call: Any) -> str:
    return "\n".join(
        message["content"] for message in call.kwargs["messages"]
    )


def selection_catalog(call: Any) -> list[str]:
    return [
        line
        for message in call.kwargs["messages"]
        if message["role"] == "user"
        for line in message["content"].splitlines()
        if re.search(r" \[D\d+\]$", line)
    ]


@pytest.fixture(autouse=True)
def avoid_tokenizer_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent of the tokenizer's vocabulary download."""
    monkeypatch.setattr(llm, "_count_tokens", lambda text: len(text.split()))


def test_selection_gets_names_then_answer_gets_selected_full_texts() -> None:
    first_text = (
        "Вступ до правил. " * 90
        + "Не сплачувати до 15.11.2027. Внесок — 275 грн.\n"
        + "Завершення документа. " * 90
    )
    documents = [
        document(
            1, "Правила внесків рукавички Біля каштана.pdf", first_text
        ),
        document(2, "Як утеплити житло.docx", "Невибраний секретний текст."),
    ]
    service, create = assistant(
        response(selection_payload(
            documents[0], text="Не показувати цей текст відбору."
        )),
        response(payload(
            text="Внесок — 275 грн; не сплачувати до 15.11.2027."
        )),
    )
    question = "Коли і скільки платити біля каштана?"

    reply = asyncio.run(service.answer(question, documents))

    assert not reply.needs_clarification
    assert reply.text == (
        "Внесок — 275 грн; не сплачувати до 15.11.2027."
        "\n\n— з документа «Правила внесків» рукавички «Біля каштана»"
    )
    assert documents[0].name not in reply.text
    assert documents[1].name not in reply.text
    assert "Не показувати цей текст відбору." not in reply.text
    assert create.await_count == 2
    selection_call, answer_call = create.await_args_list
    selection = json.dumps(selection_call.kwargs, ensure_ascii=False)
    answer = messages_text(answer_call)
    assert all(item.name in selection for item in documents)
    assert all(
        item.name in messages_text(selection_call) for item in documents
    )
    for item in documents:
        assert selection.count(f"{item.name} [D{item.id}]") == 1
    assert "Вступ до правил." not in selection
    assert "Невибраний секретний текст." not in selection
    assert "Не сплачувати до 15.11.2027. Внесок — 275 грн." in answer
    assert answer.count("Вступ до правил.") == 90
    assert answer.count("Завершення документа.") == 90
    assert "Невибраний секретний текст." not in answer
    schema = selection_call.kwargs["response_format"]["json_schema"]["schema"]
    item_schema = schema["properties"]["document_ids"]["items"]
    assert item_schema["type"] == "integer"
    assert item_schema["enum"] == selection_payload(*documents)["document_ids"]
    assert schema["properties"]["document_ids"]["maxItems"] == 3
    for call in create.await_args_list:
        assert call.kwargs["model"] == "test-model"
        assert question in call.kwargs["messages"][-1]["content"]
        output_format = call.kwargs["response_format"]
        assert output_format["type"] == "json_schema"
        assert output_format["json_schema"]["strict"] is True


def test_catalog_is_sorted_and_excludes_inactive_or_empty_documents() -> None:
    documents = [
        document(9, "Третій документ.txt", "Третя тема."),
        replace(
            document(2, "Неактивний.txt", "Прихований текст."), active=False
        ),
        document(5, "Порожній.txt", " \n\t "),
        document(1, "Перший документ.txt", "Перша тема."),
    ]
    service, create = assistant(
        response(payload("not_found", [], "")),
    )

    reply = asyncio.run(service.answer("Питання про архів", documents))

    assert reply.text == NOT_FOUND
    assert selection_catalog(create.await_args) == [
        "Перший документ.txt [D1]", "Третій документ.txt [D9]"
    ]


def test_catalog_keys_are_sent_once_even_for_a_large_catalog() -> None:
    documents = [
        document(
            index, f"Синтетичні правила рукавички Місце {index}.txt",
            f"Повний текст документа номер {index}.",
        )
        for index in range(1, 81)
    ]
    service, create = assistant(response(payload("not_found", [], "")))

    asyncio.run(service.answer("Нерелевантне запитання", documents))

    request = json.dumps(create.await_args.kwargs, ensure_ascii=False)
    rows = [f"{item.name} [D{item.id}]" for item in documents]
    assert selection_catalog(create.await_args) == rows
    assert all(request.count(row) == 1 for row in rows)
    assert "Повний текст документа номер" not in request


def test_identical_filenames_are_resolved_by_their_document_ids() -> None:
    documents = [
        document(7, "Однакова назва.txt", "Необране старе правило."),
        document(42, "Однакова назва.txt", "Обране нове правило."),
    ]
    service, create = assistant(
        response(selection_payload(documents[1])),
        response(payload(document_ids=[42], text="Обране нове правило.")),
    )

    reply = asyncio.run(service.answer("Яке правило?", documents))

    assert reply.text == (
        "Обране нове правило.\n\n— з документа «Однакова назва»"
    )
    selection_call, answer_call = create.await_args_list
    catalog = selection_catalog(selection_call)
    assert "Однакова назва.txt [D7]" in catalog
    assert "Однакова назва.txt [D42]" in catalog
    answer = messages_text(answer_call)
    assert "[D42] Однакова назва.txt" in answer
    assert "Обране нове правило." in answer
    assert "Необране старе правило." not in answer
    schema = answer_call.kwargs["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["document_ids"]["items"] == {
        "type": "integer", "enum": [42]
    }


def test_selection_catalog_uses_normalized_filenames() -> None:
    source = document(
        12, "  Загальні  правила\n зборів.txt  ", "Обрані правила зборів."
    )
    service, create = assistant(
        response(selection_payload(source)),
        response(payload(document_ids=[12], text="Обрані правила зборів.")),
    )

    reply = asyncio.run(service.answer("Які правила зборів?", [source]))

    assert reply.text.endswith("— з документа «Загальні правила зборів»")
    assert "Загальні правила зборів.txt [D12]" in selection_catalog(
        create.await_args_list[0]
    )


@pytest.mark.parametrize("documents", [
    [],
    [document(1, "Порожній.txt", "\n \t")],
    [replace(document(1, "Вимкнений.txt", "Текст"), active=False)],
])
def test_no_usable_documents_returns_not_found_without_model_calls(
    documents: list[Document],
) -> None:
    service, create = assistant()

    reply = asyncio.run(service.answer("Які правила?", documents))

    assert reply.text == NOT_FOUND
    assert not reply.needs_clarification
    create.assert_not_awaited()


def test_selection_can_clarify_without_sending_document_bodies() -> None:
    service, create = assistant(response(payload(
        "clarify", [], "Про яку саме рукавичку ти запитуєш?"
    )))

    reply = asyncio.run(service.answer(
        "Який внесок?", [document(1, "Внески.txt", "Зміст внесків.")]
    ))

    assert reply.needs_clarification
    assert reply.text == "Про яку саме рукавичку ти запитуєш?"
    assert create.await_count == 1
    assert "Зміст внесків." not in messages_text(create.await_args)


def test_not_found_ignores_model_text_and_does_not_invent_sources() -> None:
    service, create = assistant(response(payload(
        "not_found", [], "Подивись на вигаданий сайт чи документ."
    )))

    reply = asyncio.run(service.answer(
        "Як відремонтувати космічний корабель?",
        [document(1, "Лісові правила.txt", "Не ламати дерева.")],
    ))

    assert reply.text == NOT_FOUND
    assert not reply.needs_clarification
    assert create.await_count == 1


def test_final_answer_cites_only_actual_subset_of_selected_documents() -> None:
    documents = [
        document(1, "Точна назва № 1.pdf", "Перше джерело."),
        document(2, "Друге джерело.docx", "Друге джерело."),
    ]
    service, create = assistant(
        response(selection_payload(*documents)),
        response(payload(
            document_ids=[2], text="Відповідь із другого джерела."
        )),
    )

    reply = asyncio.run(service.answer("Питання", documents))

    assert reply.text == (
        "Відповідь із другого джерела."
        "\n\n— з документа «Друге джерело»"
    )
    assert documents[1].name not in reply.text
    assert documents[0].name not in reply.text
    assert create.await_count == 2


@pytest.mark.parametrize("filename,title", [
    ("Порядок зборів.txt", "«Порядок зборів»"),
    ("Порядок зборів.PDF", "«Порядок зборів»"),
    ("Порядок зборів.DOCX", "«Порядок зборів»"),
    ("Порядок зборів.TXT", "«Порядок зборів»"),
    ("Порядок зборів.v2.pdf", "«Порядок зборів.v2»"),
    ("«Порядок зборів».pdf", "«Порядок зборів»"),
    ('"Порядок зборів".docx', "«Порядок зборів»"),
    (
        "Порядок внесків рукавички Біля клена.pdf",
        "«Порядок внесків» рукавички «Біля клена»",
    ),
    (
        "«Порядок внесків» рукавички «Біля клена».pdf",
        "«Порядок внесків» рукавички «Біля клена»",
    ),
    (
        '"Порядок внесків" рукавички "Біля клена".txt',
        "«Порядок внесків» рукавички «Біля клена»",
    ),
    (
        '«Порядок внесків» рукавички "Біля клена".DOCX',
        "«Порядок внесків» рукавички «Біля клена»",
    ),
])
def test_single_source_uses_a_natural_title_without_file_extension(
    filename: str, title: str,
) -> None:
    source = document(1, filename, "Синтетичні правила зборів.")
    service, create = assistant(
        response(selection_payload(source)),
        response(payload(text="Відповідь із джерела.")),
    )

    reply = asyncio.run(service.answer("Які правила?", [source]))

    assert reply.text == f"Відповідь із джерела.\n\n— з документа {title}"
    assert create.await_count == 2


def test_multiple_sources_use_plural_label_and_each_actual_title() -> None:
    documents = [
        document(
            1, "Порядок внесків рукавички Біля клена.pdf", "Внесок — 275 грн."
        ),
        document(2, "Розклад зборів.DOCX", "Збори відбуваються у вівторок."),
    ]
    service, create = assistant(
        response(selection_payload(*documents)),
        response(payload(document_ids=[1, 2], text="Умови зборів та внеску.")),
    )

    reply = asyncio.run(service.answer("Які умови зборів?", documents))

    assert reply.text == (
        "Умови зборів та внеску.\n\n— з документів "
        "«Порядок внесків» рукавички «Біля клена», «Розклад зборів»"
    )
    assert create.await_count == 2


@pytest.mark.parametrize("status,text,clarification", [
    ("not_found", "Неіснуюча довідка.pdf", False),
    ("clarify", "Уточни, будь ласка, рік внеску.", True),
])
def test_final_response_can_decline_or_clarify_without_source_labels(
    status: str, text: str, clarification: bool,
) -> None:
    source = document(1, "Дійсна назва.txt", "Дійсний текст.")
    service, create = assistant(
        response(selection_payload(source)),
        response(payload(status, [], text)),
    )

    reply = asyncio.run(service.answer("Питання", [source]))

    assert reply.needs_clarification is clarification
    assert reply.text == (text if clarification else NOT_FOUND)
    assert "Дійсна назва.txt" not in reply.text
    assert create.await_count == 2


@pytest.mark.parametrize("invalid", [
    "not JSON",
    "[]",
    "null",
    {"status": "unknown", "document_ids": [], "text": "Текст"},
    {"status": "answer", "document_ids": [1], "text": ""},
    {"status": "selected", "document_ids": "1", "text": ""},
    {"status": "selected", "document_ids": ["1"], "text": ""},
    {"status": "selected", "document_ids": ["Правила.txt [D1]"], "text": ""},
    {"status": "selected", "document_ids": [True], "text": ""},
    {"status": "selected", "document_ids": [1.0], "text": ""},
    {"status": "selected", "document_ids": [1, 1], "text": ""},
    {"status": "selected", "document_ids": [999], "text": ""},
    {"status": "selected", "document_ids": [], "text": ""},
    {"status": "selected", "document_ids": [1], "text": 42},
    {"status": "selected", "document_ids": [1]},
    {"status": "clarify", "document_ids": [], "text": " \n "},
])
def test_invalid_selection_never_reaches_answer_request(invalid: Any) -> None:
    service, create = assistant(response(invalid))

    with pytest.raises(llm.AssistantError):
        asyncio.run(service.answer(
            "Питання", [document(1, "Правила.txt", "Правила архіву.")]
        ))

    assert create.await_count == 1


def test_selector_cannot_choose_more_than_three_documents() -> None:
    documents = [
        document(index, f"Правила {index}.txt", f"Текст {index}.")
        for index in range(1, 5)
    ]
    service, create = assistant(response(selection_payload(*documents)))

    with pytest.raises(llm.AssistantError):
        asyncio.run(service.answer("Питання", documents))

    assert create.await_count == 1


@pytest.mark.parametrize("invalid", [
    "not JSON",
    payload(text=" \n "),
    payload(document_ids=[True]),
    payload(document_ids=[1.0]),
    payload(document_ids=[1, 1]),
    payload(document_ids=[2]),
    payload(document_ids=[999]),
    payload(document_ids=[]),
    {
        "status": "answer", "document_ids": ["Обраний.txt [D1]"],
        "text": "Текст",
    },
    {"status": "answer", "document_ids": ["1"], "text": "Текст"},
    {"status": "selected", "document_ids": [1], "text": "Текст"},
])
def test_invalid_answer_cannot_claim_an_unselected_or_invalid_source(
    invalid: Any,
) -> None:
    documents = [
        document(1, "Обраний.txt", "Відомий факт."),
        document(2, "Необраний.txt", "Інший факт."),
    ]
    service, create = assistant(
        response(selection_payload(documents[0])), response(invalid)
    )

    with pytest.raises(llm.AssistantError):
        asyncio.run(service.answer("Питання", documents))

    assert create.await_count == 2


@pytest.mark.parametrize("phase", ["selection", "answer"])
def test_truncated_model_response_is_rejected_even_when_json_is_valid(
    phase: str,
) -> None:
    source = document(1, "Правила.txt", "Архівний текст.")
    responses = [response(
        selection_payload(source) if phase == "selection" else payload(),
        finish_reason="length",
    )]
    if phase == "answer":
        responses.insert(0, response(selection_payload(source)))
    service, create = assistant(*responses)

    with pytest.raises(llm.AssistantError):
        asyncio.run(service.answer("Питання", [source]))

    assert create.await_count == len(responses)


def test_catalog_over_budget_requests_a_narrower_question_without_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_request_tokens", lambda *args: 10_000)
    service, create = assistant()

    reply = asyncio.run(service.answer(
        "Питання", [document(1, "Правила.txt", "Архівний текст.")]
    ))

    assert reply.needs_clarification
    assert reply.text.strip()
    create.assert_not_awaited()


def test_selected_texts_over_budget_are_not_silently_dropped_or_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimates = iter([100, 10_000])
    monkeypatch.setattr(llm, "_request_tokens", lambda *args: next(estimates))
    documents = [
        document(1, "Перше джерело.txt", "Перший повний текст."),
        document(2, "Друге джерело.txt", "Другий повний текст."),
    ]
    service, create = assistant(response(selection_payload(*documents)))

    reply = asyncio.run(service.answer("Питання", documents))

    assert reply.needs_clarification
    assert reply.text.strip()
    assert create.await_count == 1


@pytest.mark.parametrize("phase", ["selection", "answer"])
@pytest.mark.parametrize("error_kind", ["connection", "timeout"])
def test_api_failures_have_safe_messages_and_suppress_sensitive_traceback(
    phase: str, error_kind: str,
) -> None:
    secret = "private-test-credential-not-a-real-key"
    request = httpx.Request(
        "POST", f"https://api.example.invalid/chat?api_key={secret}"
    )
    error = (
        groq.APIConnectionError(
            message=f"Connection failed: {secret}", request=request
        )
        if error_kind == "connection" else TimeoutError(secret)
    )
    responses = [error]
    if phase == "answer":
        responses.insert(0, response(selection_payload(
            document(1, "Правила.txt", "Текст правил.")
        )))
    service, create = assistant(*responses)

    with pytest.raises(llm.AssistantError) as caught:
        asyncio.run(service.answer(
            "Питання", [document(1, "Правила.txt", "Текст правил.")]
        ))

    visible_traceback = "".join(traceback.format_exception(caught.value))
    assert secret not in str(caught.value)
    assert secret not in visible_traceback
    assert "api.example.invalid" not in visible_traceback
    assert str(caught.value).strip()
    assert create.await_count == len(responses)


def test_questions_do_not_resend_previous_messages() -> None:
    service, create = assistant(
        response(payload("not_found", [], "")),
        response(payload("not_found", [], "")),
    )
    documents = [document(1, "Правила.txt", "Архівні правила.")]

    asyncio.run(service.answer("Унікальне попереднє питання", documents))
    asyncio.run(service.answer("Нове незалежне питання", documents))

    assert create.await_count == 2
    second_request = messages_text(create.await_args_list[1])
    assert "Нове незалежне питання" in second_request
    assert "Унікальне попереднє питання" not in second_request


def test_rate_limit_is_reported_without_exposing_provider_details() -> None:
    secret = "not-a-real-key"
    request = httpx.Request(
        "POST", f"https://api.example.invalid/?key={secret}"
    )
    error = groq.RateLimitError(
        f"Provider detail: {secret}",
        response=httpx.Response(429, request=request),
        body=None,
    )
    service, create = assistant(error)

    with pytest.raises(llm.AssistantError) as caught:
        asyncio.run(service.answer(
            "Питання", [document(1, "Правила.txt", "Текст правил.")]
        ))

    assert "Groq" in str(caught.value)
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert create.await_count == 1
