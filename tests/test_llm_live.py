"""Opt-in Groq checks with fictional documents, never the private archive.

Set RUN_GROQ_LIVE=1 and GROQ_API_KEY to enable these network requests.
The ordinary test suite skips this module's cases without contacting Groq.
"""

import asyncio
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from groq import AsyncGroq

from bot.config import load_llm_settings
from bot.services.llm import ArchiveAssistant, AssistantReply
from bot.storage import Document

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_GROQ_LIVE") != "1",
    reason="Real Groq requests require RUN_GROQ_LIVE=1.",
)


def documents() -> list[Document]:
    """Keep local alternatives beside general guides to exercise selection."""
    sources = [
        (
            "Членські внески рукавички Біля клена.txt",
            "Кожен мешканець щомісяця здає 17 каштанів до 8 числа. "
            "Внесок передають Білці у скриню під сходами. "
            "Замість 17 каштанів можна принести 5 шишок.",
        ),
        (
            "Членські внески рукавички У очереті.txt",
            "Кожен мешканець щомісяця здає 31 горіх до 19 числа. "
            "Внесок передають Видрі біля містка. "
            "Заміна внеску — 4 сухі гілки.",
        ),
        (
            "Лісовий календар.txt",
            "За календарем цього вигаданого лісу гриби збирають "
            "у травні та червні. У жовтні проводять свято листя.",
        ),
        (
            "Посібник з лісових прогулянок.txt",
            "Два свисти Сови означають густий туман. Почувши їх, "
            "слід повернутися додому. Один свист означає початок гри. "
            "З лісової прогулянки треба повертатися до рукавички "
            "до появи першої зірки.",
        ),
        (
            "Довідник їстівних рослин.txt",
            "Невідому ягоду не можна рвати чи їсти. Залиш її на кущі "
            "та запитай Лося, який знає місцеві рослини.",
        ),
        (
            "Збір грибів рукавички Біля клена.txt",
            "Мешканці рукавички Біля клена мають окремий місцевий "
            "розклад: збирають гриби у липні з провідником.",
        ),
        (
            "Збір грибів рукавички У очереті.txt",
            "Мешканці рукавички У очереті мають окремий місцевий "
            "розклад: збирають гриби у вересні після дощу.",
        ),
        (
            "Що робити при пожежі.txt",
            "Побачивши вогонь, поклич пожежну варту. Борсук б'є "
            "у дерев'яне калатало. Неси відро води до пожежного щита.",
        ),
        (
            "Правила співжиття в рукавичці Біля клена.txt",
            "Гостей приймають до 18 години. Взуття залишають біля "
            "порога. Спільну кухню прибирають після вечері.",
        ),
        (
            "Правила співжиття в рукавичці У очереті.txt",
            "Гостей приймають до 21 години. Мокрі плащі сушать на "
            "веранді. Спільну кухню прибирають після обіду.",
        ),
        (
            "Як попередити про мисливців.txt",
            "Побачивши мисливців, тричі постукай по порожньому "
            "стовбуру та попередь Оленя. Це сигнал небезпеки.",
        ),
    ]
    return [
        Document(
            id=index,
            source="drive",
            source_id=f"fictional-{index}",
            name=name,
            relative_path=Path("documents") / f"fictional-{index}.bin",
            text=text,
            modified_at=None,
            created_at="2026-01-01T00:00:00Z",
            imported_at="2026-01-01T00:00:00Z",
            active=True,
        )
        for index, (name, text) in enumerate(sources, start=1)
    ]


@pytest.fixture(autouse=True)
def cool_down_after_live_request() -> Iterator[None]:
    """Space opted-in cases apart; skipped cases never run this fixture."""
    yield
    time.sleep(30)


async def ask(question: str) -> AssistantReply:
    settings = load_llm_settings()
    async with AsyncGroq(
        api_key=settings.api_key, max_retries=0, timeout=45,
    ) as client:
        service = ArchiveAssistant(client, settings.model)
        return await service.answer(question, documents())


@pytest.mark.parametrize("question,facts,title,clarify", [
    pytest.param(
        "У рукавичці Біля клена можна принести шишки замість внеску?",
        ("5", "шиш"),
        "«Членські внески» рукавички «Біля клена»",
        False,
        id="contribution-substitution",
    ),
    pytest.param(
        "Чому Біля клена треба щомісяця віддавати 40 каштанів "
        "до 19 числа?",
        ("17", "8", "каштан"),
        "«Членські внески» рукавички «Біля клена»",
        False,
        id="contribution-false-premise",
    ),
    pytest.param(
        "У які місяці в лісі збирають гриби?",
        ("трав", "черв"),
        "«Лісовий календар»",
        False,
        id="general-mushroom-season",
    ),
    pytest.param(
        "Що означають два свисти Сови і що потрібно зробити?",
        ("туман", ("повер", "верта")),
        "«Посібник з лісових прогулянок»",
        False,
        id="general-owl-signal",
    ),
    pytest.param(
        "Коли треба повертатися до рукавички?",
        ("перш", "зір"),
        "«Посібник з лісових прогулянок»",
        False,
        id="general-return-home-without-location",
    ),
    pytest.param(
        "Початкове запитання: Коли треба повертатися до рукавички?\n"
        "Уточнення: Біля клена",
        ("перш", "зір"),
        "«Посібник з лісових прогулянок»",
        False,
        id="return-home-after-location-clarification",
    ),
    pytest.param(
        "Знайшов ягоду, якої не знаю. Як вчинити?",
        ("лос", ("залиш", "не ї", "не кушт", "не рв", "не спож")),
        "«Довідник їстівних рослин»",
        False,
        id="unknown-berry",
    ),
    pytest.param(
        "Скільки треба здавати щомісяця?",
        (),
        None,
        True,
        id="contribution-needs-location",
    ),
])
def test_live_answers_select_sources_and_preserve_facts(
    question: str,
    facts: tuple[str | tuple[str, ...], ...],
    title: str | None,
    clarify: bool,
) -> None:
    reply = asyncio.run(ask(question))

    assert reply.needs_clarification is clarify
    assert reply.text.strip()
    assert not re.search(
        r"(?i)(?<!\w)(?:[DS]\d+|source_id|document_ids)(?!\w)|\[\d+\]",
        reply.text,
    ), f"Internal source marker in {reply.text!r}"
    if clarify:
        assert "— з документа" not in reply.text
        return

    assert reply.text.endswith(f"\n\n— з документа {title}")
    text = reply.text.casefold()
    for fact in facts:
        alternatives = (fact,) if isinstance(fact, str) else fact
        assert any(
            re.search(rf"(?<!\d){re.escape(term)}(?!\d)", text)
            if term.isdecimal() else term.casefold() in text
            for term in alternatives
        ), f"Expected fact {fact!r} in {reply.text!r}"
