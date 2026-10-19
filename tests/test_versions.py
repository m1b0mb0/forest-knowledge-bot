"""Deterministic advice identity and revision ordering without API calls."""

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from bot.services.versions import (
    AdviceVersion,
    compare_versions,
    extract_version,
)


@pytest.mark.parametrize(
    ("name", "topic", "place"),
    [
        ("Членські внески рукавички Під дубом.txt",
         "членські внески", "під дубом"),
        ("Правила співжиття в рукавичці На узліссі.pdf",
         "правила співжиття", "на узліссі"),
        ("Як утеплити рукавичку Біля річки.docx",
         "як утеплити", "біля річки"),
        ('  ЧЛЕНСЬКІ  внески рукавички «Під дубом».TXT',
         "членські внески", "під дубом"),
        ("Правила співжиття у рукавичці У соснах.txt",
         "правила співжиття", "у соснах"),
    ],
)
def test_current_titles_have_exact_topic_and_place(
    name: str, topic: str, place: str,
) -> None:
    assert extract_version(name, "") == AdviceVersion(topic, place)


def test_explicit_headers_support_unstructured_filename() -> None:
    assert extract_version(
        "update.txt",
        "Тема: Порядок чергування\nМісце: Лісова галявина\n"
        "Дата: 19.10.2026\nВерсія: 2.1\nЧергування починається о 9.",
    ) == AdviceVersion(
        "порядок чергування", "лісова галявина", date(2026, 10, 19), (2, 1),
    )


def test_first_body_heading_can_supply_identity() -> None:
    assert extract_version(
        "update.txt",
        "\n# Правила співжиття в рукавичці На узліссі (v2)\n"
        "Оновлено: 2026-10-19\nНе шуміти.",
    ) == AdviceVersion(
        "правила співжиття", "на узліссі", date(2026, 10, 19), (2,),
    )


@pytest.mark.parametrize(
    "suffix",
    [" (v2)", " v2", "_v2", " (версія 2)", " - редакція: 2"],
)
def test_labeled_filename_version_is_not_part_of_place(suffix: str) -> None:
    assert extract_version(
        f"Членські внески рукавички Під дубом{suffix}.txt", "",
    ) == AdviceVersion("членські внески", "під дубом", number=(2,))


def test_multiple_filename_suffixes_supply_version_and_date() -> None:
    assert extract_version(
        "Членські внески рукавички Під дубом (v2.1) "
        "(дата 2026-10-19).txt", "",
    ) == AdviceVersion(
        "членські внески", "під дубом", date(2026, 10, 19), (2, 1),
    )


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("Членські внески.txt", "Здати до 10 числа."),
        ("notes.txt", "Тема: Членські внески"),
        ("notes.txt", "Місце: Під дубом"),
        ("Правила рукавички для гостей.txt", ""),
        ("Членські внески рукавички Під дубом та У соснах.txt", ""),
        ("Членські внески рукавички Під дубом / У соснах.txt", ""),
        ("notes.txt", "Тема: Внески\nМісце: Під дубом; У соснах"),
        ("notes.txt", "Тема: Внески\nМісце: Під дубом\nМісце: У соснах"),
        ("Членські внески рукавички Під дубом.txt", "Місце: У соснах"),
        ("Членські внески рукавички Під дубом.txt", "Тема: Утеплення"),
        ("Членські внески рукавички Під дубом.txt",
         "Членські внески рукавички У соснах"),
        ("Членські внески рукавички Під дубом (v2).txt", "Версія: 3"),
        ("Членські внески рукавички Під дубом.txt", "Дата: 2026-02-30"),
        ("Членські внески рукавички Під дубом.txt", "Дата: сьогодні"),
        ("Членські внески рукавички Під дубом (дата 2026-02-30).txt", ""),
        ("Членські внески рукавички Під дубом.txt", "Версія: два"),
        ("Членські внески рукавички Під дубом.txt",
         "Дата: 2026-10-19\nОновлено: 2026-10-20"),
    ],
)
def test_uncertain_identity_or_metadata_is_not_merged(
    name: str, text: str,
) -> None:
    assert extract_version(name, text) is None


def test_payment_dates_and_numbers_are_not_revision_metadata() -> None:
    assert extract_version(
        "Членські внески рукавички Під дубом.txt",
        "Здати 4 жолуді до 10 числа. Зустріч відбудеться 2026-10-19.",
    ) == AdviceVersion("членські внески", "під дубом")


@pytest.mark.parametrize(
    "text",
    [
        "Членські внески рукавички Під дубом та У соснах\nЗдати внесок.",
        "Членські внески рукавички Під дубом і сосною\nЗдати внесок.",
        "Членські внески рукавички Під дубом, сосною\nЗдати внесок.",
    ],
)
def test_multiple_place_heading_cannot_be_overridden_by_filename(
    text: str,
) -> None:
    assert extract_version(
        "Членські внески рукавички Під дубом.txt", text,
    ) is None


@pytest.mark.parametrize(
    "place", [
        "Під дубом та У соснах", "Під дубом і сосною", "Під дубом, сосною",
    ],
)
def test_multiple_place_filename_cannot_be_overridden_by_headers(
    place: str,
) -> None:
    assert extract_version(
        f"Членські внески рукавички {place}.txt",
        "Тема: Членські внески\nМісце: Під дубом",
    ) is None


@pytest.mark.parametrize("place", ["Дуб і сосна", "Дуб, сосна"])
def test_explicit_place_list_is_not_a_single_identity(place: str) -> None:
    assert extract_version(
        "update.txt", f"Тема: Членські внески\nМісце: {place}",
    ) is None


@pytest.mark.parametrize(
    "text", [
        "У рукавичці Під дубом платять жолудями, їх збирає Кабан.",
        "Членські внески рукавички Під дубом: жолуді, горіхи та каштани.",
    ],
)
def test_ordinary_prose_is_not_treated_as_a_multiple_place_heading(
    text: str,
) -> None:
    assert extract_version(
        "Членські внески рукавички Під дубом.txt", text,
    ) == AdviceVersion("членські внески", "під дубом")


def test_metadata_is_read_only_near_start() -> None:
    assert extract_version(
        "Членські внески рукавички Під дубом.txt",
        "\n" * 20 + "Дата: 2026-02-30\nМісце: У соснах",
    ) == AdviceVersion("членські внески", "під дубом")


def test_repeated_matching_hints_are_accepted() -> None:
    actual = extract_version(
        "Членські внески рукавички Під дубом (v1).txt",
        "Тема: ЧЛЕНСЬКІ ВНЕСКИ\nМісце: «Під дубом»\nВерсія: 1.0\n"
        "Дата: 2026-10-19\nОновлено: 2026-10-19",
    )
    assert actual is not None
    assert actual.topic == "членські внески"
    assert actual.place == "під дубом"
    assert actual.published_on == date(2026, 10, 19)
    assert actual.number in {(1,), (1, 0)}


def advice(
    *, published_on: date | None = None,
    number: tuple[int, ...] | None = None,
) -> AdviceVersion:
    return AdviceVersion("членські внески", "під дубом", published_on, number)


OLDER = datetime(2026, 10, 18, 12, tzinfo=UTC)
NEWER = datetime(2026, 10, 19, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("left", "left_time", "right", "right_time", "expected"),
    [
        (advice(), OLDER, advice(), NEWER, -1),
        (advice(), NEWER, advice(), OLDER, 1),
        (advice(), OLDER, advice(), OLDER, 0),
        (advice(number=(2,)), OLDER, advice(number=(1,)), NEWER, 1),
        (advice(number=(1, 0)), NEWER, advice(number=(1,)), OLDER, 0),
        (advice(number=(1, 10)), OLDER, advice(number=(1, 2)), NEWER, 1),
        (advice(number=(2,)), OLDER, advice(), NEWER, -1),
        (advice(published_on=date(2026, 10, 20)), OLDER,
         advice(published_on=date(2026, 10, 19)), NEWER, 1),
        (advice(published_on=date(2026, 10, 20)), OLDER,
         advice(), NEWER, 1),
        (advice(published_on=date(2026, 10, 18)), NEWER,
         advice(), OLDER, -1),
        (advice(published_on=date(2026, 10, 19)), NEWER,
         advice(published_on=date(2026, 10, 19)), OLDER, 0),
        (advice(number=(2,), published_on=date(2026, 10, 19)), OLDER,
         advice(number=(1,), published_on=date(2026, 10, 20)), NEWER, None),
        (advice(number=(1,), published_on=date(2026, 10, 19)), OLDER,
         advice(number=(1,), published_on=date(2026, 10, 20)), NEWER, None),
        (advice(number=(2,), published_on=date(2026, 10, 20)), OLDER,
         advice(number=(1,), published_on=date(2026, 10, 19)), NEWER, 1),
        (advice(number=(2,), published_on=date(2026, 10, 19)), OLDER,
         advice(number=(1,), published_on=date(2026, 10, 19)), NEWER, 1),
        (advice(), OLDER.replace(tzinfo=None), advice(), NEWER, None),
    ],
)
def test_version_order_uses_explicit_metadata_then_aware_fallback(
    left: AdviceVersion, left_time: datetime, right: AdviceVersion,
    right_time: datetime, expected: int | None,
) -> None:
    assert compare_versions(left, left_time, right, right_time) == expected
    reverse = None if expected is None else -expected
    assert compare_versions(right, right_time, left, left_time) == reverse


@pytest.mark.parametrize(
    "different", [
        AdviceVersion("утеплення", "під дубом"),
        AdviceVersion("членські внески", "у соснах"),
    ],
)
def test_different_topic_or_place_is_never_ordered(
    different: AdviceVersion,
) -> None:
    assert compare_versions(advice(), OLDER, different, NEWER) is None


def test_fallback_timestamps_compare_instants_across_timezones() -> None:
    shifted = OLDER.astimezone(timezone(timedelta(hours=3)))
    assert compare_versions(advice(), shifted, advice(), OLDER) == 0
