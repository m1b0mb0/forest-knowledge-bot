"""Conservatively identify and order versions of one local advice document."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, time

_HEADER = re.compile(
    r"^\s*(?:#+\s*)?(тема|місце|дата(?:\s+версії)?|оновлено|версія)"
    r"\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)
_SUFFIX = re.compile(
    r"(?:^|[\s_\-–—,;]+|(?=[(\[]))[(\[]?\s*"
    r"(?P<label>версія|редакція|v|дата|оновлено)\s*[:=]?\s*"
    r"(?P<value>[^\s()\[\]]+)\s*[)\]]?\s*$",
    re.IGNORECASE,
)
_HUT = re.compile(r"\bрукавич(?:ки|ці|ку|ка)\b", re.IGNORECASE)
_PREPOSITIONS = {"під", "у", "в", "на", "біля"}


@dataclass(frozen=True)
class AdviceVersion:
    """An exact topic/place identity and optional explicit revision hints."""

    topic: str
    place: str
    published_on: date | None = None
    number: tuple[int, ...] | None = None


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", value))


def _multiple_places(value: str) -> bool:
    words = _normalize(value).split()
    return bool(
        re.search(r"[/;,\n]", value)
        or {"та", "або", "й", "і"}.intersection(words)
        or sum(word in _PREPOSITIONS for word in words) > 1
    )


def _place(value: str, *, from_title: bool = False) -> str | None:
    normalized = _normalize(value)
    words = normalized.split()
    if (
        not words
        or len(words) > 8
        or re.search(r"\d", value)
        or _multiple_places(value)
    ):
        return None
    if from_title and (
        len(words) < 2 or words[0] not in _PREPOSITIONS
    ):
        return None
    return normalized


def _identity(title: str) -> tuple[str, str] | None:
    markers = list(_HUT.finditer(title))
    if not markers:
        return None
    if len(markers) > 1:
        raise ValueError("Several hut titles do not identify one place.")
    marker = markers[0]
    topic = _normalize(title[:marker.start()])
    if topic.split()[-1:] in (["в"], ["у"]):
        topic = topic.rsplit(" ", 1)[0] if " " in topic else ""
    if not topic:
        return None
    scope = title[marker.end():]
    if _multiple_places(scope):
        raise ValueError("A hut title names an ambiguous place.")
    place = _place(scope, from_title=True)
    if not place:
        return None
    return topic, place


def _date(value: str) -> date:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return date.fromisoformat(value)
    if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", value):
        return datetime.strptime(value, "%d.%m.%Y").date()
    raise ValueError("Unsupported explicit document date.")


def _number(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"v?\d{1,6}(?:\.\d{1,6}){0,5}", value, re.I):
        raise ValueError("Unsupported explicit document version.")
    return tuple(int(part) for part in value.lstrip("vV").split("."))


def _without_suffixes(
    title: str, dates: set[date], numbers: set[tuple[int, ...]],
) -> str:
    while match := _SUFFIX.search(title):
        label, value = match.group("label", "value")
        if label.casefold() in {"дата", "оновлено"}:
            dates.add(_date(value))
        else:
            numbers.add(_number(value))
        title = title[:match.start()].rstrip(" _-–—,;")
    return title


def extract_version(name: str, text: str) -> AdviceVersion | None:
    """Read clear headers or a hut title; uncertain matches stay separate.

    Metadata comes from the first 20 lines and labeled filename suffixes,
    such as ``(v2)`` or ``(дата 2026-10-19)``. Advice body dates, fuzzy topic
    synonyms and documents covering several places are not inferred.
    """
    dates: set[date] = set()
    numbers: set[tuple[int, ...]] = set()
    topics: set[str] = set()
    places: set[str] = set()
    identities: set[tuple[str, str]] = set()
    first_heading: str | None = None
    try:
        stem = re.sub(r"\.(?:pdf|docx|txt)$", "", name, flags=re.I)
        title = _without_suffixes(stem, dates, numbers)
        if identity := _identity(title):
            identities.add(identity)

        for line in unicodedata.normalize("NFC", text).splitlines()[:20]:
            if match := _HEADER.fullmatch(line):
                label, value = match.groups()
                label = label.casefold()
                if label == "тема":
                    topic = _normalize(value)
                    if not topic:
                        return None
                    topics.add(topic)
                elif label == "місце":
                    place = _place(value)
                    if place is None:
                        return None
                    places.add(place)
                elif label == "версія":
                    numbers.add(_number(value))
                else:
                    dates.add(_date(value))
            elif line.strip() and first_heading is None:
                first_heading = line.strip()

        if (
            first_heading is not None
            and _HUT.search(first_heading)
        ):
            heading = _without_suffixes(first_heading, dates, numbers)
            if ":" not in heading.rstrip(":"):
                if identity := _identity(heading):
                    identities.add(identity)
    except ValueError:
        return None

    # Both hints must agree; a header cannot silently override another place.
    if len(identities) > 1 or len(topics) > 1 or len(places) > 1:
        return None
    if identities:
        topic, place = next(iter(identities))
        if topics.difference({topic}) or places.difference({place}):
            return None
    elif len(topics) == len(places) == 1:
        topic, place = next(iter(topics)), next(iter(places))
    else:
        return None

    # Equivalent numeric spellings such as 1 and 1.0 do not conflict.
    normalized_numbers = {
        number + (0,) * (6 - len(number)) for number in numbers
    }
    if len(dates) > 1 or len(normalized_numbers) > 1:
        return None
    return AdviceVersion(
        topic, place, next(iter(dates), None), min(numbers, default=None),
    )


def compare_versions(
    left: AdviceVersion,
    left_timestamp: datetime,
    right: AdviceVersion,
    right_timestamp: datetime,
) -> int | None:
    """Compare revisions, returning None when their order is not reliable.

    Two explicit numeric versions take precedence over upload times. An
    explicit date that contradicts their order makes the match uncertain.
    Otherwise each explicit date uses UTC midnight, falling back to its
    aware modification/upload timestamp. A lone numeric version does not
    establish order, so dates/timestamps are used for that pair instead.
    """
    if (left.topic, left.place) != (right.topic, right.place):
        return None
    if (
        left_timestamp.utcoffset() is None
        or right_timestamp.utcoffset() is None
    ):
        return None

    if left.number is not None and right.number is not None:
        width = max(len(left.number), len(right.number))
        a = left.number + (0,) * (width - len(left.number))
        b = right.number + (0,) * (width - len(right.number))
        order = (a > b) - (a < b)
        if left.published_on is not None and right.published_on is not None:
            date_order = (
                (left.published_on > right.published_on)
                - (left.published_on < right.published_on)
            )
            if date_order and date_order != order:
                return None
        return order

    left_time = (
        datetime.combine(left.published_on, time.min, UTC)
        if left.published_on else left_timestamp.astimezone(UTC)
    )
    right_time = (
        datetime.combine(right.published_on, time.min, UTC)
        if right.published_on else right_timestamp.astimezone(UTC)
    )
    return (left_time > right_time) - (left_time < right_time)
