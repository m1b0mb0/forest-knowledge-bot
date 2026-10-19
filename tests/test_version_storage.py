"""Atomic version updates and migrations without private files or APIs."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot.storage import Store

OAK = "Членські внески рукавички Під дубом.txt"
PINE = "Членські внески рукавички У соснах.txt"


def upload(
    store: Store, source_id: str, day: int, *,
    metadata: str = "", name: str = OAK,
) -> int:
    text = f"{metadata}\nПорада {source_id}.".strip()
    return store.add_telegram_document(
        source_id, name, text.encode("utf-8"), text, 101, "@test_author",
        datetime(2026, 1, day, 12, tzinfo=timezone.utc),
    )


def legacy_store(tmp_path: Path) -> Store:
    """Build the committed schema that predates version metadata."""
    instance = Store(tmp_path)
    (tmp_path / "documents").mkdir()
    with sqlite3.connect(instance.database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY, source TEXT NOT NULL,
                source_id TEXT NOT NULL, name TEXT NOT NULL,
                relative_path TEXT NOT NULL, text TEXT NOT NULL,
                modified_at TEXT, created_at TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                owner_id INTEGER, owner_name TEXT, uploaded_at TEXT,
                deleted_at TEXT, UNIQUE (source, source_id)
            );
            """
        )
        for identifier in (1, 2):
            stamp = f"2026-01-0{identifier}T12:00:00+00:00"
            text = f"Дата: 2026-01-0{identifier}\nПорада {identifier}."
            path = f"documents/legacy-{identifier}.bin"
            (tmp_path / path).write_bytes(text.encode("utf-8"))
            connection.execute(
                "INSERT INTO documents (id, source, source_id, name, "
                "relative_path, text, modified_at, created_at, imported_at) "
                "VALUES (?, 'drive', ?, ?, ?, ?, ?, ?, ?)",
                (identifier, f"source-{identifier}", OAK, path, text,
                 stamp, stamp, stamp),
            )
    return instance


def test_new_advice_only_supersedes_the_same_topic_and_place(
    store: Store,
) -> None:
    old = upload(store, "old", 1)
    pine = upload(store, "pine", 1, name=PINE)
    warming = upload(
        store, "warming", 1, name="Як утеплити рукавичку Під дубом.txt",
    )
    before = {item.id: item for item in store.list_documents()}
    latest = upload(store, "latest", 2)
    after = {item.id: item for item in store.list_documents()}

    assert not after[old].active
    assert after[old].superseded_by == latest
    assert after[latest].active
    assert after[pine] == before[pine]
    assert after[warming] == before[warming]
    assert after[old].text == before[old].text
    original = store.data_dir / after[old].relative_path
    assert original.read_bytes() == before[old].text.encode("utf-8")


@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
def test_explicit_date_wins_regardless_of_arrival_order(
    store: Store, order: tuple[int, int],
) -> None:
    identifiers = {}
    for revision in order:
        identifiers[revision] = upload(
            store, str(revision), 5 - revision,
            metadata=f"Дата: 2026-01-0{revision}",
        )
    saved = {item.id: item for item in store.list_documents()}
    assert not saved[identifiers[1]].active
    assert saved[identifiers[1]].superseded_by == identifiers[2]
    assert saved[identifiers[2]].active


def test_numeric_revision_wins_over_later_upload_time(store: Store) -> None:
    latest = upload(store, "v2", 1, metadata="Версія: 2")
    old = upload(store, "v1", 5, metadata="Версія: 1")
    saved = {item.id: item for item in store.list_documents()}
    assert saved[latest].active
    assert not saved[old].active
    assert saved[old].superseded_by == latest


def test_explicit_date_and_version_conflict_keeps_both(store: Store) -> None:
    upload(store, "first", 1, metadata="Версія: 1\nДата: 2026-01-03")
    upload(store, "second", 2, metadata="Версія: 2\nДата: 2026-01-02")
    assert all(item.active for item in store.list_documents())
    assert all(item.superseded_by is None for item in store.list_documents())


def test_new_revision_can_supersede_two_equal_revisions(store: Store) -> None:
    first = upload(store, "v1-first", 1, metadata="Версія: 1")
    second = upload(store, "v1-second", 3, metadata="Версія: 1.0")
    assert all(item.active for item in store.list_documents())

    latest = upload(store, "v2", 2, metadata="Версія: 2")

    saved = {item.id: item for item in store.list_documents()}
    assert saved[latest].active
    assert not saved[first].active and not saved[second].active
    assert saved[first].superseded_by == saved[second].superseded_by == latest


def test_ambiguous_timestamp_order_does_not_break_an_explicit_tie(
    store: Store,
) -> None:
    upload(store, "v1-first", 1, metadata="Версія: 1")
    upload(store, "v1-second", 3, metadata="Версія: 1")
    upload(store, "no-version", 2)
    assert all(item.active for item in store.list_documents())


def test_contradictory_version_cycle_preserves_current_availability(
    store: Store,
) -> None:
    old = upload(store, "v1", 3, metadata="Версія: 1")
    winner = upload(store, "v2", 1, metadata="Версія: 2")
    previous = {item.id: item for item in store.list_documents()}
    uncertain = upload(store, "unknown", 2)
    saved = {item.id: item for item in store.list_documents()}
    assert saved[old] == previous[old]
    assert saved[winner] == previous[winner]
    assert saved[uncertain].active


def test_migration_recognizes_old_documents_and_is_repeatable(
    tmp_path: Path,
) -> None:
    instance = legacy_store(tmp_path)
    instance.initialize()
    previous = instance.list_documents()
    assert not previous[0].active
    assert previous[0].superseded_by == previous[1].id
    assert previous[1].active
    assert previous[0].advice_topic == "членські внески"
    assert previous[0].advice_place == "під дубом"
    assert previous[0].version_date == "2026-01-01"
    instance.initialize()
    assert instance.list_documents() == previous
    assert all(
        (tmp_path / item.relative_path).read_bytes()
        == item.text.encode("utf-8") for item in previous
    )


def test_migration_recovers_a_partial_older_schema(tmp_path: Path) -> None:
    instance = legacy_store(tmp_path)
    with sqlite3.connect(instance.database_path) as connection:
        connection.execute(
            "ALTER TABLE documents ADD COLUMN advice_topic TEXT"
        )
    instance.initialize()
    old, latest = instance.list_documents()
    assert old.advice_topic == latest.advice_topic == "членські внески"
    assert not old.active and latest.active


def test_migration_failure_rolls_back_schema_and_allows_retry(
    tmp_path: Path,
) -> None:
    instance = legacy_store(tmp_path)
    with sqlite3.connect(instance.database_path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER reject_metadata BEFORE UPDATE ON documents
            BEGIN SELECT RAISE(ABORT, 'simulated migration failure'); END;
            """
        )
    with pytest.raises(sqlite3.IntegrityError, match="migration failure"):
        instance.initialize()
    with sqlite3.connect(instance.database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(documents)")
        }
        assert "advice_topic" not in columns
        assert "superseded_by" not in columns
        assert connection.execute(
            "SELECT COUNT(*) FROM documents WHERE active = 1"
        ).fetchone()[0] == 2
        connection.execute("DROP TRIGGER reject_metadata")
    instance.initialize()
    assert not instance.list_documents()[0].active


def test_deleted_latest_metadata_blocks_a_new_upload_of_older_content(
    store: Store,
) -> None:
    upload(store, "v1", 1, metadata="Версія: 1")
    latest = upload(store, "v2", 2, metadata="Версія: 2")
    assert store.delete_telegram_document(latest, 101)
    deleted = store.list_documents()[1]
    assert deleted.text == "" and deleted.version_number == "2"
    assert deleted.advice_topic == "членські внески"

    older = upload(store, "reuploaded-v1", 5, metadata="Версія: 1")
    reopened = Store(store.data_dir)
    reopened.initialize()
    assert not any(item.active for item in reopened.list_documents())
    assert reopened.list_documents()[-1].id == older
    assert reopened.list_documents()[-1].superseded_by == latest


def test_reimport_time_does_not_make_an_old_drive_file_newer(
    store: Store,
) -> None:
    old = store.upsert_drive_document("old", OAK, b"old", "old", None)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE documents SET created_at = ?, imported_at = ? "
            "WHERE id = ?",
            ("2026-01-01T00:00:00+00:00", "2035-01-01T00:00:00+00:00", old),
        )
    latest = upload(store, "new", 2)
    store.upsert_drive_document("old", OAK, b"old", "old", None)
    saved = {item.id: item for item in store.list_documents()}
    assert not saved[old].active and saved[latest].active


def test_drive_regression_keeps_newest_known_content_and_original(
    store: Store,
) -> None:
    latest = store.upsert_drive_document(
        "drive", OAK, b"version two", "Версія: 2\nНова порада.",
        "2026-01-01T00:00:00Z",
    )
    original = store.list_documents()[0]
    files = set((store.data_dir / "documents").iterdir())
    assert store.upsert_drive_document(
        "drive", OAK, b"version one", "Версія: 1\nСтара порада.",
        "2026-01-05T00:00:00Z",
    ) == latest
    assert store.list_documents() == [original]
    assert set((store.data_dir / "documents").iterdir()) == files
    assert (store.data_dir / original.relative_path).read_bytes() == (
        b"version two"
    )


def test_failed_supersession_keeps_previous_advice_and_removes_new_file(
    store: Store,
) -> None:
    old = upload(store, "v1", 1, metadata="Версія: 1")
    previous = store.list_documents()
    files = set((store.data_dir / "documents").iterdir())
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            f"CREATE TRIGGER reject_supersession BEFORE UPDATE ON documents "
            f"WHEN OLD.id = {old} AND NEW.active = 0 "
            "BEGIN SELECT RAISE(ABORT, 'simulated version failure'); END;"
        )
    with pytest.raises(sqlite3.IntegrityError, match="version failure"):
        upload(store, "v2", 2, metadata="Версія: 2")
    assert store.list_documents() == previous
    assert set((store.data_dir / "documents").iterdir()) == files
    assert (store.data_dir / previous[0].relative_path).read_bytes() == (
        previous[0].text.encode("utf-8")
    )
