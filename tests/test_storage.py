"""Storage checks without Telegram, Google Drive, or private documents."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.storage import Store


def test_initialize_is_repeatable(tmp_path: Path) -> None:
    store = Store(tmp_path / "data")

    store.initialize()
    store.initialize()

    assert store.list_documents() == []


def test_document_and_metadata_survive_new_store_instance(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    content = "Правила лісу".encode("utf-8")

    document_id = store.upsert_drive_document(
        "file-1", "rules.txt", content, "Правила лісу", "2026-10-18T10:00:00Z"
    )
    reopened = Store(tmp_path)
    document = reopened.list_documents()[0]

    assert document.id == document_id
    assert document.source == "drive"
    assert document.source_id == "file-1"
    assert document.name == "rules.txt"
    assert document.text == "Правила лісу"
    assert document.modified_at == "2026-10-18T10:00:00Z"
    assert document.active is True
    assert datetime.fromisoformat(document.created_at).utcoffset() is not None
    assert document.created_at == document.imported_at
    assert not document.relative_path.is_absolute()
    assert (tmp_path / document.relative_path).read_bytes() == content


def test_reimport_updates_existing_document_without_duplicates(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    first_id = store.upsert_drive_document(
        "file-1", "old.txt", b"old", "old", None
    )
    original = store.list_documents()[0]

    second_id = store.upsert_drive_document(
        "file-1", "new.txt", b"new", "new", "2026-10-18T11:00:00Z"
    )
    documents = store.list_documents()

    assert len(documents) == 1
    updated = documents[0]
    assert first_id == second_id == updated.id
    assert updated.name == "new.txt"
    assert updated.text == "new"
    assert updated.modified_at == "2026-10-18T11:00:00Z"
    assert updated.created_at == original.created_at
    assert updated.imported_at >= original.imported_at
    assert (tmp_path / updated.relative_path).read_bytes() == b"new"
    assert not (tmp_path / original.relative_path).exists()


def test_untrusted_name_and_source_id_cannot_escape_data_directory(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    store = Store(data_dir)
    store.initialize()

    store.upsert_drive_document(
        "../../outside", "../../outside.txt", b"safe", "safe", None
    )
    document = store.list_documents()[0]
    stored_path = (data_dir / document.relative_path).resolve()

    assert stored_path.is_relative_to(data_dir.resolve())
    assert stored_path.read_bytes() == b"safe"
    assert document.name == "../../outside.txt"
    assert document.source_id == "../../outside"
    assert not (tmp_path / "outside.txt").exists()


def test_database_failure_keeps_previous_document_and_original_file(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    store.upsert_drive_document("file-1", "old.txt", b"old", "old", None)
    original = store.list_documents()[0]
    original_files = set((tmp_path / "documents").iterdir())

    connection = sqlite3.connect(store.database_path)
    try:
        connection.executescript(
            """
            CREATE TRIGGER reject_update
            BEFORE UPDATE ON documents
            BEGIN
                SELECT RAISE(ABORT, 'simulated database failure');
            END;
            """
        )
    finally:
        connection.close()

    with pytest.raises(
        sqlite3.IntegrityError, match="simulated database failure"
    ):
        store.upsert_drive_document("file-1", "new.txt", b"new", "new", None)

    assert store.list_documents() == [original]
    assert (tmp_path / original.relative_path).read_bytes() == b"old"
    assert set((tmp_path / "documents").iterdir()) == original_files


def test_database_failure_keeps_a_repaired_existing_original(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    store.upsert_drive_document("file-1", "rules.txt", b"original", "text", None)
    original = store.list_documents()[0]
    original_path = tmp_path / original.relative_path
    original_path.write_bytes(b"")
    with sqlite3.connect(store.database_path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER reject_update
            BEFORE UPDATE ON documents
            BEGIN
                SELECT RAISE(ABORT, 'simulated database failure');
            END;
            """
        )

    with pytest.raises(
        sqlite3.IntegrityError, match="simulated database failure"
    ):
        store.upsert_drive_document(
            "file-1", "rules.txt", b"original", "new text", None
        )

    assert store.list_documents() == [original]
    assert original_path.read_bytes() == b"original"


def test_updating_one_document_preserves_another_with_identical_content(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    store.upsert_drive_document("file-1", "first.txt", b"same", "same", None)
    second_id = store.upsert_drive_document(
        "file-2", "second.txt", b"same", "same", None
    )

    store.upsert_drive_document(
        "file-1", "first.txt", b"changed", "changed", None
    )

    second = next(doc for doc in store.list_documents() if doc.id == second_id)
    assert second.text == "same"
    assert (tmp_path / second.relative_path).read_bytes() == b"same"


def test_initialize_migrates_old_schema_without_changing_documents(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    original_path = tmp_path / "documents" / "legacy.bin"
    original_path.parent.mkdir()
    original_path.write_bytes(b"legacy content")
    created_at = "2026-10-01T09:00:00+00:00"
    with sqlite3.connect(store.database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                name TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                text TEXT NOT NULL,
                modified_at TEXT,
                created_at TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                UNIQUE (source, source_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                42, "drive", "legacy-file", "legacy.txt",
                "documents/legacy.bin", "legacy text", None,
                created_at, created_at, 1,
            ),
        )

    store.initialize()
    store.initialize()
    document = Store(tmp_path).list_documents()[0]

    assert document.id == 42
    assert document.source == "drive"
    assert document.source_id == "legacy-file"
    assert document.name == "legacy.txt"
    assert document.text == "legacy text"
    assert document.created_at == document.imported_at == created_at
    assert document.active is True
    assert document.owner_id is None
    assert document.owner_name is None
    assert document.uploaded_at is None
    assert document.deleted_at is None
    assert (tmp_path / document.relative_path).read_bytes() == (
        b"legacy content"
    )


def test_telegram_upload_and_metadata_survive_reopening(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    sent_at = datetime(
        2026, 10, 19, 13, 25, tzinfo=timezone(timedelta(hours=3))
    )

    document_id = store.add_telegram_document(
        source_id="10:25",
        name="Поради.txt",
        content="Нові поради".encode("utf-8"),
        text="Нові поради",
        owner_id=9_000_000_001,
        owner_name="Олена (@forest_reader)",
        uploaded_at=sent_at,
    )
    reopened = Store(tmp_path)
    reopened.initialize()
    document = reopened.list_documents()[0]

    assert document.id == document_id
    assert document.source == "telegram"
    assert document.source_id == "10:25"
    assert document.name == "Поради.txt"
    assert document.text == "Нові поради"
    assert document.owner_id == 9_000_000_001
    assert document.owner_name == "Олена (@forest_reader)"
    assert document.uploaded_at == "2026-10-19T10:25:00+00:00"
    assert document.modified_at is None
    assert document.active is True
    assert document.created_at == document.imported_at
    assert datetime.fromisoformat(document.created_at).utcoffset() is not None
    assert (tmp_path / document.relative_path).read_bytes() == (
        "Нові поради".encode("utf-8")
    )


@pytest.mark.parametrize("active", [True, False])
def test_replayed_telegram_message_preserves_first_record(
    tmp_path: Path, active: bool,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    document_id = store.add_telegram_document(
        "10:25", "first.txt", b"first", "first text", 100, "First sender",
        datetime(2026, 10, 19, 10, tzinfo=timezone.utc),
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE documents SET active = ? WHERE id = ?",
            (int(active), document_id),
        )
    original = store.list_documents()[0]
    original_files = set((tmp_path / "documents").iterdir())

    repeated_id = store.add_telegram_document(
        "10:25", "changed.txt", b"changed", "changed text", 200,
        "Other sender", datetime(2026, 10, 20, tzinfo=timezone.utc),
    )

    assert repeated_id == document_id
    assert store.list_documents() == [original]
    assert set((tmp_path / "documents").iterdir()) == original_files
    assert (tmp_path / original.relative_path).read_bytes() == b"first"


def test_matching_files_from_different_uploads_keep_separate_owners(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    sent_at = datetime(2026, 10, 19, tzinfo=timezone.utc)

    first_id = store.add_telegram_document(
        "10:25", "same.txt", b"same", "same", 100, "First", sent_at,
    )
    second_id = store.add_telegram_document(
        "20:25", "same.txt", b"same", "same", 200, "Second", sent_at,
    )
    third_id = store.add_telegram_document(
        "10:26", "same.txt", b"same", "same", 100, "First", sent_at,
    )
    documents = store.list_documents()

    assert len({first_id, second_id, third_id}) == len(documents) == 3
    assert [document.owner_id for document in documents] == [100, 200, 100]
    assert len({document.relative_path for document in documents}) == 3
    assert all(
        (tmp_path / document.relative_path).read_bytes() == b"same"
        for document in documents
    )


def test_telegram_upload_cannot_replace_matching_drive_source(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    drive_id = store.upsert_drive_document(
        "10:25", "same.txt", b"same", "same", None,
    )

    telegram_id = store.add_telegram_document(
        "10:25", "same.txt", b"same", "same", 100, "First",
        datetime(2026, 10, 19, tzinfo=timezone.utc),
    )
    drive, telegram = store.list_documents()

    assert drive_id != telegram_id
    assert drive.source == "drive"
    assert drive.owner_id is None
    assert drive.owner_name is None
    assert drive.uploaded_at is None
    assert telegram.source == "telegram"
    assert telegram.owner_id == 100
    assert drive.relative_path != telegram.relative_path
    assert (tmp_path / drive.relative_path).read_bytes() == b"same"
    assert (tmp_path / telegram.relative_path).read_bytes() == b"same"


def test_telegram_database_failure_removes_only_the_new_original(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    sent_at = datetime(2026, 10, 19, tzinfo=timezone.utc)
    store.upsert_drive_document(
        "drive-1", "drive.txt", b"drive", "drive", None,
    )
    store.add_telegram_document(
        "10:25", "first.txt", b"first", "first", 100, "First", sent_at,
    )
    originals = store.list_documents()
    original_files = set((tmp_path / "documents").iterdir())
    with sqlite3.connect(store.database_path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER reject_telegram_insert
            BEFORE INSERT ON documents
            WHEN NEW.source = 'telegram'
            BEGIN
                SELECT RAISE(ABORT, 'simulated database failure');
            END;
            """
        )

    with pytest.raises(
        sqlite3.IntegrityError, match="simulated database failure"
    ):
        store.add_telegram_document(
            "20:25", "second.txt", b"second", "second", 200, "Second",
            sent_at,
        )

    assert store.list_documents() == originals
    assert set((tmp_path / "documents").iterdir()) == original_files
    assert (tmp_path / originals[0].relative_path).read_bytes() == b"drive"
    assert (tmp_path / originals[1].relative_path).read_bytes() == b"first"


def test_telegram_filename_and_message_id_cannot_escape_data_directory(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    store = Store(data_dir)
    store.initialize()

    store.add_telegram_document(
        "../../outside", "../../outside.txt", b"safe", "safe", 100,
        "Sender", datetime(2026, 10, 19, tzinfo=timezone.utc),
    )
    document = store.list_documents()[0]
    original_path = (data_dir / document.relative_path).resolve()

    assert original_path.is_relative_to(data_dir.resolve())
    assert original_path.read_bytes() == b"safe"
    assert document.name == "../../outside.txt"
    assert document.source_id == "../../outside"
    assert not (tmp_path / "outside.txt").exists()


def test_telegram_upload_rejects_naive_timestamp_before_persisting(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()

    with pytest.raises(ValueError, match="timezone"):
        store.add_telegram_document(
            "10:25", "rules.txt", b"rules", "rules", 100, "Sender",
            datetime(2026, 10, 19),
        )

    assert store.list_documents() == []
    assert list((tmp_path / "documents").iterdir()) == []


def _upload(
    store: Store, source_id: str = "10:25", owner_id: int = 100,
) -> int:
    return store.add_telegram_document(
        source_id, "Поради.txt", b"shared content", "Shared advice",
        owner_id, f"Sender {owner_id}",
        datetime(2026, 10, 19, 10, tzinfo=timezone.utc),
    )


def test_own_upload_list_is_isolated_and_keeps_inactive_documents(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    first = _upload(store)
    other = _upload(store, "20:25", 200)
    second = _upload(store, "10:26")
    inactive = _upload(store, "10:27")
    deleted = _upload(store, "10:28")
    drive = store.upsert_drive_document(
        "drive-1", "drive.txt", b"drive", "Drive advice", None,
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE documents SET active = 0 WHERE id = ?", (inactive,),
        )
        connection.execute(
            "UPDATE documents SET owner_id = 100 WHERE id = ?", (drive,),
        )
    assert store.delete_telegram_document(deleted, 100) is True

    assert [doc.id for doc in store.list_telegram_documents(100)] == [
        inactive, second, first,
    ]
    assert [doc.id for doc in store.list_telegram_documents(200)] == [other]
    assert store.list_telegram_documents(999) == []


def test_deleting_own_upload_clears_content_and_survives_restart(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    document_id = _upload(store)
    original = store.list_documents()[0]

    assert store.delete_telegram_document(document_id, 100) is True

    reopened = Store(tmp_path)
    reopened.initialize()
    deleted = reopened.list_documents()[0]
    assert reopened.list_telegram_documents(100) == []
    assert deleted.id == original.id
    assert deleted.source == original.source == "telegram"
    assert deleted.source_id == original.source_id
    assert deleted.owner_id == original.owner_id == 100
    assert deleted.owner_name == original.owner_name
    assert deleted.uploaded_at == original.uploaded_at
    assert deleted.active is False
    assert deleted.text == ""
    assert deleted.deleted_at is not None
    assert datetime.fromisoformat(deleted.deleted_at).utcoffset() == (
        timedelta(0)
    )
    assert not (tmp_path / original.relative_path).exists()


@pytest.mark.parametrize("target", ["other-owner", "drive", "missing"])
def test_delete_denies_foreign_drive_and_missing_documents(
    tmp_path: Path, target: str,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    _upload(store)
    foreign = _upload(store, "20:25", 200)
    drive = store.upsert_drive_document(
        "drive-1", "drive.txt", b"drive", "Drive advice", None,
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE documents SET owner_id = 100 WHERE id = ?", (drive,),
        )
    originals = store.list_documents()
    original_files = {
        doc.relative_path: (tmp_path / doc.relative_path).read_bytes()
        for doc in originals
    }
    identifier = {"other-owner": foreign, "drive": drive, "missing": 999}[
        target
    ]

    assert store.delete_telegram_document(identifier, 100) is False

    assert store.list_documents() == originals
    assert {
        doc.relative_path: (tmp_path / doc.relative_path).read_bytes()
        for doc in originals
    } == original_files


def test_repeated_delete_and_upload_replay_do_not_restore_content(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    document_id = _upload(store)
    assert store.delete_telegram_document(document_id, 100) is True
    tombstone = store.list_documents()[0]

    assert store.delete_telegram_document(document_id, 100) is False
    replayed_id = store.add_telegram_document(
        "10:25", "Changed.txt", b"changed", "Changed advice", 200,
        "Other sender", datetime(2026, 10, 20, tzinfo=timezone.utc),
    )

    assert replayed_id == document_id
    assert store.list_documents() == [tombstone]
    assert store.list_telegram_documents(100) == []
    assert list((tmp_path / "documents").iterdir()) == []


def test_delete_failure_preserves_document_and_original(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    document_id = _upload(store)
    original = store.list_documents()[0]
    with sqlite3.connect(store.database_path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER reject_delete
            BEFORE UPDATE ON documents
            BEGIN
                SELECT RAISE(ABORT, 'simulated deletion failure');
            END;
            """
        )

    with pytest.raises(
        sqlite3.IntegrityError, match="simulated deletion failure",
    ):
        store.delete_telegram_document(document_id, 100)

    assert store.list_documents() == [original]
    assert store.list_telegram_documents(100) == [original]
    assert (tmp_path / original.relative_path).read_bytes() == (
        b"shared content"
    )


def test_original_cleanup_failure_does_not_restore_deleted_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    document_id = _upload(store)
    original = store.list_documents()[0]
    original_path = tmp_path / original.relative_path
    original_unlink = Path.unlink

    def denied_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == original_path:
            raise PermissionError("simulated filesystem denial")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", denied_unlink)

    assert store.delete_telegram_document(document_id, 100) is True

    assert store.list_telegram_documents(100) == []
    assert store.list_documents()[0].text == ""
    assert store.list_documents()[0].active is False
    assert original_path.is_file()
    assert "Could not remove obsolete archive file" in caplog.text


def test_delete_preserves_identical_uploads_and_drive_originals(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    deleted_id = _upload(store)
    _upload(store, "20:25", 200)
    _upload(store, "10:26")
    store.upsert_drive_document(
        "10:25", "drive.txt", b"shared content", "Drive advice", None,
    )
    originals = store.list_documents()[1:]

    assert store.delete_telegram_document(deleted_id, 100) is True

    assert store.list_documents()[1:] == originals
    assert all(
        (tmp_path / doc.relative_path).read_bytes() == b"shared content"
        for doc in originals
    )


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "database"])
def test_delete_never_unlinks_a_path_outside_the_documents_directory(
    tmp_path: Path, path_kind: str,
) -> None:
    store = Store(tmp_path / "data")
    store.initialize()
    document_id = _upload(store)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"must survive")
    tampered_path = {
        "relative": "../outside.bin",
        "absolute": str(outside.resolve()),
        "database": "archive.sqlite3",
    }[path_kind]
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE documents SET relative_path = ? WHERE id = ?",
            (tampered_path, document_id),
        )

    assert store.delete_telegram_document(document_id, 100) is True

    assert outside.read_bytes() == b"must survive"
    assert store.database_path.is_file()
    assert store.list_documents()[0].active is False
    assert store.list_documents()[0].text == ""


def test_initialize_migrates_the_telegram_upload_schema(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    original_path = tmp_path / "documents" / "upload.bin"
    original_path.parent.mkdir()
    original_path.write_bytes(b"uploaded content")
    timestamp = "2026-10-19T10:00:00+00:00"
    with sqlite3.connect(store.database_path) as connection:
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
                UNIQUE (source, source_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?)",
            (
                42, "telegram", "10:25", "upload.txt",
                "documents/upload.bin", "Uploaded advice", None,
                timestamp, timestamp, 1, 100, "Sender", timestamp,
            ),
        )

    store.initialize()
    store.initialize()

    document = store.list_telegram_documents(100)[0]
    assert document.id == 42
    assert document.text == "Uploaded advice"
    assert document.owner_name == "Sender"
    assert document.uploaded_at == timestamp
    assert document.deleted_at is None
    assert original_path.read_bytes() == b"uploaded content"


def test_active_document_check_accepts_empty_and_duplicate_identifiers(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    uploaded = _upload(store)
    drive = store.upsert_drive_document(
        "drive-1", "drive.txt", b"drive", "Drive advice", None,
    )

    assert store.are_documents_active(()) is True
    assert store.are_documents_active((uploaded, drive, uploaded)) is True
    assert store.are_documents_active((uploaded, 999)) is False


def test_active_document_check_rejects_inactive_and_deleted_sources(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    active = _upload(store)
    inactive = _upload(store, "10:26")
    deleted = _upload(store, "10:27")
    assert store.delete_telegram_document(deleted, 100) is True
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE documents SET active = 0 WHERE id = ?", (inactive,),
        )

    assert store.are_documents_active((active,)) is True
    assert store.are_documents_active((active, inactive)) is False
    assert store.are_documents_active((active, deleted)) is False
