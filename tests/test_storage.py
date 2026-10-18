"""Storage checks without Telegram, Google Drive, or private documents."""

import sqlite3
from datetime import datetime
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
