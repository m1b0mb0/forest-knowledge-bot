"""Persist imported archive documents and their original files."""

import hashlib
import logging
import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """Document metadata, extracted text, and relative original-file path."""

    id: int
    source: str
    source_id: str
    name: str
    relative_path: Path
    text: str
    modified_at: str | None
    created_at: str
    imported_at: str
    active: bool


class Store:
    """Use one data directory for SQLite and downloaded archive files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()
        self.database_path = self.data_dir / "archive.sqlite3"

    def initialize(self) -> None:
        """Create the data directory and schema if they do not exist."""
        (self.data_dir / "documents").mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
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

    def upsert_drive_document(
        self,
        source_id: str,
        name: str,
        content: bytes,
        text: str,
        modified_at: str | None,
    ) -> int:
        """Insert or update a Drive file, retaining its ID and creation time.

        A changed file gets a new generated path before the database commits.
        The previous file is removed only after the update succeeds.
        """
        fingerprint = hashlib.sha256(
            source_id.encode("utf-8") + b"\0" + content
        ).hexdigest()
        relative_path = Path("documents") / f"{fingerprint}.bin"
        destination = self.data_dir / relative_path
        now = datetime.now(timezone.utc).isoformat()
        created_file = False
        previous_path: Path | None = None

        with closing(self._connect()) as connection:
            try:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    previous = connection.execute(
                        """
                        SELECT relative_path FROM documents
                        WHERE source = 'drive' AND source_id = ?
                        """,
                        (source_id,),
                    ).fetchone()
                    if previous is not None:
                        previous_path = Path(previous["relative_path"])

                    if not destination.exists():
                        self._write_file(destination, content)
                        created_file = True

                    connection.execute(
                        """
                        INSERT INTO documents (
                            source, source_id, name, relative_path, text,
                            modified_at, created_at, imported_at
                        ) VALUES ('drive', ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (source, source_id) DO UPDATE SET
                            name = excluded.name,
                            relative_path = excluded.relative_path,
                            text = excluded.text,
                            modified_at = excluded.modified_at,
                            imported_at = excluded.imported_at
                        """,
                        (
                            source_id,
                            name,
                            relative_path.as_posix(),
                            text,
                            modified_at,
                            now,
                            now,
                        ),
                    )
                    row = connection.execute(
                        """
                        SELECT id FROM documents
                        WHERE source = 'drive' AND source_id = ?
                        """,
                        (source_id,),
                    ).fetchone()
                    document_id = int(row["id"])
            except Exception:
                if created_file:
                    self._remove_file(destination)
                raise

        if previous_path is not None and previous_path != relative_path:
            self._remove_file(self.data_dir / previous_path)
        return document_id

    def list_documents(self) -> list[Document]:
        """Return stored documents in insertion order."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY id"
            ).fetchall()
        return [
            Document(
                id=row["id"],
                source=row["source"],
                source_id=row["source_id"],
                name=row["name"],
                relative_path=Path(row["relative_path"]),
                text=row["text"],
                modified_at=row["modified_at"],
                created_at=row["created_at"],
                imported_at=row["imported_at"],
                active=bool(row["active"]),
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _write_file(destination: Path, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=".import-", suffix=".tmp"
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(content)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _remove_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove obsolete archive file: %s", path)
