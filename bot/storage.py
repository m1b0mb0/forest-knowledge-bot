"""Persist imported archive documents and their original files."""

import hashlib
import logging
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from bot.services.versions import (
    AdviceVersion,
    compare_versions,
    extract_version,
)

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
    owner_id: int | None = None
    owner_name: str | None = None
    uploaded_at: str | None = None
    deleted_at: str | None = None
    advice_topic: str | None = None
    advice_place: str | None = None
    version_date: str | None = None
    version_number: str | None = None
    superseded_by: int | None = None


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
                    owner_id INTEGER,
                    owner_name TEXT,
                    uploaded_at TEXT,
                    deleted_at TEXT,
                    advice_topic TEXT,
                    advice_place TEXT,
                    version_date TEXT,
                    version_number TEXT,
                    superseded_by INTEGER,
                    UNIQUE (source, source_id)
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(documents)")
            }
            backfill_versions = not {
                "advice_topic", "advice_place", "version_date",
                "version_number", "superseded_by",
            }.issubset(columns)
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                for name, column_type in (
                    ("owner_id", "INTEGER"),
                    ("owner_name", "TEXT"),
                    ("uploaded_at", "TEXT"),
                    ("deleted_at", "TEXT"),
                    ("advice_topic", "TEXT"),
                    ("advice_place", "TEXT"),
                    ("version_date", "TEXT"),
                    ("version_number", "TEXT"),
                    ("superseded_by", "INTEGER"),
                ):
                    if name not in columns:
                        connection.execute(
                            f"ALTER TABLE documents ADD COLUMN "
                            f"{name} {column_type}"
                        )
                if backfill_versions:
                    self._backfill_versions(connection)

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
        version = extract_version(name, text)

        with closing(self._connect()) as connection:
            try:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    previous = connection.execute(
                        """
                        SELECT * FROM documents
                        WHERE source = 'drive' AND source_id = ?
                        """,
                        (source_id,),
                    ).fetchone()
                    if previous is not None:
                        previous_path = Path(previous["relative_path"])
                        previous_version = self._version(previous)
                        comparison = self._compare_incoming(
                            version, modified_at, previous,
                        )
                        if comparison == -1:
                            # Keep the newest known content and its clock.
                            # A remote rollback must not erase that evidence.
                            return int(previous["id"])
                        reactivate = (
                            not previous["active"]
                            and previous["deleted_at"] is None
                            and previous["superseded_by"] is not None
                            and previous_version is not None
                            and comparison == 1
                        )
                    else:
                        reactivate = False

                    destination_exists = destination.exists()
                    if (
                        not destination_exists
                        or destination.stat().st_size != len(content)
                    ):
                        self._write_file(destination, content)
                        created_file = not destination_exists

                    connection.execute(
                        """
                        INSERT INTO documents (
                            source, source_id, name, relative_path, text,
                            modified_at, created_at, imported_at,
                            advice_topic, advice_place,
                            version_date, version_number
                        ) VALUES ('drive', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (source, source_id) DO UPDATE SET
                            name = excluded.name,
                            relative_path = excluded.relative_path,
                            text = excluded.text,
                            modified_at = excluded.modified_at,
                            imported_at = excluded.imported_at,
                            advice_topic = excluded.advice_topic,
                            advice_place = excluded.advice_place,
                            version_date = excluded.version_date,
                            version_number = excluded.version_number
                        """,
                        (
                            source_id,
                            name,
                            relative_path.as_posix(),
                            text,
                            modified_at,
                            now,
                            now,
                            *self._version_values(version),
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
                    self._supersede_versions(
                        connection, version,
                        reactivate_id=document_id if reactivate else None,
                    )
            except Exception:
                if created_file:
                    self._remove_file(destination)
                raise

        if previous_path is not None and previous_path != relative_path:
            self._remove_file(self.data_dir / previous_path)
        return document_id

    def add_telegram_document(
        self,
        source_id: str,
        name: str,
        content: bytes,
        text: str,
        owner_id: int,
        owner_name: str,
        uploaded_at: datetime,
    ) -> int:
        """Persist an upload, keeping the first record when it is replayed.

        ``source_id`` identifies the Telegram chat and message, not the file.
        Sending identical bytes in another message creates a separate source
        with its own owner. Existing uploads are never overwritten here.
        """
        if uploaded_at.utcoffset() is None:
            raise ValueError("The upload timestamp must include a timezone.")
        uploaded_time = uploaded_at.astimezone(timezone.utc).isoformat()
        fingerprint = hashlib.sha256(
            b"telegram\0" + source_id.encode("utf-8") + b"\0" + content
        ).hexdigest()
        relative_path = Path("documents") / f"{fingerprint}.bin"
        destination = self.data_dir / relative_path
        now = datetime.now(timezone.utc).isoformat()
        created_file = False
        version = extract_version(name, text)

        with closing(self._connect()) as connection:
            try:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    previous = connection.execute(
                        """
                        SELECT id FROM documents
                        WHERE source = 'telegram' AND source_id = ?
                        """,
                        (source_id,),
                    ).fetchone()
                    if previous is not None:
                        return int(previous["id"])

                    destination_exists = destination.exists()
                    if (
                        not destination_exists
                        or destination.stat().st_size != len(content)
                    ):
                        self._write_file(destination, content)
                        created_file = not destination_exists

                    cursor = connection.execute(
                        """
                        INSERT INTO documents (
                            source, source_id, name, relative_path, text,
                            created_at, imported_at,
                            owner_id, owner_name, uploaded_at,
                            advice_topic, advice_place,
                            version_date, version_number
                        ) VALUES (
                            'telegram', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            source_id,
                            name,
                            relative_path.as_posix(),
                            text,
                            now,
                            now,
                            owner_id,
                            owner_name,
                            uploaded_time,
                            *self._version_values(version),
                        ),
                    )
                    document_id = int(cursor.lastrowid)
                    self._supersede_versions(connection, version)
            except Exception:
                if created_file:
                    self._remove_file(destination)
                raise
        return document_id

    def list_documents(self) -> list[Document]:
        """Return stored documents in insertion order."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY id"
            ).fetchall()
        return [self._document(row) for row in rows]

    def list_telegram_documents(self, owner_id: int) -> list[Document]:
        """Return an owner's undeleted uploads, newest first."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                WHERE source = 'telegram' AND owner_id = ?
                    AND deleted_at IS NULL
                ORDER BY id DESC
                """,
                (owner_id,),
            ).fetchall()
        return [self._document(row) for row in rows]

    def are_documents_active(self, document_ids: Sequence[int]) -> bool:
        """Check whether every cited document is still available."""
        identifiers = set(document_ids)
        if not identifiers:
            return True
        placeholders = ", ".join("?" for _ in identifiers)
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM documents WHERE id IN ({placeholders}) "
                "AND active = 1 AND deleted_at IS NULL",
                tuple(identifiers),
            ).fetchone()
        return int(row[0]) == len(identifiers)

    def delete_telegram_document(
        self, document_id: int, owner_id: int,
    ) -> bool:
        """Remove an owner's upload while retaining its replay identity.

        Authorization and the tombstone update share one transaction. The
        original is removed only after that transaction commits, so a failed
        database write leaves both the document and its original intact.
        """
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                    SELECT relative_path FROM documents
                    WHERE id = ? AND source = 'telegram' AND owner_id = ?
                        AND deleted_at IS NULL
                    """,
                (document_id, owner_id),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                """
                    UPDATE documents SET active = 0, text = '', deleted_at = ?
                    WHERE id = ? AND source = 'telegram' AND owner_id = ?
                        AND deleted_at IS NULL
                    """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    document_id, owner_id,
                ),
            )

        destination = self.data_dir / row["relative_path"]
        try:
            resolved = destination.resolve()
            directory = self.data_dir / "documents"
            if (
                resolved == directory
                or not resolved.is_relative_to(directory)
            ):
                logger.warning("Refusing to remove an unsafe document path.")
                return True
        except (OSError, RuntimeError):
            logger.warning("Could not resolve the deleted document's path.")
            return True
        self._remove_file(destination)
        return True

    def _backfill_versions(self, connection: sqlite3.Connection) -> None:
        """Infer profiles once when upgrading a pre-version database."""
        versions: dict[tuple[str, str], AdviceVersion] = {}
        for row in connection.execute("SELECT * FROM documents").fetchall():
            version = extract_version(row["name"], row["text"])
            connection.execute(
                """
                    UPDATE documents SET advice_topic = ?, advice_place = ?,
                        version_date = ?, version_number = ? WHERE id = ?
                """,
                (*self._version_values(version), row["id"]),
            )
            if version is not None:
                versions[(version.topic, version.place)] = version
        for version in versions.values():
            self._supersede_versions(connection, version)

    @staticmethod
    def _version_values(
        version: AdviceVersion | None,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        if version is None:
            return None, None, None, None
        return (
            version.topic,
            version.place,
            version.published_on.isoformat()
            if version.published_on is not None else None,
            ".".join(str(part) for part in version.number)
            if version.number is not None else None,
        )

    @staticmethod
    def _version(row: sqlite3.Row) -> AdviceVersion | None:
        if row["advice_topic"] is None or row["advice_place"] is None:
            return None
        try:
            return AdviceVersion(
                topic=row["advice_topic"],
                place=row["advice_place"],
                published_on=(
                    date.fromisoformat(row["version_date"])
                    if row["version_date"] is not None else None
                ),
                number=(
                    tuple(
                        int(part) for part in row["version_number"].split(".")
                    )
                    if row["version_number"] is not None else None
                ),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        try:
            timestamp = datetime.fromisoformat(value) if value else None
        except ValueError:
            return None
        if timestamp is None or timestamp.utcoffset() is None:
            return None
        return timestamp

    @classmethod
    def _timestamp(cls, row: sqlite3.Row) -> datetime | None:
        value = (
            row["uploaded_at"]
            if row["source"] == "telegram" else row["modified_at"]
        )
        return cls._parse_timestamp(value) or cls._parse_timestamp(
            row["created_at"]
        )

    @classmethod
    def _compare_incoming(
        cls,
        incoming: AdviceVersion | None,
        modified_at: str | None,
        previous: sqlite3.Row,
    ) -> int | None:
        """Protect a Drive source's newest known content from regressions."""
        previous_version = cls._version(previous)
        timestamp = cls._parse_timestamp(modified_at) or cls._parse_timestamp(
            previous["created_at"]
        )
        previous_timestamp = cls._timestamp(previous)
        if (
            incoming is None
            or previous_version is None
            or (incoming.topic, incoming.place)
            != (previous_version.topic, previous_version.place)
            or timestamp is None
            or previous_timestamp is None
        ):
            return None
        return compare_versions(
            incoming, timestamp, previous_version, previous_timestamp,
        )

    @classmethod
    def _supersede_versions(
        cls,
        connection: sqlite3.Connection,
        version: AdviceVersion | None,
        *,
        reactivate_id: int | None = None,
    ) -> None:
        """Hide proven older advice while preserving deletion and ambiguity."""
        if version is None:
            return
        rows = connection.execute(
            """
                SELECT * FROM documents
                WHERE advice_topic = ? AND advice_place = ? ORDER BY id
            """,
            (version.topic, version.place),
        ).fetchall()
        newer: dict[int, set[int]] = {row["id"]: set() for row in rows}
        groups = {identifier: identifier for identifier in newer}

        def group(identifier: int) -> int:
            while identifier != groups[identifier]:
                groups[identifier] = groups[groups[identifier]]
                identifier = groups[identifier]
            return identifier

        for position, left in enumerate(rows):
            left_version = cls._version(left)
            left_timestamp = cls._timestamp(left)
            if left_version is None or left_timestamp is None:
                continue
            for right in rows[position + 1:]:
                right_version = cls._version(right)
                right_timestamp = cls._timestamp(right)
                if right_version is None or right_timestamp is None:
                    continue
                comparison = compare_versions(
                    left_version, left_timestamp,
                    right_version, right_timestamp,
                )
                if comparison == -1:
                    newer[left["id"]].add(right["id"])
                elif comparison == 1:
                    newer[right["id"]].add(left["id"])
                elif comparison == 0:
                    groups[group(left["id"])] = group(right["id"])

        grouped_newer: dict[int, set[int]] = {
            group(identifier): set() for identifier in newer
        }
        for identifier, replacements in newer.items():
            grouped_newer[group(identifier)].update(
                group(replacement) for replacement in replacements
            )
        if cls._has_version_cycle(grouped_newer):
            # Mixed explicit metadata and clocks can contradict each other.
            # Equal versions share a node, so a strict path between them is
            # also a contradiction. Preserve the current availability.
            return
        for row in rows:
            identifier = row["id"]
            if newer[identifier]:
                # Any proven newer source explains this decision; choosing a
                # pointer does not choose between incomparable active sources.
                replacement = min(newer[identifier])
                connection.execute(
                    "UPDATE documents SET active = 0, superseded_by = ? "
                    "WHERE id = ?",
                    (replacement, identifier),
                )
            elif (
                identifier == reactivate_id
                and row["deleted_at"] is None
                and row["superseded_by"] in newer
                and all(
                    identifier in newer[other["id"]]
                    for other in rows if other["id"] != identifier
                )
            ):
                connection.execute(
                    "UPDATE documents SET active = 1, superseded_by = NULL "
                    "WHERE id = ?",
                    (identifier,),
                )

    @staticmethod
    def _has_version_cycle(newer: dict[int, set[int]]) -> bool:
        visited: set[int] = set()
        visiting: set[int] = set()

        def visit(identifier: int) -> bool:
            if identifier in visiting:
                return True
            if identifier in visited:
                return False
            visiting.add(identifier)
            if any(visit(other) for other in newer[identifier]):
                return True
            visiting.remove(identifier)
            visited.add(identifier)
            return False

        return any(visit(identifier) for identifier in newer)

    @staticmethod
    def _document(row: sqlite3.Row) -> Document:
        return Document(
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
            owner_id=row["owner_id"],
            owner_name=row["owner_name"],
            uploaded_at=row["uploaded_at"],
            deleted_at=row["deleted_at"],
            advice_topic=row["advice_topic"],
            advice_place=row["advice_place"],
            version_date=row["version_date"],
            version_number=row["version_number"],
            superseded_by=row["superseded_by"],
        )

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
