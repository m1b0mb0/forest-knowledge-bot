"""Read-only Drive import checks without external HTTP requests."""

import asyncio
import sqlite3
import traceback
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from bot.config import ArchiveSettings
from bot.services import drive
from bot.storage import Store


class Response:
    def __init__(self, payload: Any = None, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.content = self

    async def __aenter__(self) -> "Response":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def json(self) -> Any:
        return self.payload

    async def iter_chunked(self, chunk_size: int) -> AsyncIterator[bytes]:
        for offset in range(0, len(self.payload), chunk_size):
            yield self.payload[offset:offset + chunk_size]


class Session:
    """Provide GET, recording requests and consuming prepared responses."""

    def __init__(self, *responses: Response | Exception) -> None:
        self.responses = deque(responses)
        self.requests: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def get(self, url: str, **kwargs: Any) -> Response:
        self.requests.append((url, kwargs))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def settings(tmp_path: Path) -> ArchiveSettings:
    return ArchiveSettings("folder-1", "test-key", tmp_path)


def document(file_id: str = "file-1", **extra: Any) -> dict[str, Any]:
    return {
        "id": file_id,
        "name": "rules.txt",
        "mimeType": "text/plain",
        "modifiedTime": "2026-10-18T12:00:00Z",
        **extra,
    }


def install_session(monkeypatch: pytest.MonkeyPatch, session: Session) -> None:
    monkeypatch.setattr(
        drive.aiohttp, "ClientSession", lambda **kwargs: session
    )


def test_all_pages_are_imported_using_get_and_reimport_does_not_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    # Parsing is exercised separately; this test checks HTTP/import boundaries.
    monkeypatch.setattr(
        drive, "extract_document_text", lambda data, name: data.decode()
    )

    for index, (content, modified_at) in enumerate(
        [
            (b"old advice", "2026-10-18T12:00:00Z"),
            (b"new advice", "2026-10-18T13:00:00Z"),
        ]
    ):
        responses = [
            Response({"mimeType": drive.FOLDER_MIME}),
            Response(
                {
                    "files": [document(modifiedTime=modified_at)],
                    "nextPageToken": "page-2",
                }
            ),
            Response({"files": [document("file-2")]}),
            Response(content),
        ]
        if index == 0:
            responses.append(Response(b"other advice"))
        session = Session(*responses)
        install_session(monkeypatch, session)

        assert (
            asyncio.run(drive.import_archive(settings(tmp_path), store)) == 2
        )
        assert session.requests[2][1]["params"]["pageToken"] == "page-2"
        assert session.requests[1][1]["params"]["q"] == (
            "'folder-1' in parents and trashed = false"
        )
        assert all(
            request[1]["params"]["key"] == "test-key"
            for request in session.requests
        )
        assert not session.responses
        media_requests = [
            url
            for url, kwargs in session.requests
            if kwargs["params"].get("alt") == "media"
        ]
        assert media_requests == (
            [f"{drive.API_URL}/file-1", f"{drive.API_URL}/file-2"]
            if index == 0 else [f"{drive.API_URL}/file-1"]
        )
        if index == 0:
            first_import = store.list_documents()

    documents = store.list_documents()
    assert len(documents) == 2
    assert documents[0].id == first_import[0].id
    assert documents[0].text == "new advice"
    assert documents[0].modified_at == "2026-10-18T13:00:00Z"
    assert documents[1] == first_import[1]
    assert (
        tmp_path / documents[0].relative_path
    ).read_bytes() == b"new advice"


def test_unchanged_documents_survive_restart_without_download_or_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    initial_session = Session(
        Response({"mimeType": drive.FOLDER_MIME}),
        Response({"files": [document(size="6")]}),
        Response(b"cached"),
    )
    install_session(monkeypatch, initial_session)
    assert asyncio.run(drive.import_archive(settings(tmp_path), store)) == 1
    original = store.list_documents()[0]
    restarted_store = Store(tmp_path)

    def reject_parsing(content: bytes, filename: str) -> str:
        pytest.fail("Unchanged cached documents must not be parsed again.")

    monkeypatch.setattr(drive, "extract_document_text", reject_parsing)
    session = Session(
        Response({"mimeType": drive.FOLDER_MIME}),
        Response({"files": [document(size="6")]}),
    )
    install_session(monkeypatch, session)

    assert (
        asyncio.run(drive.import_archive(settings(tmp_path), restarted_store))
        == 1
    )
    assert len(session.requests) == 2
    assert not session.responses
    assert restarted_store.list_documents() == [original]


@pytest.mark.parametrize(
    "incomplete",
    ["missing-original", "empty-original", "wrong-size-original", "empty-text"],
)
def test_incomplete_cache_is_downloaded_and_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, incomplete: str
) -> None:
    store = Store(tmp_path)
    store.initialize()
    file = document(size="6")
    original_id = store.upsert_drive_document(
        file["id"], file["name"], b"cached", "cached", file["modifiedTime"]
    )
    original = store.list_documents()[0]
    path = tmp_path / original.relative_path
    if incomplete == "missing-original":
        path.unlink()
    elif incomplete == "empty-original":
        path.write_bytes(b"")
    elif incomplete == "wrong-size-original":
        path.write_bytes(b"wrong size")
    else:
        with sqlite3.connect(store.database_path) as connection:
            connection.execute("UPDATE documents SET text = '   '")
    session = Session(
        Response({"mimeType": drive.FOLDER_MIME}),
        Response({"files": [file]}),
        Response(b"cached"),
    )
    install_session(monkeypatch, session)

    assert asyncio.run(drive.import_archive(settings(tmp_path), store)) == 1

    repaired = store.list_documents()[0]
    assert repaired.id == original_id
    assert repaired.text == "cached"
    assert path.read_bytes() == b"cached"
    assert len(session.requests) == 3
    assert session.requests[-1][1]["params"]["alt"] == "media"
    assert not session.responses


def test_absent_modified_time_does_not_establish_cache_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    store.upsert_drive_document(
        "file-1", "rules.txt", b"old", "old", None
    )
    session = Session(
        Response({"mimeType": drive.FOLDER_MIME}),
        Response({"files": [document(modifiedTime=None)]}),
        Response(b"new"),
    )
    install_session(monkeypatch, session)

    assert asyncio.run(drive.import_archive(settings(tmp_path), store)) == 1

    assert store.list_documents()[0].text == "new"
    assert len(session.requests) == 3
    assert not session.responses


def test_renamed_file_updates_its_record_even_with_same_modified_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    item = document(name="renamed.txt")
    original_id = store.upsert_drive_document(
        item["id"], "rules.txt", b"cached", "cached", item["modifiedTime"]
    )
    session = Session(
        Response({"mimeType": drive.FOLDER_MIME}),
        Response({"files": [item]}),
        Response(b"cached"),
    )
    install_session(monkeypatch, session)

    assert asyncio.run(drive.import_archive(settings(tmp_path), store)) == 1

    renamed = store.list_documents()[0]
    assert renamed.id == original_id
    assert renamed.name == "renamed.txt"
    assert len(session.requests) == 3
    assert not session.responses


def test_cached_document_still_requires_drive_download_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    item = document(capabilities={"canDownload": False})
    store.upsert_drive_document(
        item["id"], item["name"], b"cached", "cached", item["modifiedTime"]
    )
    session = Session(
        Response({"mimeType": drive.FOLDER_MIME}),
        Response({"files": [item]}),
    )
    install_session(monkeypatch, session)

    with pytest.raises(drive.DriveImportError, match="cannot be downloaded"):
        asyncio.run(drive.import_archive(settings(tmp_path), store))

    assert len(session.requests) == 2
    assert not session.responses


def test_http_error_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    install_session(monkeypatch, Session(Response(status=403)))

    with pytest.raises(drive.DriveImportError, match="HTTP 403") as error:
        asyncio.run(drive.import_archive(settings(tmp_path), store))

    assert "test-key" not in str(error.value)
    assert store.list_documents() == []


def test_connection_exception_does_not_expose_url_or_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    install_session(
        monkeypatch,
        Session(aiohttp.ClientError("private-url?key=secret-marker")),
    )

    with pytest.raises(drive.DriveImportError) as error:
        asyncio.run(drive.import_archive(settings(tmp_path), store))

    assert "secret-marker" not in str(error.value)
    assert "private-url" not in str(error.value)


@pytest.mark.parametrize(
    "failure",
    [
        Response(
            b"<html>https://private.example/?key=secret-marker</html>",
            status=403,
        ),
        aiohttp.ClientError("https://private.example/?key=secret-marker"),
    ],
    ids=["http-403", "connection-error"],
)
def test_download_error_includes_filename_and_hides_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Response | Exception,
) -> None:
    store = Store(tmp_path)
    store.initialize()
    session = Session(
        Response({"mimeType": drive.FOLDER_MIME}),
        Response({"files": [document()]}),
        failure,
    )
    install_session(monkeypatch, session)

    with pytest.raises(drive.DriveImportError, match="rules.txt") as error:
        asyncio.run(drive.import_archive(settings(tmp_path), store))

    formatted_error = "".join(traceback.format_exception(error.value))
    assert "test-key" not in formatted_error
    assert "secret-marker" not in formatted_error
    assert "private.example" not in formatted_error
    assert len(session.requests) == 3
    assert session.requests[-1][1]["params"]["alt"] == "media"
    assert not session.responses
    assert store.list_documents() == []


def test_non_folder_id_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    install_session(monkeypatch, Session(Response({"mimeType": "text/plain"})))

    with pytest.raises(drive.DriveImportError, match="does not identify"):
        asyncio.run(drive.import_archive(settings(tmp_path), store))


def test_oversized_download_is_rejected_before_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    monkeypatch.setattr(drive, "MAX_DOCUMENT_BYTES", 4)
    install_session(
        monkeypatch,
        Session(
            Response({"mimeType": drive.FOLDER_MIME}),
            Response({"files": [document()]}),
            Response(b"too large"),
        ),
    )

    with pytest.raises(drive.DriveImportError, match="Cannot read"):
        asyncio.run(drive.import_archive(settings(tmp_path), store))

    assert store.list_documents() == []


def test_empty_archive_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.initialize()
    install_session(
        monkeypatch,
        Session(
            Response({"mimeType": drive.FOLDER_MIME}),
            Response({"files": []}),
        ),
    )

    with pytest.raises(drive.DriveImportError, match="no supported"):
        asyncio.run(drive.import_archive(settings(tmp_path), store))
