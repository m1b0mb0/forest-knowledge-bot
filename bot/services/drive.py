"""Import a publicly shared Drive folder using read-only HTTP requests."""

import asyncio
import logging
from pathlib import Path
from stat import S_ISREG
from typing import Any

import aiohttp

from bot.config import ArchiveSettings
from bot.services.documents import MAX_DOCUMENT_BYTES, extract_document_text
from bot.storage import Document, Store

logger = logging.getLogger(__name__)
API_URL = "https://www.googleapis.com/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"
FORMATS = {
    "application/pdf": ".pdf",
    (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    ): ".docx",
    "text/plain": ".txt",
}


class DriveImportError(Exception):
    """A safe-to-log import error without credential-bearing URLs."""


async def import_archive(settings: ArchiveSettings, store: Store) -> int:
    """Import supported direct children, reusing unchanged local documents.

    Import completes before polling starts. An unreadable supported file stops
    startup rather than silently serving an incomplete archive.
    """
    stored_documents = await asyncio.to_thread(store.list_documents)
    cached = {
        document.source_id: document
        for document in stored_documents
        if document.source == "drive"
    }
    imported = downloaded = reused = 0
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        files = await _list_files(session, settings)
        for item in files:
            extension = FORMATS.get(item.get("mimeType"))
            if extension is None:
                continue
            file_id, name = item["id"], item["name"]
            if item.get("capabilities", {}).get("canDownload") is False:
                raise DriveImportError(
                    f"Archive file {name!r} cannot be downloaded."
                )
            if _can_reuse_document(cached.get(file_id), item, store.data_dir):
                imported += 1
                reused += 1
                logger.debug("Reused archive document %r", name)
                continue
            logger.info("Importing archive document %r", name)
            try:
                content = await _download_file(session, settings, item)
                text = await asyncio.to_thread(
                    extract_document_text, content, f"document{extension}"
                )
            except DriveImportError as error:
                raise DriveImportError(
                    f"Cannot import archive file {name!r}: {error}"
                ) from None
            except ValueError as error:
                raise DriveImportError(
                    f"Cannot read archive file {name!r}: {error}"
                ) from None
            await asyncio.to_thread(
                store.upsert_drive_document,
                file_id,
                name,
                content,
                text,
                item.get("modifiedTime"),
            )
            imported += 1
            downloaded += 1

    if not imported:
        raise DriveImportError(
            "The archive has no supported PDF, DOCX or TXT files."
        )
    logger.info(
        "Archive import complete: %s downloaded, %s reused.",
        downloaded,
        reused,
    )
    return imported


def _can_reuse_document(
    document: Document | None, item: dict[str, Any], data_dir: Path
) -> bool:
    """Require matching metadata, extracted text, and a complete original."""
    modified_at = item.get("modifiedTime")
    if (
        document is None
        or not modified_at
        or modified_at != document.modified_at
        or item["name"] != document.name
        or not document.text.strip()
    ):
        return False
    try:
        original = (data_dir / document.relative_path).stat()
        if not S_ISREG(original.st_mode) or original.st_size <= 0:
            return False
        if item.get("size") is not None:
            return original.st_size == int(item["size"])
    except (OSError, TypeError, ValueError):
        return False
    return True


async def _list_files(
    session: aiohttp.ClientSession, settings: ArchiveSettings
) -> list[dict[str, Any]]:
    folder = await _get_json(
        session,
        f"{API_URL}/{settings.folder_id}",
        {
            "key": settings.api_key,
            "fields": "id,mimeType",
            "supportsAllDrives": "true",
        },
    )
    if folder.get("mimeType") != FOLDER_MIME:
        raise DriveImportError(
            "DRIVE_FOLDER_ID does not identify a Drive folder."
        )

    files: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params = {
            "key": settings.api_key,
            "q": f"'{settings.folder_id}' in parents and trashed = false",
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime,"
            "capabilities(canDownload))",
            "pageSize": "100",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if page_token:
            params["pageToken"] = page_token
        page = await _get_json(session, API_URL, params)
        files.extend(page.get("files", []))
        page_token = page.get("nextPageToken")
        if not page_token:
            return files


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, str],
) -> dict[str, Any]:
    try:
        async with session.get(url, params=params) as response:
            _check_status(response.status)
            payload = await response.json()
    except (aiohttp.ClientError, TimeoutError, ValueError):
        raise DriveImportError(
            "Unable to read Google Drive metadata."
        ) from None
    if not isinstance(payload, dict):
        raise DriveImportError("Google Drive returned invalid metadata.")
    return payload


async def _download_file(
    session: aiohttp.ClientSession,
    settings: ArchiveSettings,
    item: dict[str, Any],
) -> bytes:
    if int(item.get("size", 0)) > MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds the size limit.")
    try:
        async with session.get(
            f"{API_URL}/{item['id']}",
            params={
                "key": settings.api_key,
                "alt": "media",
                "supportsAllDrives": "true",
            },
        ) as response:
            _check_status(response.status)
            content = bytearray()
            async for chunk in response.content.iter_chunked(64 * 1024):
                content.extend(chunk)
                if len(content) > MAX_DOCUMENT_BYTES:
                    raise ValueError("Document exceeds the size limit.")
            return bytes(content)
    except (aiohttp.ClientError, TimeoutError):
        raise DriveImportError(
            "Unable to download an archive document."
        ) from None


def _check_status(status: int) -> None:
    if status != 200:
        raise DriveImportError(
            f"Google Drive returned HTTP {status}. "
            "Check public folder sharing, "
            "the folder ID, Drive API activation and API key restrictions."
        )
