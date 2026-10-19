"""Small factories for documents and mocked Groq responses."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from bot.storage import Document


def make_document(
    document_id: int = 42,
    name: str = "Умови внеску.txt",
    text: str = "Внесок — 17 каштанів.",
    **overrides: Any,
) -> Document:
    document = Document(
        id=document_id,
        source="drive",
        source_id=f"source-{document_id}",
        name=name,
        relative_path=Path("documents") / f"{document_id}.bin",
        text=text,
        modified_at="2026-10-18T12:00:00Z",
        created_at="2026-10-18T12:00:00Z",
        imported_at="2026-10-18T12:00:00Z",
        active=True,
    )
    return replace(document, **overrides)


def completion(
    value: Any, finish_reason: str = "stop",
) -> SimpleNamespace:
    content = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30),
    )


def make_client(create: AsyncMock) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
