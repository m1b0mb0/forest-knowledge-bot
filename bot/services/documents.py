"""Extract text from supported documents without executing their content."""

from collections.abc import Iterable
from io import BytesIO
from pathlib import PurePath

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})


def extract_document_text(content: bytes, filename: str) -> str:
    """Extract text or reject unsupported, empty or unreadable input."""
    extension = PurePath(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported document formats are PDF, DOCX and TXT.")
    if not content:
        raise ValueError("The document is empty.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError("The document exceeds the 10 MiB size limit.")

    if extension == ".txt":
        text = _extract_txt(content)
    elif extension == ".pdf":
        text = _extract_pdf(content)
    else:
        text = _extract_docx(content)

    text = text.strip()
    if not text:
        raise ValueError("The document contains no extractable text.")
    return text


def _extract_pdf(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF documents are unsupported.")
        text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except Exception as error:
        # Parser failures vary with the damaged PDF structure.
        raise ValueError(
            "The PDF is damaged, encrypted or unsupported."
        ) from error
    if not text.strip():
        raise ValueError(
            "The PDF contains no extractable text."
        )
    return text


def _extract_docx(content: bytes) -> str:
    try:
        document = Document(BytesIO(content))
        return _extract_docx_blocks(document.iter_inner_content())
    except Exception as error:
        raise ValueError("The DOCX is damaged or unsupported.") from error


def _extract_docx_blocks(blocks: Iterable[Paragraph | Table]) -> str:
    """Keep body order, including tables nested within table cells."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, Paragraph):
            parts.append(block.text)
        elif isinstance(block, Table):
            for row in block.rows:
                parts.append(
                    "\t".join(
                        _extract_docx_blocks(cell.iter_inner_content())
                        for cell in row.cells
                    )
                )
    return "\n".join(parts)


def _extract_txt(content: bytes) -> str:
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as e:
        raise ValueError("TXT document must use UTF-8 encoding.") from e

    if not text:
        raise ValueError("File is empty.")

    return text
