"""Extract text from supported documents without executing their content."""

import os
from collections.abc import Iterable
from io import BytesIO
from math import isfinite
from pathlib import Path, PurePath

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_OCR_IMAGE_PIXELS = 3500
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
    parser_error = "The PDF is damaged or unsupported."
    try:
        reader = pymupdf.open(stream=content, filetype="pdf")
    except Exception:
        raise ValueError(parser_error) from None

    with reader:
        try:
            is_pdf = reader.is_pdf
            encrypted = reader.is_encrypted or bool(
                (reader.metadata or {}).get("encryption")
            )
            page_count = reader.page_count
        except Exception:
            raise ValueError(parser_error) from None
        if not is_pdf:
            raise ValueError(parser_error)
        if encrypted:
            raise ValueError("The PDF is encrypted and unsupported.")
        if page_count > MAX_PDF_PAGES:
            raise ValueError("The PDF exceeds the page limit.")

        pages: list[str] = []
        for number in range(page_count):
            try:
                page = reader.load_page(number)
                text = page.get_text("text", sort=True).strip()
            except Exception:
                raise ValueError(parser_error) from None
            if not text:
                text = _ocr_pdf_page(page)
            pages.append(text)

    return "\n\n".join(pages)


def _ocr_pdf_page(page: pymupdf.Page) -> str:
    """Recognize one page in memory using PyMuPDF's built-in OCR engine."""
    tessdata_value = os.getenv("TESSDATA_PREFIX", "").strip()
    tessdata = Path(tessdata_value)
    if not tessdata_value or not all(
        (tessdata / f"{language}.traineddata").is_file()
        for language in ("ukr", "eng")
    ):
        raise ValueError(
            "PDF OCR requires Ukrainian and English language data. "
            "Run with Docker Compose or set TESSDATA_PREFIX."
        )

    longest_side = max(page.rect.width, page.rect.height)
    if not isfinite(longest_side) or longest_side <= 0:
        raise ValueError("The PDF page dimensions are invalid for OCR.")
    dpi = min(300, int(MAX_OCR_IMAGE_PIXELS * 72 / longest_side))
    if dpi < 1:
        raise ValueError("The PDF page exceeds the OCR image size limit.")
    try:
        textpage = page.get_textpage_ocr(
            language="ukr+eng", dpi=dpi, full=True, tessdata=str(tessdata)
        )
        return page.get_text("text", textpage=textpage, sort=True).strip()
    except Exception:
        raise ValueError(
            "PDF OCR failed. Check the document and OCR language data."
        ) from None


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
