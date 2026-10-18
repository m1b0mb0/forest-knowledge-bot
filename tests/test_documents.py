"""Document parsing tests using generated, non-private documents."""

import os
from io import BytesIO
from pathlib import Path

import pymupdf
import pytest
from docx import Document

from bot.services import documents
from bot.services.documents import MAX_DOCUMENT_BYTES, extract_document_text


def make_pdf(*pages: str) -> bytes:
    """Create small text PDFs without adding a PDF-generation dependency."""
    with pymupdf.open() as document:
        for text in pages:
            page = document.new_page(width=500, height=300)
            page.insert_text((30, 60), text, fontsize=20)
        return document.tobytes()


def make_image_only_pdf(text: str) -> bytes:
    """Rasterize synthetic text into a PDF with no embedded text layer."""
    with pymupdf.open(stream=make_pdf(text), filetype="pdf") as source:
        pixmap = source[0].get_pixmap(dpi=150)
        with pymupdf.open() as scanned:
            page = scanned.new_page(width=500, height=300)
            page.insert_image(page.rect, pixmap=pixmap)
            return scanned.tobytes()


def ocr_language_data_available() -> bool:
    prefix = os.getenv("TESSDATA_PREFIX")
    if not prefix:
        return False
    return all(
        (Path(prefix) / f"{language}.traineddata").is_file()
        for language in ("ukr", "eng")
    )


@pytest.fixture
def fake_ocr_language_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Provide filenames for mocked OCR without downloading language data."""
    for language in ("ukr", "eng"):
        (tmp_path / f"{language}.traineddata").write_bytes(b"")
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    return tmp_path


def test_pdf_preserves_text_without_running_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_ocr(*args: object) -> str:
        pytest.fail("Native PDF text must not require OCR.")

    monkeypatch.setattr(documents, "_ocr_pdf_page", unexpected_ocr)
    text = extract_document_text(
        make_pdf("First advice", "Second advice"), "a.PDF"
    )

    assert "First advice" in text
    assert "Second advice" in text
    assert text.index("First advice") < text.index("Second advice")


def test_docx_preserves_paragraph_and_table_order() -> None:
    document = Document()
    document.add_paragraph("Початок")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Місце"
    table.cell(0, 1).text = "Порада"
    document.add_paragraph("Завершення")
    output = BytesIO()
    document.save(output)

    assert extract_document_text(output.getvalue(), "a.docx") == (
        "Початок\nМісце\tПорада\nЗавершення"
    )


def test_docx_preserves_nested_table_text_and_cell_block_order() -> None:
    document = Document()
    document.add_paragraph("Document before")
    outer = document.add_table(rows=1, cols=2)
    cell = outer.cell(0, 0)
    cell.text = "Outer before"
    nested = cell.add_table(rows=1, cols=2)
    nested.cell(0, 0).text = "Nested left"
    nested.cell(0, 1).text = "Nested right"
    cell.paragraphs[-1].text = "Outer after"
    outer.cell(0, 1).text = "Second outer cell"
    document.add_paragraph("Document after")
    output = BytesIO()
    document.save(output)

    assert extract_document_text(output.getvalue(), "nested.docx") == (
        "Document before\nOuter before\nNested left\tNested right\n"
        "Outer after\tSecond outer cell\nDocument after"
    )


@pytest.mark.parametrize("filename", ["a.pdf", "a.docx", "a.txt"])
def test_empty_file_is_rejected(filename: str) -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_document_text(b"", filename)


@pytest.mark.parametrize("filename", ["a.pdf", "a.docx"])
def test_damaged_file_is_rejected(filename: str) -> None:
    with pytest.raises(ValueError, match="damaged"):
        extract_document_text(b"not a document", filename)


def test_pdf_parser_value_error_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_pdf(*args: object, **kwargs: object) -> None:
        raise ValueError("Internal parser detail")

    monkeypatch.setattr(documents.pymupdf, "open", invalid_pdf)

    with pytest.raises(ValueError, match="damaged") as error:
        extract_document_text(b"invalid PDF", "invalid.pdf")

    assert "Internal parser detail" not in str(error.value)


def test_image_renamed_to_pdf_is_rejected() -> None:
    with pymupdf.open(stream=make_pdf("Image"), filetype="pdf") as document:
        image = document[0].get_pixmap().tobytes("png")

    with pytest.raises(ValueError, match="damaged"):
        extract_document_text(image, "renamed.pdf")


def test_pdf_without_text_after_ocr_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(documents, "_ocr_pdf_page", lambda page: " \n\t")

    with pytest.raises(ValueError, match="no extractable text"):
        extract_document_text(make_pdf(""), "scan.pdf")


def test_pdf_page_limit_is_checked_before_ocr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_ocr(*args: object) -> str:
        pytest.fail("Over-limit PDFs must be rejected before OCR.")

    monkeypatch.setattr(documents, "MAX_PDF_PAGES", 1)
    monkeypatch.setattr(documents, "_ocr_pdf_page", unexpected_ocr)

    with pytest.raises(ValueError, match="page limit"):
        extract_document_text(make_pdf("", ""), "long.pdf")


def test_ocr_error_is_not_reported_as_a_damaged_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_ocr(*args: object) -> str:
        raise ValueError("PDF OCR is unavailable.")

    monkeypatch.setattr(documents, "_ocr_pdf_page", failed_ocr)

    with pytest.raises(ValueError, match="OCR is unavailable"):
        extract_document_text(make_pdf(""), "scan.pdf")


def test_mixed_pdf_keeps_native_and_recognized_pages_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pymupdf.open() as combined:
        for content in (
            make_pdf("Before scan"),
            make_image_only_pdf("Scanned advice"),
            make_pdf("After scan"),
        ):
            with pymupdf.open(stream=content, filetype="pdf") as part:
                combined.insert_pdf(part)
        content = combined.tobytes()

    called_pages: list[int] = []

    def recognize(page: pymupdf.Page) -> str:
        assert not page.get_text("text").strip()
        called_pages.append(page.number)
        return "Розпізнаний текст"

    monkeypatch.setattr(documents, "_ocr_pdf_page", recognize)

    text = extract_document_text(content, "mixed.pdf")

    assert text == "Before scan\n\nРозпізнаний текст\n\nAfter scan"
    assert called_pages == [1]


def test_ocr_engine_failure_has_a_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_ocr_language_data: Path,
) -> None:
    def fail_ocr(*args: object, **kwargs: object) -> None:
        raise RuntimeError("private diagnostic detail")

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", fail_ocr)

    with pytest.raises(ValueError, match="OCR") as error:
        extract_document_text(make_pdf(""), "scan.pdf")

    assert "private diagnostic detail" not in str(error.value)


@pytest.mark.parametrize("missing", ["prefix", "ukr", "eng"])
def test_ocr_language_data_configuration_is_checked_before_engine(
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
    fake_ocr_language_data: Path,
) -> None:
    if missing == "prefix":
        monkeypatch.delenv("TESSDATA_PREFIX")
    else:
        (fake_ocr_language_data / f"{missing}.traineddata").unlink()

    def unexpected_ocr(*args: object, **kwargs: object) -> None:
        pytest.fail("Incomplete language data must be rejected before OCR.")

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", unexpected_ocr)

    with pytest.raises(ValueError, match="OCR requires.*language data"):
        extract_document_text(make_pdf(""), "scan.pdf")


@pytest.mark.parametrize("page_size", [(500, 300), (1800, 2400)])
def test_ocr_uses_recognized_text_and_both_languages(
    page_size: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
    fake_ocr_language_data: Path,
) -> None:
    recognized_page = object()
    original_get_text = pymupdf.Page.get_text

    def recognize(page: pymupdf.Page, **kwargs: object) -> object:
        assert kwargs["language"] == "ukr+eng"
        assert kwargs["full"] is True
        assert kwargs["tessdata"] == str(fake_ocr_language_data)
        dpi = kwargs["dpi"]
        assert isinstance(dpi, int) and 0 < dpi <= 300
        longest_side = max(page.rect.width, page.rect.height)
        assert dpi * longest_side / 72 <= 3500
        return recognized_page

    def get_text(
        page: pymupdf.Page, option: str = "text", **kwargs: object
    ) -> str:
        if kwargs.get("textpage") is recognized_page:
            return "Recognized advice"
        return original_get_text(page, option, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", recognize)
    monkeypatch.setattr(pymupdf.Page, "get_text", get_text)

    with pymupdf.open() as document:
        document.new_page(width=page_size[0], height=page_size[1])
        content = document.tobytes()

    assert extract_document_text(content, "scan.pdf") == "Recognized advice"


@pytest.mark.skipif(
    not ocr_language_data_available(),
    reason=(
        "Real OCR requires Ukrainian and English language data; run in Docker."
    ),
)
def test_real_ocr_reads_generated_image_only_pdf() -> None:
    scanned = make_image_only_pdf("Forest archive document")

    with pymupdf.open(stream=scanned, filetype="pdf") as document:
        assert not document[0].get_text("text").strip()
    assert "forest archive document" in extract_document_text(
        scanned, "scan.pdf"
    ).lower()


def test_empty_docx_is_rejected() -> None:
    output = BytesIO()
    Document().save(output)

    with pytest.raises(ValueError, match="no extractable text"):
        extract_document_text(output.getvalue(), "empty.docx")


@pytest.mark.parametrize("user_password", ["private-password", ""])
def test_encrypted_pdf_is_rejected(user_password: str) -> None:
    with pymupdf.open() as document:
        document.new_page(width=100, height=100)
        encrypted = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw=user_password,
        )

    with pytest.raises(ValueError, match="encrypted"):
        extract_document_text(encrypted, "encrypted.pdf")


def test_unsupported_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="Supported"):
        extract_document_text(b"legacy document", "old.doc")


def test_oversized_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="size limit"):
        extract_document_text(b"x" * (MAX_DOCUMENT_BYTES + 1), "large.txt")


@pytest.mark.parametrize(
    ("content", "filename", "expected"),
    [
        pytest.param(
            "Правила лісу".encode("utf-8"),
            "rules.TXT",
            "Правила лісу",
            id="ukrainian-uppercase-extension",
        ),
        pytest.param(
            b"\xef\xbb\xbf" + " \tПравила лісу\r\n".encode("utf-8"),
            "rules.txt",
            "Правила лісу",
            id="bom-and-outer-whitespace",
        ),
        pytest.param(
            "  Перше\r\n\r\nДруге\tпункт \t".encode("utf-8"),
            "rules.txt",
            "Перше\r\n\r\nДруге\tпункт",
            id="internal-formatting",
        ),
    ],
)
def test_txt_accepts_utf8_and_preserves_internal_formatting(
    content: bytes, filename: str, expected: str
) -> None:
    assert extract_document_text(content, filename) == expected


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b" \t\r\n", id="whitespace-only"),
        pytest.param(b"\xef\xbb\xbf", id="bom-only"),
        pytest.param(b"\xef\xbb\xbf \t\r\n", id="bom-and-whitespace"),
    ],
)
def test_txt_without_text_is_rejected(content: bytes) -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_document_text(content, "rules.txt")


def test_txt_invalid_utf8_after_valid_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        extract_document_text(b"Valid prefix\n\xff", "rules.txt")
