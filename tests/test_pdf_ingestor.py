import io

import pymupdf
import pytest

import backend.ingestors.pdf_ingestor as m
from backend.filestore.base_filestore import IngestableFile as _IngestableFile
from backend.ingestors.base_ingestor import BaseIngestor, IngestionError, Metadata
from backend.ingestors.pdf_ingestor import PdfIngestor


def IngestableFile(file_obj, name=None, file_id=1):
    return _IngestableFile(file_obj, name=name, file_id=file_id)


"""Comprehensive unit tests for PdfIngestor.

Real PDFs are generated in-memory with pymupdf (text pages, scanned/blank
pages, mixed documents) so extraction is exercised end-to-end; the OCR
fallback (rapidocr) is faked at the module boundary. Coverage includes page
markers and ordering, per-page OCR fallback, error wrapping/chaining for
corrupt/empty files, IngestionError passthrough, file-handle handling
(bytes, str, failing reads), and the format-registration registry. The
isolated_registry fixture snapshots/restores the global registry so tests
never leak state.
"""


def make_text_pdf(path, pages):
    """Build a PDF whose pages contain real digital text."""
    doc = pymupdf.open()
    for content in pages:
        page = doc.new_page()
        page.insert_text((72, 72), content)
    doc.save(path)
    doc.close()


def make_image_pdf(path, num_pages=1):
    """Build a PDF whose pages are blank (no text layer), forcing OCR."""
    doc = pymupdf.open()
    for _ in range(num_pages):
        page = doc.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 100), False)
        page.insert_image(page.rect, stream=pix.tobytes("png"))
    doc.save(path)
    doc.close()


def make_mixed_pdf(path):
    """Build a PDF with a digital-text page followed by a blank page."""
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "digital text page")
    page2 = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 100), False)
    page2.insert_image(page2.rect, stream=pix.tobytes("png"))
    doc.save(path)
    doc.close()


class FailingReadBytesIO(io.BytesIO):
    """BytesIO whose read() always fails."""

    def read(self, size=-1):
        raise OSError("disk read failed")


class FakeOtherIngestor(BaseIngestor):
    """A different ingestor class, used to provoke registry conflicts."""

    def __init__(self, accepted_formats=None):
        super().__init__("other", accepted_formats, "other")

    def extract_text(self, file):
        raise NotImplementedError


@pytest.fixture
def isolated_registry():
    """Snapshot the global ingestor registry and restore it after the test,
    so format registrations never leak across tests."""
    saved_formats = set(BaseIngestor.all_formats)
    saved_map = dict(BaseIngestor.ingestor_map)
    yield
    BaseIngestor.all_formats.clear()
    BaseIngestor.all_formats.update(saved_formats)
    BaseIngestor.ingestor_map.clear()
    BaseIngestor.ingestor_map.update(saved_map)


def make_ingestor(accepted_formats):
    """Build a PdfIngestor registered for the given formats."""
    return PdfIngestor(accepted_formats=accepted_formats)


def make_file(tmp_path, name, data=None):
    """Write `data` (or a generated PDF) to a temp file and return an open
    read handle."""
    path = tmp_path / name
    if data is None:
        make_text_pdf(path, ["content"])
    else:
        path.write_bytes(data)
    return open(path, "rb")


class TestInit:
    def test_defaults(self, isolated_registry):
        """A bare PdfIngestor has type 'pdf', name 'PDF' and no formats."""
        ingestor = make_ingestor(None)
        assert ingestor.type == "pdf"
        assert ingestor.name == "PDF"
        assert ingestor.accepted_formats == []

    def test_registers_formats_in_global_registry(self, isolated_registry):
        """The constructor registers every format in all_formats and
        ingestor_map."""
        ingestor = make_ingestor(["pdf"])
        assert "pdf" in BaseIngestor.all_formats
        assert BaseIngestor.ingestor_map["pdf"] is ingestor

    def test_same_class_re_registration_is_idempotent(self, isolated_registry):
        """A second PdfIngestor instance may claim the same formats."""
        first = make_ingestor(["pdf"])
        second = make_ingestor(["pdf"])
        assert BaseIngestor.ingestor_map["pdf"] is second
        assert first is not second

    def test_cross_class_conflict_raises_ingestion_error(self, isolated_registry):
        """A different class claiming the pdf format raises IngestionError
        naming the format and the owning ingestor."""
        FakeOtherIngestor(accepted_formats=["pdf-conflict"])
        with pytest.raises(IngestionError, match="already registered by") as exc:
            make_ingestor(["pdf-conflict"])
        assert "pdf-conflict" in str(exc.value)

    def test_unexpected_init_error_wraps_as_ingestion_error(
        self, isolated_registry, monkeypatch
    ):
        """A non-IngestionError during init surfaces as a chained
        IngestionError, not the raw exception."""

        def boom(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(m.BaseIngestor, "__init__", boom)
        with pytest.raises(
            IngestionError, match="Failed to initialize PdfIngestor"
        ) as exc:
            make_ingestor(["pdf"])
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_ingestion_error_from_init_propagates_unchanged(self, isolated_registry):
        """A registration conflict during init propagates unwrapped."""
        FakeOtherIngestor(accepted_formats=["conflict-pdf"])
        with pytest.raises(IngestionError, match="already registered") as exc:
            make_ingestor(["conflict-pdf"])
        assert exc.value.__cause__ is None


class TestExtractText:
    def test_text_pdf_extracts_all_pages_with_markers(
        self, tmp_path, isolated_registry
    ):
        """Every page appears with a [Page N] marker in document order."""
        pdf_path = tmp_path / "sample.pdf"
        make_text_pdf(pdf_path, ["Hello from page one", "Second page content"])

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            text, metadata = ingestor.extract_text(IngestableFile(f))

        assert "[Page 1]" in text
        assert "[Page 2]" in text
        assert text.index("[Page 1]") < text.index("[Page 2]")
        assert "Hello from page one" in text
        assert "Second page content" in text

    def test_returns_string_and_metadata_types(self, tmp_path, isolated_registry):
        """extract_text returns a str and a Metadata instance."""
        pdf_path = tmp_path / "sample.pdf"
        make_text_pdf(pdf_path, ["content"])

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            text, metadata = ingestor.extract_text(IngestableFile(f))

        assert type(text) is str
        assert type(metadata) is Metadata

    def test_metadata_values(self, tmp_path, isolated_registry):
        """Metadata carries the file name, extension and ingestor type."""
        pdf_path = tmp_path / "docs.pdf"
        make_text_pdf(pdf_path, ["content"])

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            text, metadata = ingestor.extract_text(IngestableFile(f, "docs.pdf"))

        assert metadata.file_name == "docs.pdf"
        assert metadata.extension == "pdf"
        assert metadata.type == "pdf"
        assert isinstance(metadata.created_at_ts, float)

    def test_scanned_pdf_uses_ocr_fallback(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """Pages with no text layer are OCR'd and included with markers."""
        pdf_path = tmp_path / "scan.pdf"
        make_image_pdf(pdf_path)
        monkeypatch.setattr(m, "ocr_bytes", lambda b: "scanned ocr text")

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            text, _ = ingestor.extract_text(IngestableFile(f))

        assert "[Page 1]" in text
        assert "scanned ocr text" in text

    def test_mixed_text_and_scanned_pages(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """Digital-text pages keep their text and blank pages get OCR'd."""
        pdf_path = tmp_path / "mixed.pdf"
        make_mixed_pdf(pdf_path)
        monkeypatch.setattr(m, "ocr_bytes", lambda b: "image page ocr")

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            text, _ = ingestor.extract_text(IngestableFile(f))

        assert "digital text page" in text
        assert "image page ocr" in text
        assert "image page ocr" not in text.split("digital text page")[0]

    def test_ocr_failure_wraps_in_parse_error(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """An OCR crash on a blank page surfaces as a chained IngestionError."""
        pdf_path = tmp_path / "scan.pdf"
        make_image_pdf(pdf_path)
        monkeypatch.setattr(
            m,
            "ocr_bytes",
            lambda b: (_ for _ in ()).throw(RuntimeError("ocr engine crash")),
        )

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            with pytest.raises(IngestionError, match="could not be parsed") as exc:
                ingestor.extract_text(IngestableFile(f))
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_ocr_ingestion_error_propagates_unchanged(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """An IngestionError from the OCR layer is not re-wrapped."""
        pdf_path = tmp_path / "scan.pdf"
        make_image_pdf(pdf_path)
        monkeypatch.setattr(
            m, "ocr_bytes", lambda b: (_ for _ in ()).throw(IngestionError("ocr boom"))
        )

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            with pytest.raises(IngestionError, match="ocr boom") as exc:
                ingestor.extract_text(IngestableFile(f))
        assert exc.value.__cause__ is None

    def test_corrupt_pdf_raises_chained_ingestion_error(
        self, tmp_path, isolated_registry
    ):
        """Garbage bytes fail parsing with a chained IngestionError."""
        with make_file(tmp_path, "broken.pdf", data=b"this is not a pdf at all") as f:
            ingestor = make_ingestor(["pdf"])
            with pytest.raises(IngestionError, match="could not be parsed") as exc:
                ingestor.extract_text(IngestableFile(f))
        assert isinstance(exc.value.__cause__, Exception)

    def test_empty_pdf_raises_ingestion_error(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """A page with neither text nor OCR results fails extraction."""
        pdf_path = tmp_path / "empty.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.save(pdf_path)
        doc.close()
        monkeypatch.setattr(m, "ocr_bytes", lambda b: "")

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            with pytest.raises(IngestionError, match="produced no text"):
                ingestor.extract_text(IngestableFile(f))

    def test_scanned_pdf_without_ocr_result_raises(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """A scanned page whose OCR returns nothing fails extraction."""
        pdf_path = tmp_path / "noscantext.pdf"
        make_image_pdf(pdf_path)
        monkeypatch.setattr(m, "ocr_bytes", lambda b: "")

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            with pytest.raises(IngestionError, match="produced no text"):
                ingestor.extract_text(IngestableFile(f))

    def test_accepts_bytes_io_buffer(self, tmp_path, isolated_registry):
        """Extraction works from an in-memory BytesIO buffer, not just files."""
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "from buffer")
        buffer = io.BytesIO(doc.tobytes())
        doc.close()

        ingestor = make_ingestor(["pdf"])
        text, _ = ingestor.extract_text(IngestableFile(buffer, "buffer.pdf"))

        assert "from buffer" in text

    def test_accepts_str_io_buffer(self, tmp_path, isolated_registry):
        """A text-mode StringIO handle (str data) is encoded before parsing."""
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "from string buffer")
        buffer = io.StringIO(doc.tobytes().decode("latin-1"))
        doc.close()

        ingestor = make_ingestor(["pdf"])
        text, _ = ingestor.extract_text(IngestableFile(buffer, "buffer.pdf"))

        assert "from string buffer" in text

    def test_resets_file_pointer_between_calls(self, tmp_path, isolated_registry):
        """seek(0) before reading lets the same file be parsed repeatedly."""
        pdf_path = tmp_path / "repeat.pdf"
        make_text_pdf(pdf_path, ["repeatable"])

        with open(pdf_path, "rb") as f:
            ingestor = make_ingestor(["pdf"])
            ingestable = IngestableFile(f)
            first, _ = ingestor.extract_text(ingestable)
            second, _ = ingestor.extract_text(ingestable)

        assert first == second
        assert "repeatable" in second

    def test_file_read_failure_wraps_as_ingestion_error(self, isolated_registry):
        """A read failure surfaces as a chained IngestionError, not OSError."""
        ingestor = make_ingestor(["pdf"])
        ingestable = IngestableFile(FailingReadBytesIO(b"x"), "broken.pdf")

        with pytest.raises(IngestionError, match="could not be parsed") as exc:
            ingestor.extract_text(ingestable)
        assert isinstance(exc.value.__cause__, OSError)
