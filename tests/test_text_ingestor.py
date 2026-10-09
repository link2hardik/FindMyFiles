import io

import pytest

from backend.filestore.base_filestore import IngestableFile as _IngestableFile
from backend.ingestors.base_ingestor import BaseIngestor, IngestionError, Metadata
from backend.ingestors.text_ingestor import TextIngestor


def IngestableFile(file_obj, name=None, file_id=1):
    return _IngestableFile(file_obj, name=name, file_id=file_id)


"""Comprehensive unit tests for TextIngestor.

Covers text extraction behavior (plain text, markdown, multiple accepted
formats, binary decode, empty files), metadata construction, the
extension-mismatch contract, error wrapping/chaining in extract_text and
__init__, and the class-level format registration registry (idempotent
same-class re-registration, cross-class conflicts, registry population).
The isolated_registry fixture snapshots/restores the global registry so
tests never leak state into each other or into the rest of the suite.
"""


class FakeAudioIngestor(BaseIngestor):
    """A BaseIngestor subclass used to simulate cross-class format conflicts."""

    def extract_text(self, file):
        return "", self.extract_metadata(file)


class FailingBytesIO(io.BytesIO):
    """BytesIO whose read() always fails, to exercise the read-error path."""

    def read(self, *args, **kwargs):
        raise OSError("read failed")


class FailingSeekIO(io.BytesIO):
    """BytesIO whose seek() always fails, to exercise the seek-error path."""

    def seek(self, *args, **kwargs):
        raise OSError("seek failed")


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
    """Build a TextIngestor registered for the given formats."""
    return TextIngestor(accepted_formats=accepted_formats)


def make_file(tmp_path, name, data, binary=True):
    """Write `data` to a temp file and return an open read handle."""
    path = tmp_path / name
    if binary:
        path.write_bytes(data)
    else:
        path.write_text(data)
    return open(path, "rb" if binary else "r")


class TestExtractText:
    def test_extracts_plain_text_and_metadata(self, tmp_path, isolated_registry):
        """extract_text returns the exact file contents as str plus a valid
        Metadata instance."""
        with make_file(tmp_path, "test.txt", "Hello World", binary=False) as f:
            ingestor = make_ingestor(["txt"])
            text, metadata = ingestor.extract_text(IngestableFile(f, "test.txt"))

        assert text == "Hello World"
        assert type(text) is str
        assert type(metadata) is Metadata

    def test_extracts_with_any_accepted_format(self, tmp_path, isolated_registry):
        """A file whose extension is one of several accepted formats is
        extracted, and metadata records the actual extension."""
        with make_file(tmp_path, "notes.md", "# Title", binary=False) as f:
            ingestor = make_ingestor(["md", "txt"])
            text, metadata = ingestor.extract_text(IngestableFile(f, "notes.md"))

        assert text == "# Title"
        assert metadata.extension == "md"

    def test_extracts_from_bytes_io(self, isolated_registry):
        """extract_text works on an in-memory BytesIO buffer."""
        buffer = io.BytesIO(b"from buffer")
        ingestor = make_ingestor(["txt"])

        text, metadata = ingestor.extract_text(IngestableFile(buffer, "buffer.txt"))

        assert text == "from buffer"
        assert metadata.file_name == "buffer.txt"

    def test_decodes_utf8_bytes(self, tmp_path, isolated_registry):
        """Binary-mode file objects are decoded as UTF-8 before returning."""
        with make_file(tmp_path, "uni.txt", "héllo wörld".encode("utf-8")) as f:
            ingestor = make_ingestor(["txt"])
            text, metadata = ingestor.extract_text(IngestableFile(f, "uni.txt"))

        assert text == "héllo wörld"

    def test_invalid_utf8_bytes_are_ignored(self, tmp_path, isolated_registry):
        """Undecodable bytes are dropped (errors='ignore'), not fatal."""
        with make_file(tmp_path, "bin.txt", b"\xff\xfeHello") as f:
            ingestor = make_ingestor(["txt"])
            text, metadata = ingestor.extract_text(IngestableFile(f, "bin.txt"))

        assert "Hello" in text

    def test_empty_file_yields_empty_text(self, tmp_path, isolated_registry):
        """An empty file produces an empty string with valid metadata."""
        with make_file(tmp_path, "empty.txt", b"") as f:
            ingestor = make_ingestor(["txt"])
            text, metadata = ingestor.extract_text(IngestableFile(f, "empty.txt"))

        assert text == ""
        assert type(metadata) is Metadata


class TestExtensionValidation:
    def test_extension_mismatch_raises_ingestion_error(
        self, tmp_path, isolated_registry
    ):
        """An unsupported extension raises IngestionError mentioning the
        offending extension, before any file reading happens."""
        with make_file(tmp_path, "doc.csv", "a,b", binary=False) as f:
            ingestor = make_ingestor(["md"])
            with pytest.raises(IngestionError, match="does not match any type") as exc:
                ingestor.extract_text(IngestableFile(f, "doc.csv"))

        assert "csv" in str(exc.value)

    def test_extension_check_uses_instance_formats(self, tmp_path, isolated_registry):
        """The accepted-formats check reflects this instance's formats: a txt
        file is rejected when only md was configured."""
        with make_file(tmp_path, "doc.txt", "x", binary=False) as f:
            ingestor = make_ingestor(["md"])
            with pytest.raises(IngestionError):
                ingestor.extract_text(IngestableFile(f, "doc.txt"))

    def test_ingestor_with_no_formats_rejects_everything(
        self, tmp_path, isolated_registry
    ):
        """A TextIngestor constructed without accepted formats accepts nothing
        but raises a clean IngestionError, not a raw error."""
        with make_file(tmp_path, "doc.txt", "x", binary=False) as f:
            ingestor = TextIngestor()
            with pytest.raises(IngestionError, match="does not match any type"):
                ingestor.extract_text(IngestableFile(f, "doc.txt"))


class TestMetadata:
    def test_metadata_fields_are_correct(self, tmp_path, isolated_registry):
        """Metadata carries the original file name, extension, ingestor type,
        and a float UTC timestamp."""
        with make_file(tmp_path, "doc.txt", b"data") as f:
            ingestor = make_ingestor(["txt"])
            text, metadata = ingestor.extract_text(IngestableFile(f, "doc.txt"))

        assert metadata.file_name == "doc.txt"
        assert metadata.extension == "txt"
        assert metadata.type == "text"
        assert isinstance(metadata.created_at_ts, float)


class TestErrorHandling:
    def test_read_failure_wraps_as_ingestion_error(self, isolated_registry):
        """A stream read failure surfaces as a chained IngestionError, not the
        raw OSError."""
        ingestor = make_ingestor(["txt"])
        ingestable = IngestableFile(FailingBytesIO(b"x"), "doc.txt")

        with pytest.raises(IngestionError, match="Failed to extract text") as exc:
            ingestor.extract_text(ingestable)

        assert isinstance(exc.value.__cause__, OSError)

    def test_seek_failure_wraps_as_ingestion_error(self, isolated_registry):
        """A stream seek failure surfaces as a chained IngestionError."""
        ingestor = make_ingestor(["txt"])
        ingestable = IngestableFile(FailingSeekIO(b"x"), "doc.txt")

        with pytest.raises(IngestionError, match="Failed to extract text") as exc:
            ingestor.extract_text(ingestable)

        assert isinstance(exc.value.__cause__, OSError)

    def test_metadata_error_passes_through_unwrapped(self, tmp_path, isolated_registry):
        """An IngestionError from extract_metadata propagates unchanged, not
        re-wrapped as a generic extraction failure."""
        with make_file(tmp_path, "doc.txt", b"x") as f:
            ingestor = make_ingestor(["txt"])
            ingestor.extract_metadata = lambda file: (_ for _ in ()).throw(
                IngestionError("metadata boom")
            )

            with pytest.raises(IngestionError, match="metadata boom") as exc:
                ingestor.extract_text(IngestableFile(f, "doc.txt"))

        assert "Failed to extract text" not in str(exc.value)

    def test_constructor_wraps_unexpected_errors(self, isolated_registry):
        """An unexpected failure inside super().__init__ (here: an unhashable
        format) is wrapped in a chained IngestionError."""
        with pytest.raises(
            IngestionError, match="Failed to update ingestor map"
        ) as exc:
            TextIngestor(accepted_formats=[["unhashable"]])

        assert isinstance(exc.value.__cause__, TypeError)

    def test_constructor_with_no_formats_does_not_crash(self, isolated_registry):
        """TextIngestor() with the default accepted_formats=None initializes
        cleanly with an empty formats list."""
        ingestor = TextIngestor()

        assert ingestor.accepted_formats == []


class TestRegistration:
    def test_registration_populates_registry(self, isolated_registry):
        """Constructing an ingestor registers its formats in the global map,
        pointing at the instance."""
        ingestor = make_ingestor(["custom-ext"])

        assert "custom-ext" in BaseIngestor.all_formats
        assert BaseIngestor.ingestor_map["custom-ext"] is ingestor

    def test_same_class_re_registration_is_noop(self, isolated_registry):
        """Instantiating the same class with an already-registered format does
        not raise; the registry points at the latest instance."""
        first = make_ingestor(["txt"])
        second = make_ingestor(["txt"])

        assert BaseIngestor.ingestor_map["txt"] is second
        assert BaseIngestor.ingestor_map["txt"] is not first

    def test_cross_class_conflict_raises_ingestion_error(self, isolated_registry):
        """A different class claiming an owned format raises IngestionError
        naming the format and the owning ingestor."""
        FakeAudioIngestor(accepted_formats=["conflict-ext"])

        with pytest.raises(IngestionError, match="already registered by") as exc:
            make_ingestor(["conflict-ext"])

        assert "conflict-ext" in str(exc.value)
