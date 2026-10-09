"""Integration tests for the Celery ingestion task (backend.tasks.process_ingest_file).

All dependencies (FILE_STORE, CHUNKER, VECTOR_STORE, APP_STATE) are mocked at
their import site in backend.tasks. Tests verify the full pipeline orchestration:
file retrieval → ingestor dispatch → text extraction → chunking → embedding →
status updates, plus all error-handling branches.
"""

import io
from unittest.mock import patch

import pytest

from backend.app_state import AppStateError
from backend.filestore.base_filestore import IngestableFile
from backend.ingestors.base_ingestor import BaseIngestor, IngestionError, Metadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Snapshot and restore the global ingestor registry."""
    saved_formats = set(BaseIngestor.all_formats)
    saved_map = dict(BaseIngestor.ingestor_map)
    yield
    BaseIngestor.all_formats.clear()
    BaseIngestor.all_formats.update(saved_formats)
    BaseIngestor.ingestor_map.clear()
    BaseIngestor.ingestor_map.update(saved_map)


def _fake_file(name="test.txt", content=b"hello world"):
    """Return an IngestableFile with readable BytesIO."""
    return IngestableFile(io.BytesIO(content), name=name, file_id=1)


def _fake_metadata(file_name="test.txt", extension="txt"):
    return Metadata(file_id=1, file_name=file_name, type="text", extension=extension)


class FakeIngestor(BaseIngestor):
    """Stub ingestor that returns configurable text and metadata."""

    def __init__(self, text="extracted text", metadata=None, error=None):
        self._text = text
        self._metadata = metadata
        self._error = error
        super().__init__(type="fake", accepted_formats=["fake"], name="FakeIngestor")

    def extract_text(self, file):
        if self._error:
            raise self._error
        meta = self._metadata or _fake_metadata(file.file_name, file.extension)
        return self._text, meta


# ===================================================================
# Successful pipeline
# ===================================================================


class TestSuccessfulPipeline:
    @patch("backend.tasks.VECTOR_STORE")
    @patch("backend.tasks.CHUNKER")
    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_full_pipeline_updates_status_at_each_stage(
        self, mock_fs, mock_state, mock_chunker, mock_vs
    ):
        """The task calls APP_STATE.update_file at every pipeline stage and
        ends with 'Ingestion Successful'."""
        from backend.tasks import process_ingest_file

        ingestor = FakeIngestor(text="some content here")
        BaseIngestor.ingestor_map["fake"] = ingestor

        mock_fs.get.return_value = _fake_file("doc.fake")
        mock_chunker.split_text.return_value = ["chunk1", "chunk2"]
        mock_vs.add.return_value = None

        process_ingest_file(file_id=1, app_state_id=10)

        # Verify status updates in order
        update_calls = mock_state.update_file.call_args_list
        statuses = [c.args[1] for c in update_calls if len(c.args) > 1]
        assert statuses == [
            "Ingestion Complete",
            "Chunking Complete",
            "Embedding Complete",
            "Ingestion Successful",
        ]

    @patch("backend.tasks.VECTOR_STORE")
    @patch("backend.tasks.CHUNKER")
    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_pipeline_passes_chunks_and_metadata_to_vector_store(
        self, mock_fs, mock_state, mock_chunker, mock_vs
    ):
        """The task passes the chunked text and replicated metadata to VECTOR_STORE.add."""
        from backend.tasks import process_ingest_file

        meta = _fake_metadata("doc.fake", "fake")
        ingestor = FakeIngestor(text="content", metadata=meta)
        BaseIngestor.ingestor_map["fake"] = ingestor

        mock_fs.get.return_value = _fake_file("doc.fake")
        mock_chunker.split_text.return_value = ["c1", "c2", "c3"]

        process_ingest_file(file_id=1, app_state_id=10)

        mock_vs.add.assert_called_once_with(
            ["c1", "c2", "c3"],
            [meta, meta, meta],
        )

    @patch("backend.tasks.VECTOR_STORE")
    @patch("backend.tasks.CHUNKER")
    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_returns_nothing_on_success(
        self, mock_fs, mock_state, mock_chunker, mock_vs
    ):
        """A successful task returns None (no error string)."""
        from backend.tasks import process_ingest_file

        ingestor = FakeIngestor(text="content")
        BaseIngestor.ingestor_map["fake"] = ingestor

        mock_fs.get.return_value = _fake_file("doc.fake")
        mock_chunker.split_text.return_value = ["c1"]

        result = process_ingest_file(file_id=1, app_state_id=10)

        assert result is None


# ===================================================================
# IngestionError handling
# ===================================================================


class TestIngestionErrorHandling:
    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_no_ingestor_for_extension(self, mock_fs, mock_state):
        """When no ingestor exists for the file extension, the task records
        'Ingestion Failed' and returns an error string."""
        from backend.tasks import process_ingest_file

        mock_fs.get.return_value = _fake_file("data.xyz")

        result = process_ingest_file(file_id=1, app_state_id=10)

        assert "No ingestor found" in result
        mock_state.update_file.assert_called_once_with(
            10,
            status="Ingestion Failed",
            error_msg="Ingestion Error: No ingestor found for file type: xyz",
        )

    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_empty_text_after_extraction(self, mock_fs, mock_state):
        """When extraction returns empty/whitespace text, the task records
        'Ingestion Failed' and returns an error string."""
        from backend.tasks import process_ingest_file

        ingestor = FakeIngestor(text="   ")
        BaseIngestor.ingestor_map["txt"] = ingestor

        mock_fs.get.return_value = _fake_file("blank.txt")

        result = process_ingest_file(file_id=1, app_state_id=10)

        assert "No text could be extracted" in result
        mock_state.update_file.assert_called_once_with(
            10,
            status="Ingestion Failed",
            error_msg=mock_state.update_file.call_args[1]["error_msg"],
        )

    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_extraction_failure_records_error(self, mock_fs, mock_state):
        """When ingestor.extract_text() raises IngestionError, the task
        records 'Ingestion Failed' with the error message."""
        from backend.tasks import process_ingest_file

        ingestor = FakeIngestor(error=IngestionError("OCR failed"))
        BaseIngestor.ingestor_map["txt"] = ingestor

        mock_fs.get.return_value = _fake_file("scan.txt")

        result = process_ingest_file(file_id=1, app_state_id=10)

        assert "OCR failed" in result
        mock_state.update_file.assert_called_once_with(
            10, status="Ingestion Failed", error_msg="Ingestion Error: OCR failed"
        )


# ===================================================================
# AppStateError handling
# ===================================================================


class TestAppStateErrorHandling:
    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_appstate_error_during_update_records_failure(self, mock_fs, mock_state):
        """When APP_STATE.update_file raises AppStateError during the pipeline,
        the task catches it and records 'Ingestion Failed'."""
        from backend.tasks import process_ingest_file

        ingestor = FakeIngestor(text="content")
        BaseIngestor.ingestor_map["txt"] = ingestor

        mock_fs.get.return_value = _fake_file("doc.txt")
        # First call (Ingestion Complete) raises AppStateError
        mock_state.update_file.side_effect = AppStateError("DB write failed")

        result = process_ingest_file(file_id=1, app_state_id=10)

        assert "DB write failed" in result

    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_appstate_error_handler_also_fails_returns_appstate_error(
        self, mock_fs, mock_state
    ):
        """When both the pipeline and the error handler's update_file fail,
        the task returns the AppStateError from the handler."""
        from backend.tasks import process_ingest_file

        ingestor = FakeIngestor(text="content")
        BaseIngestor.ingestor_map["txt"] = ingestor

        mock_fs.get.return_value = _fake_file("doc.txt")
        # Both calls fail — first pipeline, then error handler
        mock_state.update_file.side_effect = [
            AppStateError("first fail"),  # Ingestion Complete
            AppStateError("second fail"),  # error handler's update_file
        ]

        result = process_ingest_file(file_id=1, app_state_id=10)

        assert "AppState Error" in result


# ===================================================================
# Unexpected error handling
# ===================================================================


class TestUnexpectedErrorHandling:
    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_unexpected_exception_records_error(self, mock_fs, mock_state):
        """A non-IngestionError, non-AppStateError exception is caught by the
        generic handler and recorded as 'Ingestion Failed'."""
        from backend.tasks import process_ingest_file

        mock_fs.get.return_value = _fake_file("doc.txt")
        mock_fs.get.side_effect = RuntimeError("unexpected boom")

        result = process_ingest_file(file_id=1, app_state_id=10)

        assert "Unexpected error" in result
        assert "unexpected boom" in result
        mock_state.update_file.assert_called_once_with(
            10,
            status="Ingestion Failed",
            error_msg="Unexpected error during ingestion: unexpected boom",
        )

    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_unexpected_error_handler_also_fails(self, mock_fs, mock_state):
        """When the generic error handler's update_file also fails, the task
        returns the AppStateError from the handler."""
        from backend.tasks import process_ingest_file

        mock_fs.get.side_effect = [
            RuntimeError("boom"),  # pipeline error
            RuntimeError("should not matter"),
        ]
        mock_state.update_file.side_effect = AppStateError("handler fail")

        result = process_ingest_file(file_id=1, app_state_id=10)

        assert "AppState Error" in result
        assert "handler fail" in result


# ===================================================================
# IngestionError from error handler also fails
# ===================================================================


class TestDoubleFailure:
    @patch("backend.tasks.APP_STATE")
    @patch("backend.tasks.FILE_STORE")
    def test_ingestion_error_handler_update_also_fails(self, mock_fs, mock_state):
        """When the IngestionError handler's update_file also fails, the task
        returns the AppStateError from the handler."""
        from backend.tasks import process_ingest_file

        ingestor = FakeIngestor(error=IngestionError("extract fail"))
        BaseIngestor.ingestor_map["txt"] = ingestor

        mock_fs.get.return_value = _fake_file("doc.txt")
        mock_state.update_file.side_effect = AppStateError("handler db fail")

        result = process_ingest_file(file_id=1, app_state_id=10)

        assert "AppState Error" in result
        assert "handler db fail" in result
