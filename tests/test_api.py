"""Integration tests for FastAPI routes in backend.app.

All external dependencies (FILE_STORE, VECTOR_STORE, APP_STATE, process_ingest_file)
are mocked at their import site in backend.app. Tests use FastAPI's TestClient
for synchronous request/response testing. No real databases, file systems, or
Celery workers are involved.
"""

import io
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.app_state import AppStateError
from backend.filestore.base_filestore import FileStoreError, IngestableFile
from backend.filestore.local_filestore import DataNotFoundError
from backend.ingestors.base_ingestor import BaseIngestor
from backend.vector_store.local_vec_store import ChromaDBError

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_file(name="test.txt", content=b"hello"):
    """Return a files dict for multipart uploads."""
    return {"file": (name, io.BytesIO(content), "application/octet-stream")}


def _fake_multi_files(files):
    """Return a files list for multipart batch uploads.

    `files` is an iterable of (name, content) pairs.
    """
    return [
        ("files", (name, io.BytesIO(content), "application/octet-stream"))
        for name, content in files
    ]


def _fake_ingestable(name="test.txt", content=b"hello"):
    """Return an IngestableFile with a readable BytesIO."""
    return IngestableFile(io.BytesIO(content), name=name)


def _fake_status_row(
    file_id=1,
    file_name="test.txt",
    file_type="txt",
    status="Ingestion Successful",
    error_message=None,
):
    """Return a dict matching AppState.get_status_all() format."""
    return {
        "file_id": file_id,
        "app_state_id": 1,
        "file_name": file_name,
        "file_type": file_type,
        "add_timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "last_update_timestamp": datetime(2025, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
        "status": status,
        "error_message": error_message,
    }


@pytest.fixture(autouse=True)
def _patch_globals(monkeypatch):
    """Reset BaseIngestor registries and patch module-level singletons
    referenced by backend.app so every test is fully isolated."""
    saved_formats = set(BaseIngestor.all_formats)
    saved_map = dict(BaseIngestor.ingestor_map)
    yield
    BaseIngestor.all_formats.clear()
    BaseIngestor.all_formats.update(saved_formats)
    BaseIngestor.ingestor_map.clear()
    BaseIngestor.ingestor_map.update(saved_map)


# ===================================================================
# GET /  (statistics)
# ===================================================================


class TestStatistics:
    @patch("backend.app.APP_STATE")
    def test_returns_status_list(self, mock_state):
        """GET / returns the list from APP_STATE.get_status_all()."""
        mock_state.get_status_all.return_value = [_fake_status_row()]

        resp = client.get("/")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["file_name"] == "test.txt"

    @patch("backend.app.APP_STATE")
    def test_db_error_returns_500(self, mock_state):
        """GET / returns 500 when APP_STATE raises AppStateError."""
        mock_state.get_status_all.side_effect = AppStateError("DB down")

        resp = client.get("/")

        assert resp.status_code == 500
        assert "DB down" in resp.json()["detail"]


# ===================================================================
# POST /upload/
# ===================================================================


class TestUpload:
    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_successful_upload(self, mock_fs, mock_state, mock_task, monkeypatch):
        """POST /upload/ with a valid extension stores file, inserts state,
        dispatches celery task, and returns 200 with ids."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.return_value = 42
        mock_state.insert_file.return_value = 99
        mock_task.delay.return_value = MagicMock(id="task-abc")

        resp = client.post(
            "/upload/",
            files=_fake_file("note.txt", b"data"),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["file_id"] == 42
        assert body["app_state_id"] == 99
        assert body["task_id"] == "task-abc"
        mock_fs.store.assert_called_once()
        mock_state.insert_file.assert_called_once()
        mock_task.delay.assert_called_once_with(42, 99)

    def test_unsupported_extension_returns_422(self, monkeypatch):
        """POST /upload/ with an unsupported extension returns 422."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})

        resp = client.post(
            "/upload/",
            files=_fake_file("image.exe", b"MZ"),
        )

        assert resp.status_code == 422
        body = resp.json()["detail"]
        assert "Unsupported file extension" in body["message"]
        assert "exe" in body["message"]
        assert body["acceptable_formats"] == ["txt"]

    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_store_failure_returns_500(
        self, mock_fs, mock_state, mock_task, monkeypatch
    ):
        """POST /upload/ returns 500 when FILE_STORE.store() raises."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.side_effect = FileStoreError("disk full")

        resp = client.post(
            "/upload/",
            files=_fake_file("note.txt", b"data"),
        )

        assert resp.status_code == 500
        assert "disk full" in resp.json()["detail"]
        mock_state.insert_file.assert_not_called()
        mock_task.delay.assert_not_called()

    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_insert_failure_returns_500(
        self, mock_fs, mock_state, mock_task, monkeypatch
    ):
        """POST /upload/ returns 500 when APP_STATE.insert_file() raises."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.return_value = 42
        mock_state.insert_file.side_effect = AppStateError("db write failed")

        resp = client.post(
            "/upload/",
            files=_fake_file("note.txt", b"data"),
        )

        assert resp.status_code == 500
        assert "db write failed" in resp.json()["detail"]
        mock_task.delay.assert_not_called()

    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_celery_dispatch_failure_returns_500(
        self, mock_fs, mock_state, mock_task, monkeypatch
    ):
        """POST /upload/ returns 500 when Celery .delay() raises."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.return_value = 42
        mock_state.insert_file.return_value = 99
        mock_task.delay.side_effect = ConnectionError("redis down")

        resp = client.post(
            "/upload/",
            files=_fake_file("note.txt", b"data"),
        )

        assert resp.status_code == 500
        assert "redis down" in resp.json()["detail"]
        mock_state.update_file.assert_called_once_with(
            99,
            status="Ingestion Failed",
            error_msg="Failed to queue ingestion: redis down",
        )

    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_dispatch_failure_still_returns_500_if_status_update_fails(
        self, mock_fs, mock_state, mock_task, monkeypatch
    ):
        """If recording the failure also fails, the original 500 is still returned."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.return_value = 42
        mock_state.insert_file.return_value = 99
        mock_task.delay.side_effect = ConnectionError("redis down")
        mock_state.update_file.side_effect = AppStateError("db locked")

        resp = client.post(
            "/upload/",
            files=_fake_file("note.txt", b"data"),
        )

        assert resp.status_code == 500
        assert "redis down" in resp.json()["detail"]


# ===================================================================
# POST /upload/batch/  (multi-file upload)
# ===================================================================


class TestBatchUpload:
    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_successful_batch(self, mock_fs, mock_state, mock_task, monkeypatch):
        """POST /upload/batch/ with two valid files stores, tracks, and
        dispatches both, returning one result per file."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.side_effect = [42, 43]
        mock_state.insert_file.side_effect = [99, 100]
        mock_task.delay.side_effect = [MagicMock(id="task-a"), MagicMock(id="task-b")]

        resp = client.post(
            "/upload/batch/",
            files=_fake_multi_files([("a.txt", b"aaa"), ("b.txt", b"bbb")]),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["queued_count"] == 2
        assert body["rejected_count"] == 0
        assert body["failed_count"] == 0
        files = body["files"]
        assert [f["file_name"] for f in files] == ["a.txt", "b.txt"]
        assert [f["status"] for f in files] == ["queued", "queued"]
        assert [f["file_id"] for f in files] == [42, 43]
        assert [f["app_state_id"] for f in files] == [99, 100]
        assert [f["task_id"] for f in files] == ["task-a", "task-b"]
        assert all(f["error"] is None for f in files)
        assert mock_fs.store.call_count == 2
        assert mock_state.insert_file.call_count == 2
        assert mock_task.delay.call_args_list == [call(42, 99), call(43, 100)]

    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_mixed_batch_queues_supported_and_rejects_unsupported(
        self, mock_fs, mock_state, mock_task, monkeypatch
    ):
        """An unsupported file is rejected per-file without failing the batch."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.return_value = 42
        mock_state.insert_file.return_value = 99
        mock_task.delay.return_value = MagicMock(id="task-a")

        resp = client.post(
            "/upload/batch/",
            files=_fake_multi_files([("good.txt", b"data"), ("bad.exe", b"MZ")]),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["queued_count"] == 1
        assert body["rejected_count"] == 1
        assert body["failed_count"] == 0
        good, bad = body["files"]
        assert good["status"] == "queued"
        assert good["file_id"] == 42
        assert bad["status"] == "rejected"
        assert bad["file_id"] is None
        assert bad["app_state_id"] is None
        assert "Unsupported file extension: exe" in bad["error"]
        mock_fs.store.assert_called_once()
        mock_state.insert_file.assert_called_once()
        mock_task.delay.assert_called_once_with(42, 99)

    def test_all_unsupported_returns_422(self, monkeypatch):
        """A batch where every file has an unsupported extension returns 422."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})

        resp = client.post(
            "/upload/batch/",
            files=_fake_multi_files([("a.exe", b"MZ"), ("b.exe", b"MZ")]),
        )

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "No files with supported extensions" in detail["message"]
        assert detail["acceptable_formats"] == ["txt"]

    def test_missing_files_field_returns_422(self):
        """A request without the `files` field is rejected by validation."""
        resp = client.post(
            "/upload/batch/",
            files=[("file", ("a.txt", io.BytesIO(b"data"), "text/plain"))],
        )

        assert resp.status_code == 422
        assert any(
            err["type"] == "missing" and err["loc"] == ["body", "files"]
            for err in resp.json()["detail"]
        )

    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_store_failure_fails_only_that_file(
        self, mock_fs, mock_state, mock_task, monkeypatch
    ):
        """A store failure marks only that file as failed; others proceed."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.side_effect = [FileStoreError("disk full"), 43]
        mock_state.insert_file.return_value = 100
        mock_task.delay.return_value = MagicMock(id="task-b")

        resp = client.post(
            "/upload/batch/",
            files=_fake_multi_files([("a.txt", b"aaa"), ("b.txt", b"bbb")]),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["queued_count"] == 1
        assert body["failed_count"] == 1
        failed, queued = body["files"]
        assert failed["file_name"] == "a.txt"
        assert failed["status"] == "failed"
        assert "disk full" in failed["error"]
        assert queued["file_name"] == "b.txt"
        assert queued["status"] == "queued"
        assert queued["file_id"] == 43
        mock_state.insert_file.assert_called_once()
        mock_task.delay.assert_called_once_with(43, 100)

    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_dispatch_failure_marks_file_failed(
        self, mock_fs, mock_state, mock_task, monkeypatch
    ):
        """A Celery dispatch failure marks that file failed and records it."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.side_effect = [42, 43]
        mock_state.insert_file.side_effect = [99, 100]
        mock_task.delay.side_effect = [
            ConnectionError("redis down"),
            MagicMock(id="task-b"),
        ]

        resp = client.post(
            "/upload/batch/",
            files=_fake_multi_files([("a.txt", b"aaa"), ("b.txt", b"bbb")]),
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["queued_count"] == 1
        assert body["failed_count"] == 1
        first, second = body["files"]
        assert first["status"] == "failed"
        assert "redis down" in first["error"]
        assert second["status"] == "queued"
        mock_state.update_file.assert_called_once_with(
            99,
            status="Ingestion Failed",
            error_msg="Failed to queue ingestion: redis down",
        )

    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_all_server_failures_return_500_with_per_file_details(
        self, mock_fs, mock_state, mock_task, monkeypatch
    ):
        """When every file hits a server-side error, return 500 plus details."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.side_effect = [
            FileStoreError("disk full"),
            FileStoreError("disk full"),
        ]

        resp = client.post(
            "/upload/batch/",
            files=_fake_multi_files([("a.txt", b"aaa"), ("b.txt", b"bbb")]),
        )

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail["message"] == "All uploads failed"
        assert [f["status"] for f in detail["files"]] == ["failed", "failed"]
        assert all("disk full" in f["error"] for f in detail["files"])
        mock_state.insert_file.assert_not_called()
        mock_task.delay.assert_not_called()

    @patch("backend.app.process_ingest_file")
    @patch("backend.app.APP_STATE")
    @patch("backend.app.FILE_STORE")
    def test_rejected_and_failed_without_queued_returns_500(
        self, mock_fs, mock_state, mock_task, monkeypatch
    ):
        """A batch that queued nothing and had server failures returns 500."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})
        mock_fs.store.side_effect = FileStoreError("disk full")

        resp = client.post(
            "/upload/batch/",
            files=_fake_multi_files([("a.txt", b"aaa"), ("bad.exe", b"MZ")]),
        )

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert detail["message"] == "All uploads failed"
        statuses = [f["status"] for f in detail["files"]]
        assert "rejected" in statuses
        assert "failed" in statuses


# ===================================================================
# POST /search/
# ===================================================================


class TestSearch:
    @patch("backend.app.VECTOR_STORE")
    def test_search_returns_results(self, mock_vs):
        """POST /search/ returns shaped results from the vector store."""
        mock_vs.get.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["some text"]],
            "metadatas": [
                [{"file_name": "a.txt", "extension": "txt", "created_at_ts": 1000.0}]
            ],
            "distances": [[0.5]],
        }

        resp = client.post("/search/", json={"q": "test query", "k": 5})

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "test query"
        assert body["total_results"] == 1
        assert body["results"][0]["chunk_text"] == "some text"
        mock_vs.get.assert_called_once()

    @patch("backend.app.VECTOR_STORE")
    def test_search_empty_results(self, mock_vs):
        """POST /search/ with no matches returns empty results list."""
        mock_vs.get.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        resp = client.post("/search/", json={"q": "no match"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_results"] == 0
        assert body["results"] == []

    @patch("backend.app.VECTOR_STORE")
    def test_vector_store_error_returns_500(self, mock_vs):
        """POST /search/ returns 500 when VECTOR_STORE.get() raises."""
        mock_vs.get.side_effect = ChromaDBError("connection refused")

        resp = client.post("/search/", json={"q": "test"})

        assert resp.status_code == 500
        assert "connection refused" in resp.json()["detail"]

    def test_unknown_extension_returns_422(self, monkeypatch):
        """POST /search/ with unsupported extension returns 422."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt"})

        resp = client.post(
            "/search/",
            json={"q": "test", "extension": ["txt", "xyz"]},
        )

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "xyz" in detail["unknown_extensions"]

    def test_date_from_after_date_to_returns_422(self, monkeypatch):
        """POST /search/ where date_from > date_to returns 422."""
        monkeypatch.setattr(BaseIngestor, "all_formats", set())

        resp = client.post(
            "/search/",
            json={
                "q": "test",
                "date_from": "2025-06-01T00:00:00",
                "date_to": "2025-01-01T00:00:00",
            },
        )

        assert resp.status_code == 422
        assert "date_from must be before date_to" in resp.json()["detail"]["message"]

    @patch("backend.app.VECTOR_STORE")
    def test_search_passes_filters_to_vector_store(self, mock_vs, monkeypatch):
        """POST /search/ builds where clause and passes it to VECTOR_STORE.get."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt", "pdf"})
        mock_vs.get.return_value = None

        resp = client.post(
            "/search/",
            json={
                "q": "query",
                "k": 3,
                "extension": ["txt"],
                "date_from": "2025-01-01T00:00:00",
                "date_to": "2025-12-31T00:00:00",
            },
        )

        assert resp.status_code == 200
        call_kwargs = mock_vs.get.call_args
        assert call_kwargs[1]["k"] == 3 or call_kwargs[0][1] == 3
        constraints = (
            call_kwargs[1].get("constraints") or call_kwargs[0][2]
            if len(call_kwargs[0]) > 2
            else call_kwargs[1].get("constraints")
        )
        assert constraints is not None

    def test_empty_query_rejected(self):
        """POST /search/ with empty query is rejected by Pydantic validation."""
        resp = client.post("/search/", json={"q": ""})

        assert resp.status_code == 422


# ===================================================================
# GET /files/  (list files)
# ===================================================================


class TestListFiles:
    @patch("backend.app.FILE_STORE")
    @patch("backend.app.APP_STATE")
    def test_returns_files_with_size(self, mock_state, mock_fs):
        """GET /files/ enriches each row with file_size from FILE_STORE."""
        row = _fake_status_row(file_id=1)
        mock_state.get_status_all.return_value = [row]
        mock_fs.get_metadata.return_value = {"size": 1024}

        resp = client.get("/files/")

        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["file_size"] == 1024
        assert data[0]["duration_seconds"] == 10.0

    @patch("backend.app.FILE_STORE")
    @patch("backend.app.APP_STATE")
    def test_missing_file_sets_size_none(self, mock_state, mock_fs):
        """GET /files/ gracefully sets file_size=None when metadata lookup fails."""
        row = _fake_status_row(file_id=1)
        mock_state.get_status_all.return_value = [row]
        mock_fs.get_metadata.side_effect = DataNotFoundError("no such file")

        resp = client.get("/files/")

        assert resp.status_code == 200
        assert resp.json()[0]["file_size"] is None

    @patch("backend.app.APP_STATE")
    def test_db_error_returns_500(self, mock_state):
        """GET /files/ returns 500 when APP_STATE raises."""
        mock_state.get_status_all.side_effect = AppStateError("DB connection lost")

        resp = client.get("/files/")

        assert resp.status_code == 500
        assert "DB connection lost" in resp.json()["detail"]

    @patch("backend.app.FILE_STORE")
    @patch("backend.app.APP_STATE")
    def test_filestore_error_sets_size_none(self, mock_state, mock_fs):
        """GET /files/ gracefully handles FileStoreError (not just DataNotFoundError)."""
        row = _fake_status_row(file_id=1)
        mock_state.get_status_all.return_value = [row]
        mock_fs.get_metadata.side_effect = FileStoreError("IO error")

        resp = client.get("/files/")

        assert resp.status_code == 200
        assert resp.json()[0]["file_size"] is None

    @patch("backend.app.FILE_STORE")
    @patch("backend.app.APP_STATE")
    def test_null_timestamps_yield_none_duration(self, mock_state, mock_fs):
        """GET /files/ sets duration_seconds=None when timestamps are missing."""
        row = _fake_status_row(file_id=1)
        row["add_timestamp"] = None
        row["last_update_timestamp"] = None
        mock_state.get_status_all.return_value = [row]
        mock_fs.get_metadata.return_value = {"size": 100}

        resp = client.get("/files/")

        assert resp.status_code == 200
        assert resp.json()[0]["duration_seconds"] is None


# ===================================================================
# GET /file/{file_id}  (download file contents)
# ===================================================================


class TestGetFileContents:
    @patch("backend.app.FILE_STORE")
    def test_returns_file_content(self, mock_fs):
        """GET /file/{id} returns the file bytes with correct media type."""
        content = b"file content here"
        ingestable = IngestableFile(io.BytesIO(content), name="report.pdf")
        mock_fs.get.return_value = ingestable

        resp = client.get("/file/1")

        assert resp.status_code == 200
        assert resp.content == content
        assert resp.headers["content-type"] == "application/pdf"
        assert "report.pdf" in resp.headers["content-disposition"]

    @patch("backend.app.FILE_STORE")
    def test_unknown_extension_uses_octet_stream(self, mock_fs):
        """GET /file/{id} uses application/octet-stream for unknown extensions."""
        ingestable = IngestableFile(io.BytesIO(b"data"), name="file.xyz")
        mock_fs.get.return_value = ingestable

        resp = client.get("/file/1")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"

    @patch("backend.app.FILE_STORE")
    def test_not_found_returns_404(self, mock_fs):
        """GET /file/{id} returns 404 when file doesn't exist."""
        mock_fs.get.side_effect = DataNotFoundError("No file entry for ID: 999")

        resp = client.get("/file/999")

        assert resp.status_code == 404
        assert "999" in resp.json()["detail"]

    @patch("backend.app.FILE_STORE")
    def test_disk_io_error_returns_500(self, mock_fs):
        """GET /file/{id} returns 500 on general FileStoreError."""
        mock_fs.get.side_effect = FileStoreError("disk read failure")

        resp = client.get("/file/1")

        assert resp.status_code == 500
        assert "disk read failure" in resp.json()["detail"]


# ===================================================================
# GET /files/{file_id}  (single file status)
# ===================================================================


class TestFileStatus:
    @patch("backend.app.APP_STATE")
    def test_returns_file_status(self, mock_state):
        """GET /files/{id} returns the status row for a single file."""
        mock_state.get_status_by_id.return_value = _fake_status_row(file_id=5)

        resp = client.get("/files/5")

        assert resp.status_code == 200
        assert resp.json()["file_id"] == 5

    @patch("backend.app.APP_STATE")
    def test_not_found_returns_404(self, mock_state):
        """GET /files/{id} returns 404 when file_id doesn't exist."""
        mock_state.get_status_by_id.side_effect = FileNotFoundError(
            "No file entry found in database for ID: 999"
        )

        resp = client.get("/files/999")

        assert resp.status_code == 404
        assert "999" in resp.json()["detail"]


# ===================================================================
# GET /formats
# ===================================================================


class TestFormats:
    def test_returns_all_registered_formats(self, monkeypatch):
        """GET /formats returns the list of all registered formats."""
        monkeypatch.setattr(BaseIngestor, "all_formats", {"txt", "pdf", "md"})

        resp = client.get("/formats")

        assert resp.status_code == 200
        assert sorted(resp.json()) == ["md", "pdf", "txt"]

    def test_empty_when_no_formats_registered(self, monkeypatch):
        """GET /formats returns empty list when no formats are registered."""
        monkeypatch.setattr(BaseIngestor, "all_formats", set())

        resp = client.get("/formats")

        assert resp.status_code == 200
        assert resp.json() == []
