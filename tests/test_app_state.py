"""Integration tests for AppState (backend.app_state).

Uses a real SQLite database backed by tmp_path to verify insert, update,
get_status_all, get_status_by_id operations plus error handling and
the get_status_by_id FileNotFoundError propagation fix.
"""

import pytest

from backend.app_state import AppState, AppStateError


@pytest.fixture
def state(tmp_path):
    """An AppState backed by a fresh tmp_path database."""
    return AppState(db_path=str(tmp_path / "app_state"))


# ===================================================================
# Insert
# ===================================================================


class TestInsert:
    def test_insert_returns_integer_id(self, state):
        """insert_file returns an integer id for the new row."""
        row_id = state.insert_file(
            file_name="doc.txt", file_type="txt", file_status="Storage Complete"
        )
        assert isinstance(row_id, int)

    def test_insert_persists_all_fields(self, state):
        """insert_file persists name, type, status, file_id in the DB."""
        row_id = state.insert_file(
            file_name="report.pdf",
            file_type="pdf",
            file_status="Storage Complete",
            file_id=42,
        )

        rows = state.get_status_all()
        assert len(rows) == 1
        row = rows[0]
        assert row["file_name"] == "report.pdf"
        assert row["file_type"] == "pdf"
        assert row["status"] == "Storage Complete"
        assert row["file_id"] == 42
        assert row["app_state_id"] == row_id

    def test_insert_multiple_rows(self, state):
        """Multiple inserts create distinct rows."""
        id1 = state.insert_file("a.txt", "txt", "pending")
        id2 = state.insert_file("b.txt", "txt", "pending")

        assert id1 != id2
        rows = state.get_status_all()
        assert len(rows) == 2

    def test_insert_missing_fields_raises_appstate_error(self, state):
        """insert_file with empty/missing fields raises AppStateError."""
        with pytest.raises(AppStateError, match="Failed to insert file"):
            state.insert_file(file_name="", file_type="txt", file_status="pending")

    def test_insert_missing_status_raises_appstate_error(self, state):
        """insert_file with empty status raises AppStateError."""
        with pytest.raises(AppStateError, match="Failed to insert file"):
            state.insert_file(file_name="doc.txt", file_type="txt", file_status="")


# ===================================================================
# Update
# ===================================================================


class TestUpdate:
    def test_update_changes_status(self, state):
        """update_file changes the status field."""
        row_id = state.insert_file("doc.txt", "txt", "pending")
        state.update_file(row_id, status="Ingestion Complete")

        rows = state.get_status_all()
        assert rows[0]["status"] == "Ingestion Complete"

    def test_update_changes_error_message(self, state):
        """update_file changes the error_message field."""
        row_id = state.insert_file("doc.txt", "txt", "pending")
        state.update_file(row_id, error_msg="something went wrong")

        rows = state.get_status_all()
        assert rows[0]["error_message"] == "something went wrong"

    def test_update_changes_both_fields(self, state):
        """update_file can change both status and error_msg at once."""
        row_id = state.insert_file("doc.txt", "txt", "pending")
        state.update_file(row_id, status="Failed", error_msg="OCR error")

        rows = state.get_status_all()
        assert rows[0]["status"] == "Failed"
        assert rows[0]["error_message"] == "OCR error"

    def test_update_nonexistent_id_raises_appstate_error(self, state):
        """update_file with a missing id raises AppStateError (wrapping
        FileNotFoundError internally)."""
        with pytest.raises(AppStateError, match="Failed to update file"):
            state.update_file(999, status="whatever")

    def test_update_preserves_other_fields(self, state):
        """update_file only changes the specified fields, leaving others intact."""
        row_id = state.insert_file("doc.txt", "txt", "pending", file_id=7)
        state.update_file(row_id, status="done")

        rows = state.get_status_all()
        assert rows[0]["file_name"] == "doc.txt"
        assert rows[0]["file_type"] == "txt"
        assert rows[0]["file_id"] == 7
        assert rows[0]["status"] == "done"


# ===================================================================
# get_status_all
# ===================================================================


class TestGetStatusAll:
    def test_empty_database_returns_empty_list(self, state):
        """get_status_all returns [] when no rows exist."""
        assert state.get_status_all() == []

    def test_returns_all_rows(self, state):
        """get_status_all returns every inserted row."""
        state.insert_file("a.txt", "txt", "pending")
        state.insert_file("b.pdf", "pdf", "done")

        rows = state.get_status_all()
        assert len(rows) == 2
        names = {r["file_name"] for r in rows}
        assert names == {"a.txt", "b.pdf"}

    def test_row_dict_has_all_keys(self, state):
        """Each row dict from get_status_all contains all expected keys."""
        state.insert_file("doc.txt", "txt", "pending")
        row = state.get_status_all()[0]

        expected_keys = {
            "file_id",
            "app_state_id",
            "file_name",
            "file_type",
            "add_timestamp",
            "last_update_timestamp",
            "status",
            "error_message",
        }
        assert set(row.keys()) == expected_keys

    def test_initial_status_is_pending(self, state):
        """A newly inserted row has status 'pending' if not specified."""
        state.insert_file("doc.txt", "txt", "pending")
        rows = state.get_status_all()
        assert rows[0]["status"] == "pending"

    def test_timestamps_are_set(self, state):
        """add_timestamp and last_update_timestamp are set on insert."""
        state.insert_file("doc.txt", "txt", "done")
        row = state.get_status_all()[0]

        assert row["add_timestamp"] is not None
        assert row["last_update_timestamp"] is not None


# ===================================================================
# get_status_by_id
# ===================================================================


class TestGetStatusById:
    def test_returns_correct_row(self, state):
        """get_status_by_id returns the row matching the given id."""
        row_id = state.insert_file("doc.txt", "txt", "done")
        result = state.get_status_by_id(row_id)

        assert result["file_name"] == "doc.txt"
        assert result["status"] == "done"
        assert result["app_state_id"] == row_id

    def test_missing_id_raises_file_not_found(self, state):
        """get_status_by_id raises FileNotFoundError (not AppStateError)
        for a non-existent id, verifying the fix in app_state.py."""
        with pytest.raises(FileNotFoundError, match="999"):
            state.get_status_by_id(999)

    def test_file_not_found_not_wrapped_in_appstate_error(self, state):
        """The FileNotFoundError from get_status_by_id is a pure
        FileNotFoundError, not a subclass of AppStateError."""
        with pytest.raises(FileNotFoundError) as exc:
            state.get_status_by_id(999)

        assert not isinstance(exc.value, AppStateError)


# ===================================================================
# Init
# ===================================================================


class TestInit:
    def test_init_creates_directory(self, tmp_path):
        """AppState.__init__ creates the database directory."""
        db_path = tmp_path / "new_dir"
        AppState(db_path=str(db_path))
        assert db_path.exists()

    def test_init_creates_database_file(self, tmp_path):
        """AppState.__init__ creates the app_state.db file."""
        db_path = tmp_path / "db"
        AppState(db_path=str(db_path))
        assert (db_path / "app_state.db").exists()

    def test_init_with_unwritable_path_raises_appstate_error(self, tmp_path):
        """An unusable path raises AppStateError."""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")

        with pytest.raises(AppStateError, match="Failed to initialize"):
            AppState(db_path=str(blocker / "sub"))

    def test_init_is_idempotent(self, tmp_path):
        """Calling AppState() twice on the same path doesn't fail."""
        db_path = tmp_path / "db"
        AppState(db_path=str(db_path))
        state2 = AppState(db_path=str(db_path))
        rows = state2.get_status_all()
        assert rows == []


# ===================================================================
# Full lifecycle
# ===================================================================


class TestLifecycle:
    def test_insert_update_get_lifecycle(self, state):
        """Full lifecycle: insert -> update through statuses -> query final."""
        row_id = state.insert_file("scan.png", "png", "Storage Complete")
        assert state.get_status_by_id(row_id)["status"] == "Storage Complete"

        state.update_file(row_id, status="Ingestion Complete")
        assert state.get_status_by_id(row_id)["status"] == "Ingestion Complete"

        state.update_file(row_id, status="Chunking Complete")
        state.update_file(row_id, status="Embedding Complete")
        state.update_file(row_id, status="Ingestion Successful")

        final = state.get_status_by_id(row_id)
        assert final["status"] == "Ingestion Successful"
        assert final["error_message"] is None

    def test_error_lifecycle(self, state):
        """Lifecycle ending in failure: insert -> update -> failure with message."""
        row_id = state.insert_file("bad.pdf", "pdf", "Storage Complete")
        state.update_file(row_id, status="Ingestion Failed", error_msg="OCR crashed")

        final = state.get_status_by_id(row_id)
        assert final["status"] == "Ingestion Failed"
        assert final["error_message"] == "OCR crashed"


class TestIdSpaces:
    def test_status_lookup_uses_app_state_id_not_file_id(self, state):
        """get_status_by_id resolves the AppState ID; the returned file_id is
        the separate file-store ID, so the two never get conflated."""
        app_state_id = state.insert_file("doc.txt", "txt", "pending", file_id=42)

        row = state.get_status_by_id(app_state_id)

        assert row["app_state_id"] == app_state_id
        assert row["file_id"] == 42
        assert app_state_id != 42

    def test_file_store_id_is_not_a_status_key(self, state):
        """Looking up a file_id that is not an AppState ID finds nothing."""
        state.insert_file("doc.txt", "txt", "pending", file_id=42)

        with pytest.raises(FileNotFoundError):
            state.get_status_by_id(42)
