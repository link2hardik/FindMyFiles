"""Unit tests for backend.search module.

Covers build_where filter construction, normalize_distance scoring,
SearchRequest validation, and shape_search_response output formatting.
No external dependencies -- all tests use pure data.
"""

from datetime import datetime, timezone

import pytest

from backend.search import (
    SearchRequest,
    build_where,
    normalize_distance,
    shape_search_response,
)


class TestBuildWhere:
    def test_no_filters_returns_none(self):
        assert build_where() is None

    def test_extension_filter_only(self):
        result = build_where(extension=["txt", "pdf"])
        assert result == {"extension": {"$in": ["txt", "pdf"]}}

    def test_date_from_only(self):
        dt = datetime(2025, 3, 15, tzinfo=timezone.utc)
        result = build_where(date_from=dt)
        assert result == {"created_at_ts": {"$gte": dt.timestamp()}}

    def test_date_to_only(self):
        dt = datetime(2025, 12, 31, tzinfo=timezone.utc)
        result = build_where(date_to=dt)
        assert result == {"created_at_ts": {"$lte": dt.timestamp()}}

    def test_extension_and_date_from(self):
        dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = build_where(extension=["md"], date_from=dt)
        assert result == {
            "$and": [
                {"extension": {"$in": ["md"]}},
                {"created_at_ts": {"$gte": dt.timestamp()}},
            ]
        }

    def test_all_three_filters(self):
        dt_from = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dt_to = datetime(2025, 6, 30, tzinfo=timezone.utc)
        result = build_where(extension=["txt"], date_from=dt_from, date_to=dt_to)
        assert result == {
            "$and": [
                {"extension": {"$in": ["txt"]}},
                {"created_at_ts": {"$gte": dt_from.timestamp()}},
                {"created_at_ts": {"$lte": dt_to.timestamp()}},
            ]
        }

    def test_empty_extension_list_ignored(self):
        assert build_where(extension=[]) is None

    def test_date_both_filters(self):
        dt_from = datetime(2025, 1, 1, tzinfo=timezone.utc)
        dt_to = datetime(2025, 12, 31, tzinfo=timezone.utc)
        result = build_where(date_from=dt_from, date_to=dt_to)
        assert result == {
            "$and": [
                {"created_at_ts": {"$gte": dt_from.timestamp()}},
                {"created_at_ts": {"$lte": dt_to.timestamp()}},
            ]
        }


class TestNormalizeDistance:
    def test_zero_distance_returns_one(self):
        assert normalize_distance(0) == 1.0

    def test_distance_one_returns_half(self):
        assert normalize_distance(1.0) == 0.5

    def test_large_distance_approaches_zero(self):
        result = normalize_distance(1000)
        assert result == round(1 / 1001, 4)

    def test_result_is_rounded_to_4_decimals(self):
        result = normalize_distance(0.333333)
        assert result == round(1 / (1 + 0.333333), 4)


class TestSearchRequest:
    def test_valid_request(self):
        req = SearchRequest(q="hello")
        assert req.q == "hello"
        assert req.k == 10
        assert req.extension is None

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError):
            SearchRequest(q="")

    def test_k_bounds(self):
        req = SearchRequest(q="test", k=1)
        assert req.k == 1
        req = SearchRequest(q="test", k=100)
        assert req.k == 100

    def test_k_too_low_rejected(self):
        with pytest.raises(ValueError):
            SearchRequest(q="test", k=0)

    def test_k_too_high_rejected(self):
        with pytest.raises(ValueError):
            SearchRequest(q="test", k=101)


class TestShapeSearchResponse:
    def _make_request(self, q="test", k=10, **kwargs):
        return SearchRequest(q=q, k=k, **kwargs)

    def _make_raw(self, ids, documents, metadatas, distances):
        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def test_empty_raw_returns_zero_results(self):
        req = self._make_request()
        result = shape_search_response(None, req)
        assert result["total_results"] == 0
        assert result["results"] == []
        assert result["files"] == []

    def test_empty_ids_returns_zero_results(self):
        req = self._make_request()
        raw = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        result = shape_search_response(raw, req)
        assert result["total_results"] == 0
        assert result["results"] == []

    def test_single_result(self):
        req = self._make_request()
        raw = self._make_raw(
            ids=["abc123"],
            documents=["the quick brown fox"],
            metadatas=[
                {"file_name": "a.txt", "extension": "txt", "created_at_ts": 1000.0}
            ],
            distances=[0.5],
        )
        result = shape_search_response(raw, req)

        assert result["total_results"] == 1
        assert result["results"][0]["rank"] == 1
        assert result["results"][0]["chunk_id"] == "abc123"
        assert result["results"][0]["chunk_text"] == "the quick brown fox"
        assert result["results"][0]["distance"] == 0.5
        assert result["results"][0]["score"] == normalize_distance(0.5)
        assert result["results"][0]["file"]["file_name"] == "a.txt"

    def test_multiple_results_ranked(self):
        req = self._make_request()
        raw = self._make_raw(
            ids=["c1", "c2", "c3"],
            documents=["text1", "text2", "text3"],
            metadatas=[
                {"file_name": "a.txt"},
                {"file_name": "b.txt"},
                {"file_name": "a.txt"},
            ],
            distances=[0.3, 0.7, 0.1],
        )
        result = shape_search_response(raw, req)

        assert result["total_results"] == 3
        ranks = [r["rank"] for r in result["results"]]
        assert ranks == [1, 2, 3]

    def test_file_summary_groups_by_name(self):
        req = self._make_request()
        raw = self._make_raw(
            ids=["c1", "c2", "c3"],
            documents=["t1", "t2", "t3"],
            metadatas=[
                {"file_name": "a.txt"},
                {"file_name": "a.txt"},
                {"file_name": "b.txt"},
            ],
            distances=[0.2, 0.8, 0.4],
        )
        result = shape_search_response(raw, req)

        assert len(result["files"]) == 2
        file_names = [f["file_name"] for f in result["files"]]
        assert "a.txt" in file_names
        assert "b.txt" in file_names
        a_file = next(f for f in result["files"] if f["file_name"] == "a.txt")
        assert a_file["hit_count"] == 2

    def test_file_summary_best_score(self):
        req = self._make_request()
        raw = self._make_raw(
            ids=["c1", "c2"],
            documents=["t1", "t2"],
            metadatas=[
                {"file_name": "a.txt"},
                {"file_name": "a.txt"},
            ],
            distances=[0.2, 0.8],
        )
        result = shape_search_response(raw, req)

        a_file = result["files"][0]
        assert a_file["best_score"] == normalize_distance(0.2)
        assert a_file["best_distance"] == 0.2

    def test_files_sorted_by_best_score_desc(self):
        req = self._make_request()
        raw = self._make_raw(
            ids=["c1", "c2", "c3"],
            documents=["t1", "t2", "t3"],
            metadatas=[
                {"file_name": "bad.txt"},
                {"file_name": "good.txt"},
                {"file_name": "mid.txt"},
            ],
            distances=[0.9, 0.1, 0.5],
        )
        result = shape_search_response(raw, req)

        file_names = [f["file_name"] for f in result["files"]]
        assert file_names == ["good.txt", "mid.txt", "bad.txt"]

    def test_filters_reflected_in_response(self):
        req = self._make_request(
            extension=["txt"],
            date_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
            date_to=datetime(2025, 6, 30, tzinfo=timezone.utc),
        )
        raw = self._make_raw(ids=[], documents=[], metadatas=[], distances=[])
        result = shape_search_response(raw, req)

        filters = result["filters"]
        assert filters["extension"] == ["txt"]
        assert "2025-01-01" in filters["date_from"]
        assert "2025-06-30" in filters["date_to"]

    def test_missing_metadata_defaults(self):
        req = self._make_request()
        raw = self._make_raw(
            ids=["c1"],
            documents=["text"],
            metadatas=[{}],
            distances=[0.5],
        )
        result = shape_search_response(raw, req)

        assert result["results"][0]["file"]["file_name"] == "unknown"
        assert result["results"][0]["file"]["extension"] is None

    def test_no_filters_in_response_when_none_provided(self):
        req = self._make_request()
        raw = self._make_raw(ids=[], documents=[], metadatas=[], distances=[])
        result = shape_search_response(raw, req)

        assert result["filters"]["extension"] is None
        assert result["filters"]["date_from"] is None
        assert result["filters"]["date_to"] is None


class TestSameNameFiles:
    def _make_request(self):
        return SearchRequest(q="test", k=10)

    def test_same_name_different_file_ids_stay_separate(self):
        """Two uploads named report.pdf are summarised as two files."""
        raw = {
            "ids": [["c1", "c2", "c3"]],
            "documents": [["t1", "t2", "t3"]],
            "metadatas": [
                [
                    {"file_name": "report.pdf", "file_id": 1},
                    {"file_name": "report.pdf", "file_id": 2},
                    {"file_name": "report.pdf", "file_id": 1},
                ]
            ],
            "distances": [[0.2, 0.9, 0.4]],
        }
        result = shape_search_response(raw, self._make_request())

        assert len(result["files"]) == 2
        by_id = {f["file_id"]: f for f in result["files"]}
        assert by_id[1]["hit_count"] == 2
        assert by_id[2]["hit_count"] == 1
        assert by_id[1]["best_distance"] == 0.2

    def test_summary_entries_carry_file_id(self):
        """Each file summary and result exposes its file_id."""
        raw = {
            "ids": [["c1"]],
            "documents": [["t1"]],
            "metadatas": [[{"file_name": "a.txt", "file_id": 9}]],
            "distances": [[0.5]],
        }
        result = shape_search_response(raw, self._make_request())

        assert result["files"][0]["file_id"] == 9
        assert result["results"][0]["file"]["file_id"] == 9

    def test_legacy_chunks_without_file_id_group_by_name(self):
        """Chunks indexed before file_id existed still group by file name."""
        raw = {
            "ids": [["c1", "c2"]],
            "documents": [["t1", "t2"]],
            "metadatas": [[{"file_name": "old.txt"}, {"file_name": "old.txt"}]],
            "distances": [[0.1, 0.3]],
        }
        result = shape_search_response(raw, self._make_request())

        assert len(result["files"]) == 1
        assert result["files"][0]["hit_count"] == 2
        assert result["files"][0]["file_id"] is None
