import hashlib

import pytest
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents

from backend.ingestors.base_ingestor import Metadata
from backend.vector_store.base_vector_store import VectorStoreError
from backend.vector_store.local_vec_store import (
    ChromaDBError,
    ChromaDBVectorStore,
)

"""Comprehensive unit tests for ChromaDBVectorStore.

Success paths run against the REAL chromadb engine via
chromadb.EphemeralClient (in-memory, no server process, no disk), so
indexing, querying, and where-filtering are verified decisively. A
deterministic hash-based embedding function keeps tests fully offline and
fast. Error paths that a healthy client cannot produce are simulated via
the client_factory seam with minimal fakes, verifying our wrapping,
chaining, and validation logic.
"""


class DeterministicEmbedding(EmbeddingFunction):
    """Hash-based embedding so the real chromadb pipeline runs with zero
    downloads (chromadb's default ONNX model is ~80 MB and needs network)."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> list[list[float]]:
        return [
            [float(b) for b in hashlib.md5(chunk.encode("utf-8")).digest()]
            for chunk in input
        ]

    @staticmethod
    def name() -> str:
        return "deterministic-hash"

    def get_config(self) -> dict:
        return {"name": self.name()}

    @classmethod
    def build_from_config(cls, config: dict) -> "DeterministicEmbedding":
        return cls()


class FakeCollection:
    """Stand-in for a chromadb Collection that records add/query calls and
    can be configured to raise on demand."""

    def __init__(self, add_error=None, query_error=None):
        self.add_error = add_error
        self.query_error = query_error
        self.added_calls = []
        self.query_calls = []

    def add(self, ids, documents, metadatas):
        if self.add_error:
            raise self.add_error
        self.added_calls.append((ids, documents, metadatas))

    def query(self, **kwargs):
        if self.query_error:
            raise self.query_error
        self.query_calls.append(kwargs)
        return {"ids": [], "documents": [], "metadatas": [], "distances": []}


class FakeClient:
    """Stand-in for a chromadb Client that returns a FakeCollection and
    records the collection-creation arguments."""

    def __init__(self, create_error=None, collection=None):
        self.create_error = create_error
        self.collection = collection if collection is not None else FakeCollection()
        self.created_with = None

    def get_or_create_collection(self, name, metadata=None, **kwargs):
        if self.create_error:
            raise self.create_error
        self.created_with = {"name": name, "metadata": metadata, **kwargs}
        return self.collection


@pytest.fixture
def store(tmp_path):
    """A ChromaDBVectorStore wired to the real in-process chromadb engine,
    persisted per-test in tmp_path for isolation (EphemeralClient shares one
    process-wide in-memory database, which leaks state between tests)."""
    return ChromaDBVectorStore(
        path=str(tmp_path),
        client_factory=lambda: chromadb.PersistentClient(
            path=str(tmp_path / "chroma"),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        ),
        embedding_function=DeterministicEmbedding(),
    )


def make_store(fake_client):
    """Build a store wired to `fake_client` through the client_factory seam."""
    return ChromaDBVectorStore(path="unused", client_factory=lambda: fake_client)


def make_metadata(file_name="a.txt", extension="txt", created_at_ts=None, file_id=1):
    """Build a minimal valid Metadata instance for testing."""
    kwargs = {"type": "text"}
    if created_at_ts is not None:
        kwargs["created_at_ts"] = created_at_ts
    return Metadata(file_id=file_id, file_name=file_name, extension=extension, **kwargs)


def md5_ids(chunks):
    """Mirror BaseVectorStore.get_md5 for expected-id assertions."""
    return [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in chunks]


class TestAddBehavior:
    def test_add_then_query_returns_inserted_document(self, store):
        """A chunk added to the real in-memory engine is returned verbatim by
        get(), under its md5 id, proving the full add-index-query path."""
        chunks = ["the quick brown fox", "jumps over the lazy dog"]
        store.add(chunks, [make_metadata(), make_metadata("b.txt")])

        results = store.get(query=chunks[0], k=1)

        assert results["documents"][0] == [chunks[0]]
        assert results["ids"][0] == md5_ids([chunks[0]])

    def test_collection_count_matches_added_chunks(self, store):
        """The real collection count equals the number of added chunks."""
        chunks = [f"chunk number {i}" for i in range(5)]
        store.add(chunks, [make_metadata()] * len(chunks))

        assert store.collection.count() == len(chunks)

    def test_get_returns_top_k_results(self, store):
        """get(k=n) returns exactly n results when enough chunks exist."""
        chunks = [f"document {i}" for i in range(6)]
        store.add(chunks, [make_metadata()] * len(chunks))

        results = store.get(query=chunks[0], k=3)

        assert len(results["ids"][0]) == 3

    def test_metadata_round_trip(self, store):
        """Query-metadatas equal the dumped Metadata dicts that were added."""
        chunks = ["alpha", "beta"]
        metadatas = [
            make_metadata("a.txt", created_at_ts=1000.0),
            make_metadata("b.pdf", extension="pdf", created_at_ts=2000.0),
        ]
        store.add(chunks, metadatas)

        results = store.get(query=chunks[0], k=1)
        metadata_dicts = [m.model_dump() for m in metadatas]

        assert results["metadatas"][0] == [metadata_dicts[0]]

    def test_add_same_chunk_twice_deduplicates(self, store):
        """Identical chunks share an md5 id, so re-adding is an upsert and
        the collection count stays at the number of unique chunks."""
        store.add(["dup"], [make_metadata()])
        store.add(["dup"], [make_metadata("b.txt")])

        assert store.collection.count() == 1


class TestWhereFilters:
    def test_get_extension_where_filter(self, store):
        """The extension $in filter returns only matching documents from the
        real engine."""
        store.add(
            ["txt content", "pdf content"],
            [
                make_metadata("a.txt", extension="txt"),
                make_metadata("b.pdf", extension="pdf"),
            ],
        )

        results = store.get(
            query="content", k=5, constraints={"extension": {"$in": ["txt"]}}
        )

        assert len(results["ids"][0]) == 1
        assert results["metadatas"][0][0]["extension"] == "txt"

    def test_get_date_range_where_filter(self, store):
        """The created_at_ts $gte/$lte filters work against real metadata."""
        store.add(
            ["old", "new"],
            [make_metadata(created_at_ts=1000.0), make_metadata(created_at_ts=2000.0)],
        )

        results = store.get(
            query="content", k=5, constraints={"created_at_ts": {"$gte": 1500.0}}
        )

        assert len(results["ids"][0]) == 1
        assert results["metadatas"][0][0]["created_at_ts"] == 2000.0

    def test_get_combined_and_where_filter(self, store):
        """An $and combination of extension and date filters returns only
        documents satisfying both."""
        store.add(
            ["old txt", "new txt", "old pdf"],
            [
                make_metadata(extension="txt", created_at_ts=1000.0),
                make_metadata(extension="txt", created_at_ts=2000.0),
                make_metadata(extension="pdf", created_at_ts=1000.0),
            ],
        )

        results = store.get(
            query="content",
            k=5,
            constraints={
                "$and": [
                    {"extension": {"$in": ["txt"]}},
                    {"created_at_ts": {"$gte": 1500.0}},
                ]
            },
        )

        assert len(results["ids"][0]) == 1
        assert results["metadatas"][0][0]["file_name"].endswith(".txt")
        assert results["metadatas"][0][0]["created_at_ts"] == 2000.0


class TestValidation:
    def test_add_rejects_empty_chunks(self):
        """add() with an empty chunks list raises a typed error and never
        touches the collection."""
        store = make_store(FakeClient())

        with pytest.raises(VectorStoreError, match="must not be empty") as exc:
            store.add([], [make_metadata()])

        assert not isinstance(exc.value, ChromaDBError)
        assert store.client.collection.added_calls == []

    def test_add_rejects_length_mismatch(self):
        """add() with chunks/metadatas of different lengths raises a typed
        error naming both lengths."""
        store = make_store(FakeClient())

        with pytest.raises(VectorStoreError, match="same length") as exc:
            store.add(["a", "b"], [make_metadata()])

        assert not isinstance(exc.value, ChromaDBError)
        assert store.client.collection.added_calls == []

    def test_add_rejects_non_metadata_items(self):
        """add() with a non-Metadata item raises a clean typed error instead
        of an AttributeError from model_dump."""
        store = make_store(FakeClient())

        with pytest.raises(VectorStoreError, match="instances of Metadata") as exc:
            store.add(["a"], [{"file_name": "x"}])

        assert not isinstance(exc.value, ChromaDBError)
        assert store.client.collection.added_calls == []


class TestErrorWrapping:
    def test_add_collection_failure_wraps_as_chroma_db_error(self):
        """A collection.add failure surfaces as a chained ChromaDBError, not
        the raw exception."""
        fake = FakeClient(collection=FakeCollection(add_error=ValueError("boom")))

        with pytest.raises(ChromaDBError, match="Failed to add chunks") as exc:
            make_store(fake).add(["a"], [make_metadata()])

        assert isinstance(exc.value.__cause__, ValueError)

    def test_get_query_failure_wraps_as_chroma_db_error(self):
        """A collection.query failure surfaces as a chained ChromaDBError, not
        the raw exception."""
        fake = FakeClient(collection=FakeCollection(query_error=RuntimeError("boom")))

        with pytest.raises(ChromaDBError, match="Failed to get chunks") as exc:
            make_store(fake).get("q")

        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_add_get_md5_failure_keeps_vector_store_error(self, monkeypatch):
        """A get_md5 failure keeps its VectorStoreError type instead of being
        double-wrapped into ChromaDBError."""

        def failing_md5(chunks):
            raise VectorStoreError("md5 failed")

        monkeypatch.setattr(ChromaDBVectorStore, "get_md5", staticmethod(failing_md5))

        with pytest.raises(VectorStoreError, match="md5 failed") as exc:
            make_store(FakeClient()).add(["a"], [make_metadata()])

        assert not isinstance(exc.value, ChromaDBError)

    def test_client_factory_failure_wraps_as_chroma_db_error(self):
        """A client_factory that cannot connect surfaces as a chained
        ChromaDBError."""

        def factory():
            raise ConnectionError("refused")

        store = ChromaDBVectorStore(path="unused", client_factory=factory)

        with pytest.raises(ChromaDBError, match="Failed to create client") as exc:
            _ = store.client

        assert isinstance(exc.value.__cause__, ConnectionError)

    def test_failed_client_creation_is_retried_on_next_access(self):
        """cached_property does not cache exceptions: a failed client factory
        is called again on the next access."""
        calls = []

        def factory():
            calls.append(1)
            raise ConnectionError("refused")

        store = ChromaDBVectorStore(path="unused", client_factory=factory)

        with pytest.raises(ChromaDBError):
            _ = store.client
        with pytest.raises(ChromaDBError):
            _ = store.client

        assert len(calls) == 2

    def test_collection_creation_failure_wraps_as_chroma_db_error(self):
        """A get_or_create_collection failure surfaces as a chained
        ChromaDBError."""
        fake = FakeClient(create_error=ValueError("collection error"))

        with pytest.raises(
            ChromaDBError, match="Failed to get or create collection"
        ) as exc:
            _ = make_store(fake).collection

        assert isinstance(exc.value.__cause__, ValueError)

    def test_client_error_not_double_wrapped_by_collection(self):
        """A ChromaDBError from client creation propagates through the
        collection property unchanged, preserving its type and message."""
        error = ChromaDBError("Failed to create client")

        def factory():
            raise error

        store = ChromaDBVectorStore(path="unused", client_factory=factory)

        with pytest.raises(ChromaDBError) as exc:
            _ = store.collection

        assert exc.value is error


class TestClientSeam:
    def test_client_uses_provided_factory_and_is_cached(self):
        """The client_factory is called once per store instance; client and
        collection are cached across accesses."""
        fake = FakeClient()
        calls = []

        def factory():
            calls.append(1)
            return fake

        store = ChromaDBVectorStore(path="unused", client_factory=factory)

        assert store.client is fake
        assert store.client is fake
        assert store.collection is fake.collection
        assert len(calls) == 1

    def test_collection_created_with_default_name_and_description(self):
        """The collection is created with the hardcoded name and description
        unless an embedding_function is configured."""
        fake = FakeClient()
        _ = make_store(fake).collection

        assert fake.created_with == {
            "name": "my-collection",
            "metadata": {"description": "vector-store for data"},
        }

    def test_embedding_function_forwarded_to_collection(self):
        """When an embedding_function is configured it is passed to
        get_or_create_collection."""
        fake = FakeClient()
        embedding = DeterministicEmbedding()
        store = ChromaDBVectorStore(
            path="unused", client_factory=lambda: fake, embedding_function=embedding
        )

        _ = store.collection

        assert fake.created_with["embedding_function"] is embedding

    def test_default_client_factory_uses_persistent_client(self, monkeypatch):
        """Without an injected factory, client creation falls back to
        chromadb.PersistentClient with the configured path."""
        captured = {}

        class FakePersistentClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(chromadb, "PersistentClient", FakePersistentClient)

        store = ChromaDBVectorStore(path="/test/path")
        _ = store.client

        assert captured == {"path": "/test/path"}


class TestContracts:
    def test_chroma_db_error_is_vector_store_error(self):
        """ChromaDBError must subclass VectorStoreError so a single catch-all
        handles every vector-store failure."""
        assert issubclass(ChromaDBError, VectorStoreError)
