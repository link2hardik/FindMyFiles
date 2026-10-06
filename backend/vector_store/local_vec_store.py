from .base_vector_store import BaseVectorStore, VectorStoreError
from ..ingestors.base_ingestor import Metadata
import chromadb
from functools import cached_property


class ChromaDBError(VectorStoreError):
    pass


class ChromaDBVectorStore(BaseVectorStore):
    @cached_property
    def client(self):
        try:
            if self._client_factory:
                return self._client_factory()
            return chromadb.PersistentClient(path=self.path)
        except ChromaDBError as e:
            raise e
        except Exception as e:
            raise ChromaDBError(f"Failed to create client: {e}") from e

    @cached_property
    def collection(self):
        try:
            kwargs = {
                "name": "my-collection",
                "metadata": {"description": "vector-store for data"},
            }
            if self._embedding_function:
                kwargs["embedding_function"] = self._embedding_function
            return self.client.get_or_create_collection(**kwargs)
        except ChromaDBError as e:
            raise e
        except Exception as e:
            raise ChromaDBError(f"Failed to get or create collection: {e}") from e

    def __init__(
        self,
        path="backend/data/vecstore/",
        client_factory=None,
        embedding_function=None,
    ):
        super().__init__(path)
        self._client_factory = client_factory
        self._embedding_function = embedding_function

    def close(self) -> None:
        self.__dict__.pop("collection", None)
        self.__dict__.pop("client", None)

    def add(self, chunks: list[str], metadatas: list[Metadata]):
        try:
            if not chunks or not metadatas:
                raise VectorStoreError(
                    f"Chunks and metadatas must not be empty len(chunks)={len(chunks)} len(metadatas)={len(metadatas)}"
                )

            if len(chunks) != len(metadatas):
                raise VectorStoreError(
                    f"Chunks and metadatas must have the same length len(chunks)={len(chunks)} len(metadatas)={len(metadatas)}"
                )

            if not all(isinstance(m, Metadata) for m in metadatas):
                raise VectorStoreError("All metadatas must be instances of Metadata")

            metadata_dicts = [m.model_dump() for m in metadatas]

            self.collection.add(
                ids=self.get_md5(chunks), documents=chunks, metadatas=metadata_dicts
            )
        except VectorStoreError as e:
            raise e
        except Exception as e:
            raise ChromaDBError(f"Failed to add chunks: {e}") from e

    def get(self, query, k=10, constraints=None):
        try:
            kwargs = {
                "query_texts": [query],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"],
            }
            if constraints:
                kwargs["where"] = constraints
            return self.collection.query(**kwargs)
        except VectorStoreError as e:
            raise e
        except Exception as e:
            raise ChromaDBError(f"Failed to get chunks: {e}") from e
