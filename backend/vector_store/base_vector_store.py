from abc import abstractmethod, ABC
import hashlib


class VectorStoreError(Exception):
    """
    Custom exception for VectorStore errors.
    """

    pass


class BaseVectorStore(ABC):
    def __init__(self, path: str = None):
        self.path = path

    @abstractmethod
    def add(self, chunks: list[str], metadata: dict = None):
        "store chunks in the vector database."

    @abstractmethod
    def get(self, query: str, k=10, constraints: dict = None):
        "return the relevant chunks."
        # TODO: Standardise the output format for any vectorstore implimentation

    @abstractmethod
    def close(self) -> None:
        "releases resources owned by the vector store"
        pass

    @staticmethod
    def get_md5(chunks: list[str]) -> list[str]:
        "Get the md5 for chunk to serve as unique ids for vector store."
        try:
            return [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in chunks]
        except Exception as e:
            raise VectorStoreError(f"Failed to get md5 for chunks: {e}") from e

    @classmethod
    def get_chunk_ids(cls, chunks: list[str], metadatas: list) -> list[str]:
        """Build IDs unique per (file, chunk position, content).

        Text hashes alone collide when the same text appears twice in one
        file or in two files. Scoping by file_id and position keeps every
        chunk distinct while still reflecting its content.
        """
        hashes = cls.get_md5(chunks)
        return [
            f"{meta.file_id}:{index}:{digest}"
            for index, (meta, digest) in enumerate(zip(metadatas, hashes, strict=True))
        ]
