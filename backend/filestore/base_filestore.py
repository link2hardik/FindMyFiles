from abc import abstractmethod, ABC
import io
from pathlib import Path


class FileStoreError(Exception):
    "Base exception for all FileStore errors."

    pass


class IngestableFile:
    def __init__(self, file_obj: io.IOBase, name: str | None = None, file_id :int | None = None):
        """
        Accepts any file-like object passed from the application.
        """
        if not isinstance(file_obj, io.IOBase):
            raise FileStoreError("file_obj must be an io.IOBase instance")
        
        self.file_id = file_id
        self.file_obj = file_obj
        self.file_name = name if name else getattr(file_obj, "name", "unknown_source")
        self.extension = Path(self.file_name).suffix[1:].lower() or "unknown"

    def get_file(self):
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "extension": self.extension,
            "file": self.file_obj,
        }


class BaseFileStore(ABC):
    def __init__(self, path: str | None = None):
        self.path = path

    @abstractmethod
    def get(self, id):
        "returns the file given the unique id of the file."
        pass

    @abstractmethod
    def store(self, file: IngestableFile) -> id:
        "stores file on the system and returns it's unique id"
        pass

    @abstractmethod
    def close(self) -> None:
        "releases resources owned by the file store"
        pass
