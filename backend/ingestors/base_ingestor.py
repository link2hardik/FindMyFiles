from abc import abstractmethod, ABC
from backend.filestore.base_filestore import IngestableFile
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class Metadata(BaseModel):
    file_id:int
    file_name: str
    type: str
    extension: str
    created_at_ts: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )


class IngestionError(Exception):
    "Generic Ingestion Error when extracting text from a file failed."

    pass


class BaseIngestor(ABC):
    "Class to Ingest file into chunks and metadata."

    all_formats = set()  # to store all accepted formats
    ingestor_map = dict()  # stores which format corrosponds to which ingestor

    def __init__(
        self,
        type: str | None = None,
        accepted_formats: list[str] | None = None,
        name: str = "default",
    ):
        self.name = name
        self.type = type
        self.accepted_formats = accepted_formats or []
        try:
            self.update_ingestor_map()  # register this ingestor with the format map (used by child classes)
        except IngestionError as e:
            raise e
        except Exception as e:
            raise IngestionError(f"Failed to update ingestor map: {str(e)}") from e

    def update_ingestor_map(self):
        for format in self.accepted_formats:
            existing = BaseIngestor.ingestor_map.get(format)
            if existing is not None and existing.__class__ is not self.__class__:
                raise IngestionError(
                    f"Format {format} is already registered by {existing.name}."
                )
        for format in self.accepted_formats:
            BaseIngestor.all_formats.add(format)
            BaseIngestor.ingestor_map[format] = self

    @abstractmethod
    def extract_text(self, file: IngestableFile) -> tuple[str, Metadata]:
        """
        Given the media object, extract the relevant text and metadata.
        """
        pass

    def extract_metadata(self, file: IngestableFile) -> Metadata:
        try:
            return Metadata(
                file_id = file.file_id,
                file_name=file.file_name,
                type=self.type,
                extension=file.extension
            )
        except Exception as e:
            raise IngestionError(f"Failed to extract metadata: {e}") from e
