from .base_filestore import BaseFileStore, IngestableFile, FileStoreError
from datetime import datetime
from sqlmodel import Field, SQLModel, create_engine, Session, text, select
import hashlib
import os
import shutil


class DataNotFoundError(FileStoreError):
    """
    Raised when the entry for a file is not found in the metadata
    database, because the file has not been ingested yet, or some other reason.
    """

    pass


class FileDB(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    original_name: str
    md5_name: str
    type: str | None = None
    size: int | None = None
    date_added: datetime = Field(
        sa_column_kwargs={"server_default": text("(CURRENT_TIMESTAMP)")}
    )


def md5_hasher(file_obj):
    file_obj.seek(0)
    hasher = hashlib.md5()
    while chunk := file_obj.read(8192):
        hasher.update(chunk)
    file_obj.seek(0)
    return hasher.hexdigest()


class LocalSQLiteFileStore(BaseFileStore):
    def __init__(self, path="backend/data"):
        super().__init__(path)

        try:
            DATABASE_URL = f"sqlite:///{os.path.join(self.path, 'file_metadata.db')}"
            os.makedirs(self.path, exist_ok=True)

            self.engine = create_engine(DATABASE_URL, echo=True)

            SQLModel.metadata.create_all(self.engine)
        except Exception as e:
            raise FileStoreError(
                f"Failed to create/initialize database (file_metadata.db): {e}"
            ) from e

    def close(self) -> None:
        self.engine.dispose()

    def get_metadata(self, id):
        with Session(self.engine) as session:
            try:
                statement = select(FileDB).where(FileDB.id == id)
                file_row = session.exec(statement).one_or_none()
            except Exception as e:
                raise FileStoreError(
                    f"Failed to retrieve file from database: {e}"
                ) from e

            if not file_row:
                raise DataNotFoundError(f"No file entry found in database for ID: {id}")

            return {
                "file_id": file_row.id,
                "file_name": file_row.original_name,
                "md5_name": file_row.md5_name,
                "type": file_row.type,
                "size": file_row.size,
                "date_added": file_row.date_added,
            }

    def get(self, id):
        with Session(self.engine) as session:
            try:
                statement = select(FileDB).where(FileDB.id == id)
                file_row = session.exec(statement).one_or_none()
            except Exception as e:
                raise FileStoreError(
                    f"Failed to retrieve file from database: {e}"
                ) from e

            if not file_row:
                raise DataNotFoundError(f"No file entry found in database for ID: {id}")

            os.makedirs(self.path, exist_ok=True)
            file_path = os.path.join(
                self.path, f"{file_row.md5_name}.{file_row.type or 'unknown'}"
            )

            try:
                f = open(file_path, "rb")
                return IngestableFile(file_obj=f, name=file_row.original_name, file_id=id)
            except Exception as e:
                raise FileStoreError(
                    f"Failed to open file from disk at {file_path}: {e}"
                ) from e

    def store(self, file: IngestableFile):

        try:
            md5_name = md5_hasher(file.file_obj)
            file_type = file.extension if file.extension else "txt"
        except Exception as e:
            raise FileStoreError(f"Failed to compute MD5 hash: {e}") from e

        # store metadata in database

        try:
            file_row = FileDB(
                original_name=file.file_name,
                md5_name=md5_name,
                type=file_type,
            )
        except Exception as e:
            raise FileStoreError(f"Failed to create database record: {e}") from e

        file.file_obj.seek(0)
        file_path = None

        with Session(self.engine) as session:
            try:
                session.add(file_row)

                # store file in data folder
                os.makedirs(self.path, exist_ok=True)
                file_path = os.path.join(
                    self.path, f"{file_row.md5_name}.{file_row.type}"
                )

                file.file_obj.seek(0)

                with open(file_path, "wb") as f_out:
                    shutil.copyfileobj(file.file_obj, f_out)

                file_row.size = os.path.getsize(file_path)

                session.commit()
                session.refresh(file_row)

                return file_row.id

            except Exception as e:
                session.rollback()
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as remove_e:
                        raise FileStoreError(
                            f"Failed to remove file from disk: {remove_e}"
                        ) from remove_e
                raise FileStoreError(
                    f"Database/Disk atomic operations sync failed: {e}"
                ) from e
