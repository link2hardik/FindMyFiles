from datetime import datetime
from sqlmodel import SQLModel, Field, Session, select
from sqlalchemy import func, create_engine
import os


class AppStateError(Exception):
    """Custom exception for AppState errors."""

    pass


class AppStateDB(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    file_type: str | None = None
    file_id: int | None = None

    add_timestamp: datetime | None = Field(
        sa_column_kwargs={"server_default": func.now()}
    )
    last_update_timestamp: datetime = Field(
        sa_column_kwargs={"server_default": func.now(), "onupdate": func.now()}
    )

    status: str = Field(default="pending")
    error_message: str | None = Field(default=None)


def row_to_dict(file_row):
    return {
        "file_id": file_row.file_id,
        "app_state_id": file_row.id,
        "file_name": file_row.name,
        "file_type": file_row.file_type,
        "add_timestamp": file_row.add_timestamp,
        "last_update_timestamp": file_row.last_update_timestamp,
        "status": file_row.status,
        "error_message": file_row.error_message,
    }


class AppState:
    def __init__(self, db_path="backend/data/app_data"):
        try:
            self.DATABASE_URL = f"sqlite:///{os.path.join(db_path, 'app_state.db')}"
            os.makedirs(db_path, exist_ok=True)
            self.engine = create_engine(
                self.DATABASE_URL,
                echo=True,
                connect_args={"check_same_thread": False},
            )
            with self.engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            SQLModel.metadata.create_all(self.engine)
        except Exception as e:
            raise AppStateError(f"Failed to initialize AppState Database: {e}") from e

    def close(self) -> None:
        self.engine.dispose()

    def insert_file(
        self,
        file_name: str,
        file_type: str,
        file_status: str,
        file_id: int | None = None,
    ):
        try:
            if not all([file_name, file_type, file_status]):
                raise ValueError("need all fields file name, status, type")

            with Session(self.engine) as session:
                row = AppStateDB(
                    file_type=file_type,
                    status=file_status,
                    name=file_name,
                    file_id=file_id,
                )
                session.add(row)
                session.commit()
                session.refresh(row)

                return row.id
        except Exception as e:
            raise AppStateError(f"Failed to insert file: {e}") from e

    def update_file(self, file_id: int, status=None, error_msg=None):
        try:
            with Session(self.engine) as session:
                statement = select(AppStateDB).where(AppStateDB.id == file_id)
                file_row = session.exec(statement).one_or_none()

                if not file_row:
                    raise FileNotFoundError(
                        f"No file entry found in database for ID: {file_id}"
                    )

                if status is not None:
                    file_row.status = status
                if error_msg is not None:
                    file_row.error_message = error_msg
                session.commit()
        except Exception as e:
            raise AppStateError(f"Failed to update file: {e}") from e

    def get_status_by_id(self, file_id):
        try:
            with Session(self.engine) as session:
                statement = select(AppStateDB).where(AppStateDB.id == file_id)
                file_row = session.exec(statement).one_or_none()
        except Exception as e:
            raise AppStateError(f"Failed to get status by id: {e}") from e

        if not file_row:
            raise FileNotFoundError(
                f"No file entry found in database for ID: {file_id}"
            )

        return row_to_dict(file_row)

    def get_status_all(self):
        try:
            with Session(self.engine) as session:
                statement = select(AppStateDB)
                file_rows = session.exec(statement).all()

                return [row_to_dict(row) for row in file_rows]
        except Exception as e:
            raise AppStateError(f"Failed to get status all: {e}") from e
