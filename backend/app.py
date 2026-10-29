from fastapi import FastAPI, File, HTTPException, Response, UploadFile
import asyncio
import logging
from typing import Annotated
from backend.filestore.base_filestore import FileStoreError, IngestableFile
from backend.filestore.local_filestore import DataNotFoundError
from backend.app_state import AppStateError
from backend.vector_store.base_vector_store import VectorStoreError
from backend.ingestors.base_ingestor import BaseIngestor
from backend.tasks import process_ingest_file
from backend.config import FILE_STORE, VECTOR_STORE, APP_STATE
from backend.search import SearchRequest, build_where, shape_search_response

app = FastAPI()
logger = logging.getLogger(__name__)

MEDIA_TYPES = {
    "txt": "text/plain",
    "md": "text/plain",
    "pdf": "application/pdf",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}


@app.get("/")
def statistics():
    "route to return the statistics of application, i.e. Ingestion status , health etc"
    try:
        return APP_STATE.get_status_all()
    except AppStateError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _ingest_upload(upload: UploadFile) -> dict:
    """Store one uploaded file and queue its ingestion.

    Never raises: per-file problems are captured in the returned dict so a
    batch keeps processing the remaining files. `status` is one of:

    - "queued":   stored, tracked, and dispatched to the Celery worker.
    - "rejected": client-side problem (unsupported extension).
    - "failed":   server-side problem while storing or dispatching.
    """
    result = {
        "file_name": upload.filename,
        "status": "queued",
        "task_id": None,
        "app_state_id": None,
        "file_id": None,
        "error": None,
    }

    try:
        ingestable_file = IngestableFile(upload.file, upload.filename)
    except FileStoreError as e:
        result["status"] = "failed"
        result["error"] = str(e)
        return result

    if ingestable_file.extension not in BaseIngestor.all_formats:
        result["status"] = "rejected"
        result["error"] = f"Unsupported file extension: {ingestable_file.extension}"
        return result

    try:
        file_id = await asyncio.to_thread(FILE_STORE.store, ingestable_file)
    except FileStoreError as e:
        result["status"] = "failed"
        result["error"] = str(e)
        return result
    result["file_id"] = file_id

    try:
        app_state_id = APP_STATE.insert_file(
            file_name=ingestable_file.file_name,
            file_type=ingestable_file.extension,
            file_status="Storage Complete",
            file_id=file_id,
        )
    except AppStateError as e:
        result["status"] = "failed"
        result["error"] = str(e)
        return result
    result["app_state_id"] = app_state_id

    try:
        task = process_ingest_file.delay(file_id, app_state_id)
    except Exception as e:
        # The file is stored but will never be processed; record that so
        # the row does not sit in "Storage Complete" indefinitely.
        try:
            APP_STATE.update_file(
                app_state_id,
                status="Ingestion Failed",
                error_msg=f"Failed to queue ingestion: {e}",
            )
        except AppStateError:
            logger.exception("Could not mark app_state %s as failed", app_state_id)
        result["status"] = "failed"
        result["error"] = str(e)
        return result
    result["task_id"] = task.id
    return result


@app.post("/upload/")
async def upload_file(file: Annotated[UploadFile, File()]):
    result = await _ingest_upload(file)

    if result["status"] == "queued":
        return {
            "message": "Task has been sent to the background worker pool",
            "task_id": result["task_id"],
            "app_state_id": result["app_state_id"],
            "file_id": result["file_id"],
        }

    if result["status"] == "rejected":
        raise HTTPException(
            status_code=422,
            detail={
                "message": result["error"],
                "acceptable_formats": list(BaseIngestor.all_formats),
            },
        )

    raise HTTPException(status_code=500, detail=result["error"])


@app.post("/upload/batch/")
async def upload_files(files: Annotated[list[UploadFile], File()]):
    results = [await _ingest_upload(upload) for upload in files]

    queued = [r for r in results if r["status"] == "queued"]
    rejected = [r for r in results if r["status"] == "rejected"]

    if not queued:
        if len(rejected) == len(results):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "No files with supported extensions",
                    "acceptable_formats": list(BaseIngestor.all_formats),
                },
            )
        raise HTTPException(
            status_code=500,
            detail={"message": "All uploads failed", "files": results},
        )

    return {
        "message": "Batch upload complete",
        "queued_count": len(queued),
        "rejected_count": len(rejected),
        "failed_count": len(results) - len(queued) - len(rejected),
        "files": results,
    }


@app.post("/search/")
def search(request: SearchRequest):
    if request.extension:
        unknown = [
            ext for ext in request.extension if ext not in BaseIngestor.all_formats
        ]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Unsupported file extension(s)",
                    "unknown_extensions": unknown,
                    "acceptable_formats": list(BaseIngestor.all_formats),
                },
            )

    if request.date_from and request.date_to and request.date_from > request.date_to:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "date_from must be before date_to",
                "date_from": request.date_from.isoformat(),
                "date_to": request.date_to.isoformat(),
            },
        )

    where = build_where(request.extension, request.date_from, request.date_to)
    try:
        raw = VECTOR_STORE.get(request.q, k=request.k, constraints=where)
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return shape_search_response(raw, request)


@app.get("/files/")
def list_files():
    try:
        rows = APP_STATE.get_status_all()
    except AppStateError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    for row in rows:
        file_id = row.get("file_id")
        row["file_size"] = None
        if file_id is not None:
            try:
                meta = FILE_STORE.get_metadata(file_id)
                row["file_size"] = meta["size"]
            except FileStoreError:
                pass

        if row.get("add_timestamp") and row.get("last_update_timestamp"):
            row["duration_seconds"] = round(
                (row["last_update_timestamp"] - row["add_timestamp"]).total_seconds(), 2
            )
        else:
            row["duration_seconds"] = None
    return rows


@app.get("/file/{file_id}")
def get_file_contents(file_id: int):
    try:
        ingestable_file = FILE_STORE.get(file_id)
    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileStoreError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    media_type = MEDIA_TYPES.get(ingestable_file.extension, "application/octet-stream")
    file_obj = ingestable_file.file_obj
    try:
        file_obj.seek(0)
        content = file_obj.read()
    finally:
        file_obj.close()

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{ingestable_file.file_name}"'
        },
    )


@app.get("/files/{app_state_id}")
def file_status(app_state_id: int):
    """Ingestion status by AppState ID (the `app_state_id` from POST /upload/).

    The response's `file_id` is the file-store ID used by GET /file/{file_id}.
    """
    try:
        return APP_STATE.get_status_by_id(app_state_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/formats")
def get_formats():
    return list(BaseIngestor.all_formats)
