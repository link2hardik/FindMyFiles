import gradio as gr
import math
import os
import re
import shutil
import tempfile
import time
from datetime import datetime

import requests

BACKEND_URL = os.getenv("FINDMYFILES_BACKEND_URL", "http://127.0.0.1:8000")

REQUEST_TIMEOUT = (10, 300)
POLL_INTERVAL_SECONDS = 1.0
POLL_TIMEOUT_SECONDS = 600
PAGE_SIZE = 10
ID = os.getenv("USER_ID", "admin")
PASSWORD = os.getenv("PASSWORD", "123")

STATUS_ORDER = [
    "Storage Complete",
    "Ingestion Complete",
    "Chunking Complete",
    "Embedding Complete",
    "Ingestion Successful",
    "Ingestion Failed",
]
TERMINAL_STATUSES = {"Ingestion Successful", "Ingestion Failed"}

STATUS_STYLES = {
    "Storage Complete": "#1f6feb",
    "Ingestion Complete": "#1f6feb",
    "Chunking Complete": "#1f6feb",
    "Embedding Complete": "#1f6feb",
    "Ingestion Successful": "#2da44e",
    "Ingestion Failed": "#cf222e",
}


def get_formats():
    try:
        response = requests.get(f"{BACKEND_URL}/formats", timeout=(5, 15))
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def extract_error(response):
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return f"HTTP {response.status_code}"


def status_text(row):
    status = row.get("status", "unknown")
    lines = [f"Status: {status}"]
    if status == "Ingestion Failed":
        lines.append(f"Failure reason: {row.get('error_message') or 'Unknown'}")
    return "\n".join(lines)


def render_formats(formats):
    if not formats:
        return "Could not load supported formats from the backend (is it running?)."
    return "Supported formats: " + ", ".join(formats)


def upload_file(file_path):
    with open(file_path, "rb") as f:
        response = requests.post(
            url=f"{BACKEND_URL}/upload/",
            files={
                "file": (os.path.basename(file_path), f, "application/octet-stream")
            },
            timeout=REQUEST_TIMEOUT,
        )
    return response


def poll_file_status(app_state_id, progress):
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    seen = set()
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SECONDS)
        try:
            response = requests.get(
                f"{BACKEND_URL}/files/{app_state_id}", timeout=(5, 15)
            )
        except requests.RequestException as e:
            yield f"Error: failed to reach backend while polling: {e}"
            return
        if response.status_code == 404:
            yield (
                f"Error: no status entry found for this upload "
                f"(`/files/{app_state_id}` returned 404)."
            )
            return
        if response.status_code != 200:
            yield f"Error: status request failed: {extract_error(response)}"
            return

        row = response.json()
        status = row.get("status", "unknown")

        if status in STATUS_ORDER and status not in seen:
            seen.add(status)
            fraction = STATUS_ORDER.index(status) / (len(STATUS_ORDER) - 1)
            progress(fraction, desc=status)

        yield status_text(row)

        if status in TERMINAL_STATUSES:
            return

    yield (
        f"Timeout: status did not reach a terminal state after "
        f"{POLL_TIMEOUT_SECONDS}s. Check the worker queue."
    )


def ingest_file(file_path, progress=gr.Progress()):  # noqa: B008
    if not file_path:
        return "Error: no file selected.", None

    file_name = os.path.basename(file_path)
    extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    supported = get_formats()
    if supported and extension not in supported:
        return (
            f"Error: `{extension or '(no extension)'}` is not a supported format. "
            f"Supported: {', '.join(supported)}",
            None,
        )

    try:
        response = upload_file(file_path)
    except requests.RequestException as e:
        return f"Error: upload failed: {e}", None

    if response.status_code != 200:
        return f"Error: upload failed: {extract_error(response)}", None

    try:
        job = response.json()
    except ValueError:
        return "Error: unexpected response from backend.", None

    if "app_state_id" not in job:
        formats = job.get("acceptable_formats") or supported
        return (
            f"Error: `{extension or '(no extension)'}` is not a supported format. "
            f"Supported: {', '.join(formats)}",
            None,
        )

    app_state_id = job["app_state_id"]

    for item in poll_file_status(app_state_id, progress):
        yield item, job


def build_ingest_tab():
    with gr.Tab("Ingest"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "## Ingest a file\n"
                    "Upload a file to the backend. It is stored immediately, then "
                    "asynchronously extracted, chunked and embedded by the worker pool."
                )
                gr.Markdown(value=render_formats(get_formats()))
                file_input = gr.File(label="File to ingest", type="filepath")
                upload_button = gr.Button("Upload & Ingest", variant="primary")
            with gr.Column(scale=1):
                status_box = gr.Textbox(
                    label="Ingestion Status", interactive=False, lines=3
                )
                details_box = gr.JSON(label="Job Response")
        upload_button.click(
            ingest_file,
            inputs=[file_input],
            outputs=[status_box, details_box],
        )


def format_ts(ts):
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return ""


def format_size(size):
    if not size:
        return ""
    size = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def get_file_metadata_map():
    try:
        response = requests.get(f"{BACKEND_URL}/files/", timeout=(5, 15))
        response.raise_for_status()
    except requests.RequestException:
        return {}
    return {
        row["file_id"]: row for row in response.json() if row.get("file_id") is not None
    }


def slice_chunks_page(results, page):
    total_pages = max(1, math.ceil(len(results) / PAGE_SIZE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    slice_ = results[start : start + PAGE_SIZE]
    rows = [
        [
            hit["rank"],
            hit["file"]["file_name"],
            hit["score"],
            hit["distance"],
            format_ts(hit["file"].get("created_at_ts")),
            hit["chunk_text"],
        ]
        for hit in slice_
    ]
    return rows, page, total_pages


def search_files(query, k, date_from, date_to, selected_formats):
    if not query or not query.strip():
        raise gr.Error("Please enter a query.")
    if date_from and date_to and date_from > date_to:
        raise gr.Error("Date from must be before date to.")

    payload = {
        "q": query.strip(),
        "k": int(k),
        "extension": selected_formats or None,
        "date_from": date_from.isoformat() if date_from else None,
        "date_to": date_to.isoformat() if date_to else None,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/search/", json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as e:
        raise gr.Error(f"Search failed: {e}") from e
    if response.status_code != 200:
        raise gr.Error(f"Search failed: {extract_error(response)}")

    data = response.json()
    results = data.get("results", [])
    file_summary = data.get("files", [])

    filters = data.get("filters", {})
    filter_parts = []
    if filters.get("date_from"):
        filter_parts.append(f"from {filters['date_from'][:10]}")
    if filters.get("date_to"):
        filter_parts.append(f"to {filters['date_to'][:10]}")
    if filters.get("extension"):
        filter_parts.append(", ".join(filters["extension"]))
    filter_text = f" ({', '.join(filter_parts)})" if filter_parts else ""

    summary = (
        f"**Query:** `{data.get('query', '')}`   **Results:** {data.get('total_results', 0)}"
        f"{filter_text}"
    )

    metadata_map = get_file_metadata_map()
    file_rows = []
    download_choices = []
    seen_ids = set()
    for entry in file_summary:
        name = entry["file_name"]
        file_id = entry.get("file_id")
        meta = metadata_map.get(file_id, {}) if file_id is not None else {}
        if file_id is not None and file_id not in seen_ids:
            seen_ids.add(file_id)
            download_choices.append((name, file_id))
        file_rows.append(
            [
                name,
                meta.get("file_type") or "",
                format_size(meta.get("file_size")),
                entry["hit_count"],
                entry["best_score"],
            ]
        )

    rows, page, total_pages = slice_chunks_page(results, 1)

    return (
        summary,
        file_rows,
        gr.update(choices=download_choices, value=None),
        rows,
        f"Page {page} of {total_pages}",
        results,
        1,
    )


def go_to_page(results, page, delta):
    if not results:
        return [], "Page 1 of 1", 1
    rows, page, total_pages = slice_chunks_page(results, page + delta)
    return rows, f"Page {page} of {total_pages}", page


def next_page(results, page):
    return go_to_page(results, page, 1)


def prev_page(results, page):
    return go_to_page(results, page, -1)


def download_file(file_id):
    if not file_id:
        raise gr.Error("Select a file first.")
    try:
        response = requests.get(
            f"{BACKEND_URL}/file/{file_id}", timeout=REQUEST_TIMEOUT, stream=True
        )
    except requests.RequestException as e:
        raise gr.Error(f"Download failed: {e}") from e
    if response.status_code != 200:
        raise gr.Error(f"Download failed: {extract_error(response)}")

    match = re.search(
        r'filename="([^"]+)"', response.headers.get("Content-Disposition", "")
    )
    name = match.group(1) if match else "download.bin"
    fd, path = tempfile.mkstemp(prefix="findmyfiles_", suffix=os.path.splitext(name)[1])
    with os.fdopen(fd, "wb") as out:
        shutil.copyfileobj(response.raw, out)
    return path


def build_retrieve_tab():
    with gr.Tab("Retrieve"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Search your files")
                query = gr.Textbox(
                    label="Query",
                    lines=2,
                    placeholder="Natural language query, e.g. Q3 sales report",
                )
                k = gr.Slider(
                    label="Number of results (k)",
                    minimum=1,
                    maximum=100,
                    value=10,
                    step=1,
                )
                date_from = gr.DateTime(
                    label="Date from", include_time=False, type="datetime"
                )
                date_to = gr.DateTime(
                    label="Date to", include_time=False, type="datetime"
                )
                formats = gr.CheckboxGroup(
                    label="File formats",
                    choices=supported_formats,
                    value=supported_formats,
                )
                search_button = gr.Button("Search", variant="primary")
            with gr.Column(scale=2):
                summary_box = gr.Markdown("")
                files_table = gr.Dataframe(
                    label="Files",
                    headers=["File", "Type", "Size", "Hits", "Best Score"],
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    file_dropdown = gr.Dropdown(
                        label="Download a file", choices=[], interactive=True
                    )
                    download_button = gr.Button("Download")
                downloaded_file = gr.File(label="Downloaded file", interactive=False)
                gr.Markdown("### Matching chunks")
                chunks_table = gr.Dataframe(
                    label="Chunks",
                    headers=["Rank", "File", "Score", "Distance", "Added", "Context"],
                    interactive=False,
                    wrap=True,
                    max_height=400,
                )
                with gr.Row():
                    prev_button = gr.Button("Previous")
                    page_text = gr.Textbox(
                        value="", show_label=False, interactive=False
                    )
                    next_button = gr.Button("Next")

        results_state = gr.State([])
        page_state = gr.State(1)

        search_button.click(
            search_files,
            inputs=[query, k, date_from, date_to, formats],
            outputs=[
                summary_box,
                files_table,
                file_dropdown,
                chunks_table,
                page_text,
                results_state,
                page_state,
            ],
        )
        next_button.click(
            next_page,
            inputs=[results_state, page_state],
            outputs=[chunks_table, page_text, page_state],
        )
        prev_button.click(
            prev_page,
            inputs=[results_state, page_state],
            outputs=[chunks_table, page_text, page_state],
        )
        download_button.click(
            download_file, inputs=[file_dropdown], outputs=[downloaded_file]
        )


def format_timestamp(ts):
    if not ts:
        return ""
    return str(ts)[:16].replace("T", " ")


def colored_status(status):
    color = STATUS_STYLES.get(status)
    if not color:
        return str(status or "")
    return f"<span style='color:{color};font-weight:600'>{status}</span>"


def load_stats():
    try:
        response = requests.get(f"{BACKEND_URL}/files/", timeout=(5, 15))
    except requests.RequestException as e:
        raise gr.Error(f"Failed to load stats: {e}") from e
    if response.status_code != 200:
        raise gr.Error(f"Failed to load stats: {extract_error(response)}")

    rows = response.json()
    rows.sort(key=lambda r: r.get("add_timestamp") or "", reverse=True)

    total = len(rows)
    successful = sum(1 for r in rows if r.get("status") == "Ingestion Successful")
    failed = sum(1 for r in rows if r.get("status") == "Ingestion Failed")
    in_progress = total - successful - failed

    file_rows = [
        [
            r.get("file_name"),
            r.get("file_type") or "",
            format_size(r.get("file_size")),
            colored_status(r.get("status")),
            r.get("error_message") or "",
            r.get("duration_seconds"),
            format_timestamp(r.get("add_timestamp")),
            format_timestamp(r.get("last_update_timestamp")),
        ]
        for r in rows
    ]

    return (
        f"Total: {total}",
        f"Successful: {successful}",
        f"Failed: {failed}",
        f"In progress: {in_progress}",
        file_rows,
    )


def build_stats_tab():
    with gr.Tab("Stats"):
        with gr.Row():
            total_label = gr.Label("Total: 0")
            successful_label = gr.Label("Successful: 0")
            failed_label = gr.Label("Failed: 0")
            in_progress_label = gr.Label("In progress: 0")
        refresh_button = gr.Button("Refresh", variant="primary")
        stats_table = gr.Dataframe(
            label="Files",
            headers=[
                "File",
                "Type",
                "Size",
                "Status",
                "Error",
                "Time Taken (s)",
                "Added",
                "Last Updated",
            ],
            datatype=["str", "str", "str", "markdown", "str", "number", "str", "str"],
            interactive=False,
            wrap=True,
            show_search="filter",
        )
        refresh_button.click(
            load_stats,
            outputs=[
                total_label,
                successful_label,
                failed_label,
                in_progress_label,
                stats_table,
            ],
        )
        return [
            total_label,
            successful_label,
            failed_label,
            in_progress_label,
            stats_table,
        ]


supported_formats = get_formats()

with gr.Blocks(title="FindMyFiles") as demo:
    build_ingest_tab()
    build_retrieve_tab()
    stats_outputs = build_stats_tab()
    demo.load(load_stats, outputs=stats_outputs)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", server_port=7860, share=False, auth=(ID, PASSWORD)
    )
