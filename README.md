# FindMyFiles

FindMyFiles is a multimodal semantic-search application for uploading files and finding them with natural-language queries. It extracts text from documents, images, and audio, stores searchable chunks in a vector database, and processes ingestion tasks asynchronously with Celery.

## Features

- Upload and search text, Markdown, PDF, image, and audio files.
- Extract PDF text with OCR fallback for scanned pages.
- Extract image text with RapidOCR and optionally generate image captions with an OpenAI-compatible vision API.
- Transcribe audio locally with Faster-Whisper or through an OpenAI-compatible API.
- Split extracted text into overlapping chunks and store embeddings in ChromaDB.
- Track ingestion progress and failures through the FastAPI backend.
- Use a Gradio interface for uploads, search, status, and file retrieval.

## Architecture

For detailed application flow [click here](docs/findmyfiles-data-flow.md)


The Docker Compose stack contains four services:

- `frontend`: Gradio web interface on port `7860`.
- `backend`: FastAPI API server on port `8000`.
- `worker`: Celery process for asynchronous ingestion.
- `redis`: Celery broker and result backend.

The backend stores application state, uploaded files, and vector data under `backend/data/`. Compose persists this data in the `app-data` volume.

```text
Gradio frontend -> FastAPI backend -> ChromaDB
			    |
			    v
		    Redis <-> Celery worker
```

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose for the containerized setup
- Redis for a non-Docker local setup
- An OpenAI-compatible API key for API-based image captioning or audio transcription

## Configuration

Create a local environment file:

```bash
cp .env.example .env
```

Important variables are documented in [`.env.example`](.env.example):

- `LLM_API_KEY`: API key for image captioning and API audio transcription.
- `LLM_BASE_URL`: OpenAI-compatible API base URL.
- `LLM_VISION_MODEL`: vision model used for image descriptions.
- `LLM_AUDIO_MODEL`: model used for API audio transcription.
- `REDIS_URL`: Redis connection used by Celery.
- `FINDMYFILES_BACKEND_URL`: backend URL used by the frontend.

## Run Locally with Python

Install dependencies:

```bash
uv sync
```

Start Redis, the backend, the worker, and the frontend in *separate terminals* :

Terminal 1:
```bash
redis-server
```
Terminal 2:
```bash
uv run uvicorn backend.app:app --reload
```
Terminal 3:
```bash
uv run celery -A backend.celery_app worker --loglevel=info
```
Terminal 4:
```bash
uv run python -m frontend.app
```

Open the interface at <http://127.0.0.1:7860>. The backend API is available at <http://127.0.0.1:8000>.

## Run with Docker Compose

Docker Compose starts Redis, the backend, the Celery worker, and the frontend:

```bash
docker compose up --build
```

Open the interface at <http://127.0.0.1:7860>. Run in the background with:

```bash
docker compose up --build -d
```

View service logs:

```bash
docker compose logs -f backend worker frontend
```

Stop services without deleting persistent volumes:

```bash
docker compose down
```

For development (backend hot-reload and source mounts), pass both files explicitly:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Use `docker compose down -v` only when you intentionally want to delete application and Redis data.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Application status and ingestion records |
| `POST` | `/upload/` | Upload a supported file and queue ingestion |
| `POST` | `/search/` | Search indexed content with a natural-language query |
| `GET` | `/files/` | List files and ingestion metadata |
| `GET` | `/files/{app_state_id}` | Get the ingestion status by the `app_state_id` returned from `/upload/` |
| `GET` | `/file/{file_id}` | Retrieve the original file by the `file_id` from `/upload/` or `/files/` |

Note: `app_state_id` (ingestion status record) and `file_id` (stored file) are different ID spaces. Status responses include both.
| `GET` | `/formats` | List supported file extensions |

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

## Testing and Code Quality

Run the test suite:

```bash
uv run pytest tests/
```

Run linting and formatting checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

GitHub Actions runs these checks and validates that the Docker image builds. Version tags such as `v0.1.0` publish the image to GitHub Container Registry.

## Evaluation Corpus (Work under progress)

The synthetic retrieval corpus is in [`eval/data/documents/`](eval/data/documents/). It currently contains 24 detailed Markdown documents across six topics:

- Cloud storage
- Cybersecurity
- Project management
- Renewable energy
- Nutrition
- Urban gardening

There are four related but distinct documents per topic. Stable document IDs and topic metadata are defined in [`manifest.json`](eval/data/documents/manifest.json). PDF, image/OCR, and audio fixtures can be added later without changing the existing IDs.

The corpus is intended for comparing extraction, chunking, embedding, and retrieval configurations using metrics such as Recall@k and MRR.


## Project Layout

```text
backend/                 FastAPI routes, ingestion, chunking, and vector storage
frontend/                Gradio user interface
tests/                   Unit and integration tests
eval/data/documents/     Synthetic evaluation corpus
Dockerfile               Production application image
docker-compose.yml        Local multi-service stack
```

## Security Notes

- Keep API keys in environment variables or a secret manager.
- `.env` is ignored by Git and excluded from Docker build context.
- Do not pass runtime credentials as Docker build arguments.
- Rotate a key immediately if it appears in Git history, logs, or a published image.
- Keep Redis private in any public deployment.

## License

See [LICENSE](LICENSE).