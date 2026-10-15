import os
from pathlib import Path

from celery import Celery
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def resolve_redis_url(env_path: Path = PROJECT_ROOT / ".env") -> str:
    load_dotenv(dotenv_path=env_path, override=False)
    return os.getenv("REDIS_URL", DEFAULT_REDIS_URL)


REDIS_URL = resolve_redis_url()

celery_app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL)

# Automatically look for tasks inside a tasks.py file
celery_app.autodiscover_tasks(["backend.tasks"])

# Optional configuration tweaks
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,  # Expire task results in 1 hour
)
