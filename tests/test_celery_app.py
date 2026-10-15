"""Tests for Redis URL resolution in backend.celery_app."""

from backend.celery_app import resolve_redis_url


def test_redis_url_read_from_dotenv(tmp_path, monkeypatch):
    """REDIS_URL set only in the project .env is honoured."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("REDIS_URL=redis://remote.example:6399/7\n")

    assert resolve_redis_url(env_file) == "redis://remote.example:6399/7"


def test_real_environment_overrides_dotenv(tmp_path, monkeypatch):
    """An exported REDIS_URL takes precedence over the .env file."""
    monkeypatch.setenv("REDIS_URL", "redis://from-env:6379/1")
    env_file = tmp_path / ".env"
    env_file.write_text("REDIS_URL=redis://remote.example:6399/7\n")

    assert resolve_redis_url(env_file) == "redis://from-env:6379/1"


def test_defaults_to_localhost_without_config(tmp_path, monkeypatch):
    """With neither env var nor .env entry, the local default is used."""
    monkeypatch.delenv("REDIS_URL", raising=False)

    assert resolve_redis_url(tmp_path / "missing.env") == "redis://localhost:6379/0"
