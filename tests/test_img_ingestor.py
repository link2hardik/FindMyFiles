import base64
import io
import time

import httpx
import openai
import pytest

import backend.ingestors.image_ingestor as m
from backend.filestore.base_filestore import IngestableFile as _IngestableFile
from backend.ingestors.base_ingestor import BaseIngestor, IngestionError, Metadata
from backend.ingestors.image_ingestor import ImageOCRIngestor


def IngestableFile(file_obj, name=None, file_id=1):
    return _IngestableFile(file_obj, name=name, file_id=file_id)


"""Comprehensive unit tests for ImageOCRIngestor.

RapidOCR and the OpenAI/Groq chat client are faked at the module boundary so
every path is exercised deterministically: OCR extraction (success, empty,
engine failures, passthrough), API captioning (success, rate-limit retries,
exhaustion, generic failures, missing key), the combined extract_text flow,
metadata construction, error wrapping/chaining, and the format-registry
registration. The isolated_registry fixture snapshots/restores the global
registry so tests never leak state.
"""


class FakeOcrEngine:
    """Callable stand-in for the RapidOCR engine."""

    def __init__(self, result=None, error=None):
        self.result = (
            result if result is not None else [("box", "hello"), ("box", "world")]
        )
        self.error = error
        self.last_bytes = None

    def __call__(self, image_bytes):
        self.last_bytes = image_bytes
        if self.error is not None:
            raise self.error
        return self.result, None


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content="a caption"):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, client):
        self.client = client

    def create(self, **kwargs):
        return self.client.create(**kwargs)


class FakeChat:
    def __init__(self, client):
        self.completions = FakeCompletions(client)


class FakeOpenAIClient:
    """Stand-in for the OpenAI/Groq chat client with scripted failures.

    `rate_limit_times` makes the first N calls raise a real
    openai.RateLimitError (429); `error` makes every call raise a generic
    exception. Successful calls return a caption and record their kwargs.
    """

    def __init__(self, content="a caption", rate_limit_times=0, error=None, **kwargs):
        self.content = content
        self.rate_limit_times = rate_limit_times
        self.error = error
        self.calls = []
        self.chat = FakeChat(self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.rate_limit_times > 0:
            self.rate_limit_times -= 1
            request = httpx.Request(
                "POST", "https://api.groq.com/openai/v1/chat/completions"
            )
            response = httpx.Response(429, request=request)
            raise openai.RateLimitError(
                "429 rate limited", response=response, body=None
            )
        return FakeResponse(self.content)


class FakeOtherIngestor(BaseIngestor):
    """A different ingestor class, used to provoke registry conflicts."""

    def __init__(self, accepted_formats=None):
        super().__init__("other", accepted_formats, "other")

    def extract_text(self, file):
        raise NotImplementedError


class RecordingBytesIO(io.BytesIO):
    """BytesIO that records the position read() was called at."""

    def __init__(self, data):
        super().__init__(data)
        self.read_position = None

    def read(self, size=-1):
        self.read_position = self.tell()
        return super().read(size)


class FailingReadBytesIO(io.BytesIO):
    """BytesIO whose read() always fails."""

    def read(self, size=-1):
        raise OSError("disk read failed")


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Isolate tests from an API key present in the real environment."""
    monkeypatch.setattr(m, "LLM_API_KEY", None)


@pytest.fixture
def isolated_registry():
    """Snapshot the global ingestor registry and restore it after the test,
    so format registrations never leak across tests."""
    saved_formats = set(BaseIngestor.all_formats)
    saved_map = dict(BaseIngestor.ingestor_map)
    yield
    BaseIngestor.all_formats.clear()
    BaseIngestor.all_formats.update(saved_formats)
    BaseIngestor.ingestor_map.clear()
    BaseIngestor.ingestor_map.update(saved_map)


@pytest.fixture
def no_sleep(monkeypatch):
    """Stub time.sleep so rate-limit retries don't block the test."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


def make_ingestor(accepted_formats, use_api=True):
    """Build an ImageOCRIngestor registered for the given formats."""
    return ImageOCRIngestor(accepted_formats=accepted_formats, use_api=use_api)


def make_file(tmp_path, name, data=b"image"):
    """Write `data` to a temp file and return an open read handle."""
    path = tmp_path / name
    path.write_bytes(data)
    return open(path, "rb")


def make_client_with_key(monkeypatch):
    """Point LLM_API_KEY and OpenAI at the fake client and return the ingestor."""
    monkeypatch.setattr(m, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(m, "OpenAI", FakeOpenAIClient)
    return make_ingestor(["png"])


class TestEncodeImage:
    def test_encodes_bytes_as_base64(self):
        """encode_image returns the base64 text of the file contents."""
        assert (
            m.encode_image(io.BytesIO(b"hello")) == base64.b64encode(b"hello").decode()
        )


class TestInit:
    def test_defaults(self, isolated_registry):
        """Defaults: type 'img', name 'OCR', use_api True, no client."""
        ingestor = make_ingestor(["png"])
        assert ingestor.type == "img"
        assert ingestor.name == "OCR"
        assert ingestor.use_api is True
        assert ingestor.client is None
        assert ingestor.accepted_formats == ["png"]

    def test_registers_formats_in_global_registry(self, isolated_registry):
        """Constructor registers every format in all_formats and ingestor_map."""
        ingestor = make_ingestor(["jpg", "png"])
        for format in ("jpg", "png"):
            assert format in BaseIngestor.all_formats
            assert BaseIngestor.ingestor_map[format] is ingestor

    def test_same_class_re_registration_is_idempotent(self, isolated_registry):
        """A second instance of the same class may claim the same formats."""
        first = make_ingestor(["png"])
        second = make_ingestor(["png"])
        assert BaseIngestor.ingestor_map["png"] is second
        assert first is not second

    def test_cross_class_conflict_raises_ingestion_error(self, isolated_registry):
        """A different class claiming an owned format raises IngestionError
        naming the format and the owning ingestor."""
        FakeOtherIngestor(accepted_formats=["png-conflict"])
        with pytest.raises(IngestionError, match="already registered by") as exc:
            make_ingestor(["png-conflict"])
        assert "png-conflict" in str(exc.value)

    def test_creates_openai_client_when_key_present(
        self, isolated_registry, monkeypatch
    ):
        """With LLM_API_KEY set, an OpenAI client is built with the configured base URL."""
        ingestor = make_client_with_key(monkeypatch)
        assert isinstance(ingestor.client, FakeOpenAIClient)

    def test_no_client_without_key(self, isolated_registry):
        """Without LLM_API_KEY, client stays None even with use_api=True."""
        ingestor = make_ingestor(["png"], use_api=True)
        assert ingestor.client is None

    def test_unexpected_init_error_wraps_as_ingestion_error(
        self, isolated_registry, monkeypatch
    ):
        """A non-IngestionError during init surfaces as a chained
        IngestionError, not the raw exception."""

        def boom(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(m.BaseIngestor, "__init__", boom)
        with pytest.raises(
            IngestionError, match="Failed to initialize ImageOCRIngestor"
        ) as exc:
            make_ingestor(["png"])
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_ingestion_error_from_init_propagates_unchanged(self, isolated_registry):
        """A registration conflict during init propagates unwrapped."""
        FakeOtherIngestor(accepted_formats=["conflict-png"])
        with pytest.raises(IngestionError, match="already registered") as exc:
            make_ingestor(["conflict-png"])
        assert exc.value.__cause__ is None


class TestOcrExtract:
    def test_joins_ocr_text_with_spaces(self, tmp_path, isolated_registry, monkeypatch):
        """OCR result boxes are joined with spaces and the engine receives
        the full file bytes."""
        engine = FakeOcrEngine()
        monkeypatch.setattr(m, "get_ocr_engine", lambda: engine)

        with make_file(tmp_path, "pic.png", data=b"png-bytes") as f:
            ingestor = make_ingestor(["png"], use_api=False)
            text = ingestor.ocr_extract(IngestableFile(f, "pic.png"))

        assert text == "hello world"
        assert engine.last_bytes == b"png-bytes"

    def test_empty_ocr_result_returns_empty_string(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """No recognized text yields an empty string (not an error)."""
        monkeypatch.setattr(m, "get_ocr_engine", lambda: FakeOcrEngine(result=[]))

        with make_file(tmp_path, "pic.png") as f:
            ingestor = make_ingestor(["png"], use_api=False)
            assert ingestor.ocr_extract(IngestableFile(f, "pic.png")) == ""

    def test_resets_file_pointer_between_calls(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """seek(0) before reading lets the same file be OCR'd repeatedly."""
        engine = FakeOcrEngine()
        monkeypatch.setattr(m, "get_ocr_engine", lambda: engine)

        with make_file(tmp_path, "pic.png") as f:
            ingestor = make_ingestor(["png"], use_api=False)
            ingestable = IngestableFile(f, "pic.png")
            assert ingestor.ocr_extract(ingestable) == "hello world"
            assert ingestor.ocr_extract(ingestable) == "hello world"
        assert engine.last_bytes == b"image"

    def test_engine_runtime_error_wraps_as_chained_ingestion_error(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """An engine crash surfaces as a chained IngestionError."""
        monkeypatch.setattr(
            m,
            "get_ocr_engine",
            lambda: FakeOcrEngine(error=RuntimeError("engine crash")),
        )

        with make_file(tmp_path, "pic.png") as f:
            ingestor = make_ingestor(["png"], use_api=False)
            with pytest.raises(
                IngestionError, match="Failed to extract text via OCR"
            ) as exc:
                ingestor.ocr_extract(IngestableFile(f, "pic.png"))
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_engine_ingestion_error_propagates_unchanged(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """An IngestionError from the engine is not re-wrapped."""
        monkeypatch.setattr(
            m, "get_ocr_engine", lambda: FakeOcrEngine(error=IngestionError("ocr boom"))
        )

        with make_file(tmp_path, "pic.png") as f:
            ingestor = make_ingestor(["png"], use_api=False)
            with pytest.raises(IngestionError, match="ocr boom") as exc:
                ingestor.ocr_extract(IngestableFile(f, "pic.png"))
        assert exc.value.__cause__ is None

    def test_file_read_failure_wraps_as_ingestion_error(
        self, isolated_registry, monkeypatch
    ):
        """A read failure surfaces as a chained IngestionError, not OSError."""
        monkeypatch.setattr(m, "get_ocr_engine", lambda: FakeOcrEngine())
        ingestor = make_ingestor(["png"], use_api=False)
        ingestable = IngestableFile(FailingReadBytesIO(b"x"), "pic.png")

        with pytest.raises(
            IngestionError, match="Failed to extract text via OCR"
        ) as exc:
            ingestor.ocr_extract(ingestable)
        assert isinstance(exc.value.__cause__, OSError)


class TestApiCaption:
    def test_missing_key_raises_ingestion_error(self, tmp_path, isolated_registry):
        """No client (no LLM_API_KEY) raises IngestionError, not ValueError."""
        ingestor = make_ingestor(["png"], use_api=True)
        with make_file(tmp_path, "pic.png") as f:
            with pytest.raises(IngestionError, match="API key not provided."):
                ingestor.api_caption(IngestableFile(f, "pic.png"))

    def test_zero_timeout_raises_without_calling_api(self, tmp_path, isolated_registry):
        """timeout=0 fails fast without any API call."""
        client = FakeOpenAIClient()
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = client
        with make_file(tmp_path, "pic.png") as f:
            with pytest.raises(IngestionError, match="Request limit reached"):
                ingestor.api_caption(IngestableFile(f, "pic.png"), timeout=0)
        assert client.calls == []

    def test_success_returns_caption_and_sends_expected_request(
        self, tmp_path, isolated_registry
    ):
        """A successful caption is returned with the right model, tokens,
        temperature, and a data-URI image payload."""
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = FakeOpenAIClient(content="a sunny beach")

        with make_file(tmp_path, "pic.png", data=b"img-data") as f:
            caption = ingestor.api_caption(IngestableFile(f, "pic.png"))

        assert caption == "a sunny beach"
        (call,) = ingestor.client.calls
        assert call["model"] == m.LLM_VISION_MODEL
        assert call["max_tokens"] == 300
        assert call["temperature"] == 0.0
        image_url = call["messages"][0]["content"][1]["image_url"]["url"]
        assert image_url == (
            "data:image/jpeg;base64," + base64.b64encode(b"img-data").decode()
        )

    def test_seeks_file_before_encoding(self, isolated_registry):
        """The file pointer is reset to 0 before the image is read/encoded."""
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = FakeOpenAIClient()

        ingestable = IngestableFile(RecordingBytesIO(b"img-data"), "pic.png")
        ingestor.api_caption(ingestable)
        assert ingestable.file_obj.read_position == 0

    def test_rate_limit_retries_then_succeeds(
        self, tmp_path, isolated_registry, no_sleep, monkeypatch
    ):
        """A 429 is retried (with a sleep) and a later attempt succeeds."""
        sleeps = []
        monkeypatch.setattr(time, "sleep", sleeps.append)
        client = FakeOpenAIClient(rate_limit_times=1)
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = client

        with make_file(tmp_path, "pic.png") as f:
            caption = ingestor.api_caption(IngestableFile(f, "pic.png"))

        assert caption == "a caption"
        assert len(client.calls) == 2
        assert sleeps == [5]

    def test_persistent_rate_limit_exhausts_attempts(
        self, tmp_path, isolated_registry, no_sleep
    ):
        """Persistent 429s stop after REQUEST_LIMIT attempts with a clean
        IngestionError."""
        client = FakeOpenAIClient(rate_limit_times=100)
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = client

        with make_file(tmp_path, "pic.png") as f:
            with pytest.raises(IngestionError, match="Request limit reached"):
                ingestor.api_caption(IngestableFile(f, "pic.png"))

        assert len(client.calls) == m.REQUEST_LIMIT

    def test_custom_timeout_limits_attempts(
        self, tmp_path, isolated_registry, no_sleep
    ):
        """A caller-supplied timeout caps the retry budget."""
        client = FakeOpenAIClient(rate_limit_times=100)
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = client

        with make_file(tmp_path, "pic.png") as f:
            with pytest.raises(IngestionError, match="Request limit reached"):
                ingestor.api_caption(IngestableFile(f, "pic.png"), timeout=2)

        assert len(client.calls) == 2

    def test_generic_api_error_wraps_as_chained_ingestion_error(
        self, tmp_path, isolated_registry, no_sleep
    ):
        """A non-rate-limit API failure surfaces as a chained IngestionError."""
        client = FakeOpenAIClient(error=RuntimeError("api down"))
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = client

        with make_file(tmp_path, "pic.png") as f:
            with pytest.raises(
                IngestionError, match="Failed to fetch image captions via API"
            ) as exc:
                ingestor.api_caption(IngestableFile(f, "pic.png"))
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert len(client.calls) == 1

    def test_encode_failure_wraps_as_ingestion_error(self, isolated_registry):
        """A file read failure during encoding is wrapped, not raw."""
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = FakeOpenAIClient()
        ingestable = IngestableFile(FailingReadBytesIO(b"x"), "pic.png")

        with pytest.raises(
            IngestionError, match="Failed to fetch image captions via API"
        ) as exc:
            ingestor.api_caption(ingestable)
        assert isinstance(exc.value.__cause__, OSError)


class TestExtractText:
    def test_ocr_only_mode_returns_text_and_metadata(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """use_api=False returns OCR text with valid Metadata."""
        monkeypatch.setattr(m, "get_ocr_engine", lambda: FakeOcrEngine())

        with make_file(tmp_path, "pic.png") as f:
            ingestor = make_ingestor(["png"], use_api=False)
            text, metadata = ingestor.extract_text(IngestableFile(f, "pic.png"))

        assert text == "hello world"
        assert type(text) is str
        assert type(metadata) is Metadata
        assert metadata.file_name == "pic.png"
        assert metadata.extension == "png"
        assert metadata.type == "img"
        assert isinstance(metadata.created_at_ts, float)

    def test_api_mode_appends_caption(self, tmp_path, isolated_registry, monkeypatch):
        """use_api=True appends the caption to the OCR text."""
        monkeypatch.setattr(m, "get_ocr_engine", lambda: FakeOcrEngine())
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = FakeOpenAIClient(content="a sunny beach")

        with make_file(tmp_path, "pic.png") as f:
            text, _ = ingestor.extract_text(IngestableFile(f, "pic.png"))

        assert text == "hello world, Image Description: a sunny beach"

    def test_empty_ocr_without_api_raises(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """No OCR text and no API means extraction fails."""
        monkeypatch.setattr(m, "get_ocr_engine", lambda: FakeOcrEngine(result=[]))

        with make_file(tmp_path, "pic.png") as f:
            ingestor = make_ingestor(["png"], use_api=False)
            with pytest.raises(IngestionError, match="didn't produce text via OCR"):
                ingestor.extract_text(IngestableFile(f, "pic.png"))

    def test_empty_ocr_with_api_caption_returns_caption(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """A scanned/blank image with no OCR text still works via caption."""
        monkeypatch.setattr(m, "get_ocr_engine", lambda: FakeOcrEngine(result=[]))
        ingestor = make_ingestor(["png"], use_api=True)
        ingestor.client = FakeOpenAIClient(content="a blank page")

        with make_file(tmp_path, "pic.png") as f:
            text, metadata = ingestor.extract_text(IngestableFile(f, "pic.png"))

        assert text == ", Image Description: a blank page"
        assert metadata.type == "img"

    def test_caption_ingestion_error_propagates_unchanged(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """A missing-key failure inside api_caption is not re-wrapped."""
        monkeypatch.setattr(m, "get_ocr_engine", lambda: FakeOcrEngine())
        ingestor = make_ingestor(["png"], use_api=True)

        with make_file(tmp_path, "pic.png") as f:
            with pytest.raises(IngestionError, match="API key not provided.") as exc:
                ingestor.extract_text(IngestableFile(f, "pic.png"))
        assert exc.value.__cause__ is None

    def test_ocr_engine_failure_wraps_with_ingestion_context(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """extract_text surfaces engine failures as a chained IngestionError."""
        monkeypatch.setattr(
            m,
            "get_ocr_engine",
            lambda: FakeOcrEngine(error=RuntimeError("engine crash")),
        )
        ingestor = make_ingestor(["png"], use_api=False)

        with make_file(tmp_path, "pic.png") as f:
            with pytest.raises(
                IngestionError, match="Failed to extract text via OCR"
            ) as exc:
                ingestor.extract_text(IngestableFile(f, "pic.png"))
        assert isinstance(exc.value.__cause__, RuntimeError)
