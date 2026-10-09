import pytest

import backend.ingestors.audio_ingestor as m
from backend.filestore.base_filestore import IngestableFile as _IngestableFile
from backend.ingestors.audio_ingestor import AudioIngestor
from backend.ingestors.base_ingestor import BaseIngestor, IngestionError, Metadata


def IngestableFile(file_obj, name=None, file_id=1):
    return _IngestableFile(file_obj, name=name, file_id=file_id)


"""Comprehensive unit tests for AudioIngestor.

The heavy dependencies (faster-whisper model, OpenAI/Groq API) are faked at
the module boundary, so both transcription paths (local Whisper and API) are
exercised deterministically: successful transcription, metadata
construction, extension validation, error wrapping/chaining, and the
class-level format registration registry. The isolated_registry fixture
snapshots/restores the global registry so tests never leak state.
"""


class FakeSegment:
    """Stand-in for a faster-whisper segment."""

    def __init__(self, text):
        self.text = text


class FakeWhisperModel:
    """Stand-in for faster_whisper.WhisperModel that records what it was
    asked to transcribe."""

    def __init__(self, *args, **kwargs):
        self.transcribed_with = None

    def transcribe(self, file_obj):
        self.transcribed_with = file_obj
        return ([FakeSegment("hello"), FakeSegment("world")], None)


class FailingWhisperModel:
    """WhisperModel whose construction fails (e.g. download error)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("model download failed")


class FailingTranscribeModel:
    """WhisperModel whose transcribe() always fails."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, file_obj):
        raise RuntimeError("transcription failed")


class RaisingIngestionWhisper:
    """WhisperModel whose transcribe() raises IngestionError directly."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, file_obj):
        raise IngestionError("whisper boom")


class FakeTranscriptions:
    """Stand-in for client.audio.transcriptions that records calls."""

    def __init__(self, text="transcribed api text", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        if self.error:
            raise self.error
        self.calls.append(kwargs)
        return self.text


class FakeAudioNamespace:
    """Stand-in for the OpenAI client's `.audio` namespace."""

    def __init__(self, text="transcribed api text", error=None):
        self.transcriptions = FakeTranscriptions(text, error)


class FakeAudioClient:
    """Stand-in for an OpenAI client exposing a fake transcriptions API."""

    def __init__(self, text="transcribed api text", error=None):
        self.audio = FakeAudioNamespace(text, error)


class FakePdfIngestor(BaseIngestor):
    """A BaseIngestor subclass used to simulate cross-class format conflicts."""

    def extract_text(self, file):
        return "", self.extract_metadata(file)


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


def make_audio_ingestor(accepted_formats, use_api=False):
    """Build an AudioIngestor registered for the given formats."""
    return AudioIngestor(accepted_formats=accepted_formats, use_api=use_api)


def make_file(tmp_path, name, data=b"audio"):
    """Write `data` to a temp file and return an open read handle."""
    path = tmp_path / name
    path.write_bytes(data)
    return open(path, "rb")


class TestExtractTextLocal:
    def test_local_transcription_joins_segments_and_metadata(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """Local mode joins whisper segments with spaces and returns valid
        metadata."""
        monkeypatch.setattr(m, "WhisperModel", FakeWhisperModel)

        with make_file(tmp_path, "clip.wav") as f:
            ingestable = IngestableFile(f, "clip.wav")
            ingestor = make_audio_ingestor(["wav"])
            text, metadata = ingestor.extract_text(ingestable)

        assert text == "hello world"
        assert type(text) is str
        assert type(metadata) is Metadata
        assert metadata.file_name == "clip.wav"
        assert metadata.extension == "wav"
        assert metadata.type == "audio"
        assert isinstance(metadata.created_at_ts, float)

    def test_local_transcribe_failure_wraps_as_ingestion_error(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """A whisper transcribe failure surfaces as a chained IngestionError,
        not the raw RuntimeError."""
        monkeypatch.setattr(m, "WhisperModel", FailingTranscribeModel)

        with make_file(tmp_path, "clip.wav") as f:
            ingestor = make_audio_ingestor(["wav"])
            with pytest.raises(IngestionError, match="Failed to extract text") as exc:
                ingestor.extract_text(IngestableFile(f, "clip.wav"))

        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_local_whisper_initialization_failure_wraps(
        self, isolated_registry, monkeypatch
    ):
        """A WhisperModel construction failure surfaces as a chained
        IngestionError mentioning model initialization."""
        monkeypatch.setattr(m, "WhisperModel", FailingWhisperModel)
        ingestor = make_audio_ingestor(["wav"])

        with pytest.raises(
            IngestionError, match="Failed to initialize WhisperModel"
        ) as exc:
            _ = ingestor.whisper

        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_local_whisper_ingestion_error_not_double_wrapped(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """An IngestionError from the whisper layer propagates unchanged, not
        re-wrapped with a generic 'Failed to extract text' prefix."""
        monkeypatch.setattr(m, "WhisperModel", RaisingIngestionWhisper)

        with make_file(tmp_path, "clip.wav") as f:
            ingestor = make_audio_ingestor(["wav"])
            with pytest.raises(IngestionError, match="whisper boom") as exc:
                ingestor.extract_text(IngestableFile(f, "clip.wav"))

        assert "Failed to extract text" not in str(exc.value)


class TestExtractTextApi:
    def test_api_transcription_returns_text_and_metadata(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """API mode returns the transcribed text, forwards the right model/
        format/file arguments, and produces valid metadata."""
        monkeypatch.setattr(m, "LLM_API_KEY", "gsk-test")
        monkeypatch.setattr(m, "OpenAI", lambda *a, **k: FakeAudioClient())

        with make_file(tmp_path, "clip.mp3") as f:
            ingestable = IngestableFile(f, "clip.mp3")
            ingestor = AudioIngestor(accepted_formats=["mp3"], use_api=True)
            text, metadata = ingestor.extract_text(ingestable)

        assert text == "transcribed api text"
        (call,) = ingestor.client.audio.transcriptions.calls
        assert call["model"] == m.LLM_AUDIO_MODEL
        assert call["response_format"] == "text"
        assert call["file"] == ("clip.mp3", ingestable.file_obj)
        assert metadata.extension == "mp3"
        assert metadata.type == "audio"

    def test_api_failure_wraps_as_ingestion_error(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """An API transcription failure surfaces as a chained IngestionError."""
        monkeypatch.setattr(m, "LLM_API_KEY", "gsk-test")
        monkeypatch.setattr(m, "OpenAI", lambda *a, **k: FakeAudioClient())

        with make_file(tmp_path, "clip.mp3") as f:
            ingestor = AudioIngestor(accepted_formats=["mp3"], use_api=True)
            ingestor.client = FakeAudioClient(error=RuntimeError("api down"))

            with pytest.raises(IngestionError, match="Failed to extract text") as exc:
                ingestor.extract_text(IngestableFile(f, "clip.mp3"))

        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_missing_api_key_raises_ingestion_error_when_using_api(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """Constructing with use_api=True and no LLM_API_KEY is allowed,
        but extract_text raises IngestionError when trying to use the API."""
        monkeypatch.setattr(m, "LLM_API_KEY", None)

        ingestor = AudioIngestor(accepted_formats=["mp3"], use_api=True)
        assert ingestor.client is None

        with make_file(tmp_path, "clip.mp3") as f:
            with pytest.raises(
                IngestionError,
                match="LLM_API_KEY or GROQ_KEY environment variable not set",
            ):
                ingestor.extract_text(IngestableFile(f, "clip.mp3"))

    def test_local_mode_without_api_key_works(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """use_api=False constructs cleanly without an API key and transcribes
        locally."""
        monkeypatch.setattr(m, "LLM_API_KEY", None)
        monkeypatch.setattr(m, "WhisperModel", FakeWhisperModel)

        with make_file(tmp_path, "clip.wav") as f:
            ingestor = AudioIngestor(accepted_formats=["wav"], use_api=False)
            text, metadata = ingestor.extract_text(IngestableFile(f, "clip.wav"))

        assert ingestor.client is None
        assert text == "hello world"


class TestExtensionValidation:
    def test_extension_mismatch_raises_ingestion_error(
        self, tmp_path, isolated_registry
    ):
        """An unsupported extension raises IngestionError mentioning the
        offending extension, before any transcription happens."""
        with make_file(tmp_path, "clip.wav") as f:
            ingestor = make_audio_ingestor(["mp3"])
            with pytest.raises(IngestionError, match="does not match any type") as exc:
                ingestor.extract_text(IngestableFile(f, "clip.wav"))

        assert "wav" in str(exc.value)

    def test_accepts_any_accepted_format(
        self, tmp_path, isolated_registry, monkeypatch
    ):
        """Both configured formats transcribe successfully."""
        monkeypatch.setattr(m, "WhisperModel", FakeWhisperModel)

        with make_file(tmp_path, "clip.wav") as f:
            ingestor = make_audio_ingestor(["wav", "mp3"])
            text, metadata = ingestor.extract_text(IngestableFile(f, "clip.wav"))

        assert text == "hello world"
        assert metadata.extension == "wav"


class TestInit:
    def test_constructor_wraps_unexpected_errors(self, isolated_registry):
        """An unexpected failure inside super().__init__ (here: an unhashable
        format) is wrapped in a chained IngestionError."""
        with pytest.raises(
            IngestionError, match="Failed to update ingestor map"
        ) as exc:
            AudioIngestor(accepted_formats=[["unhashable"]])

        assert isinstance(exc.value.__cause__, TypeError)

    def test_default_type_is_audio(self, isolated_registry):
        """The default ingestor type is 'audio', so metadata is correct even
        when no type is passed explicitly."""
        ingestor = AudioIngestor(accepted_formats=["wav"], use_api=False)

        assert ingestor.type == "audio"


class TestRegistration:
    def test_registration_populates_registry(self, isolated_registry):
        """Constructing an ingestor registers its formats in the global map,
        pointing at the instance."""
        ingestor = make_audio_ingestor(["audio-ext"])

        assert "audio-ext" in BaseIngestor.all_formats
        assert BaseIngestor.ingestor_map["audio-ext"] is ingestor

    def test_same_class_re_registration_is_noop(self, isolated_registry):
        """Instantiating the same class with an already-registered format does
        not raise; the registry points at the latest instance."""
        first = make_audio_ingestor(["wav"])
        second = make_audio_ingestor(["wav"])

        assert BaseIngestor.ingestor_map["wav"] is second
        assert BaseIngestor.ingestor_map["wav"] is not first

    def test_cross_class_conflict_raises_ingestion_error(self, isolated_registry):
        """A different class claiming an owned format raises IngestionError
        naming the format and the owning ingestor."""
        FakePdfIngestor(accepted_formats=["audio-conflict"])

        with pytest.raises(IngestionError, match="already registered by") as exc:
            make_audio_ingestor(["audio-conflict"])

        assert "audio-conflict" in str(exc.value)
