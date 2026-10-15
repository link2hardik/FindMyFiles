import base64
import logging
import os
import time
from functools import cached_property

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from backend.ingestors.base_ingestor import IngestableFile, IngestionError, BaseIngestor
from backend.ingestors.ocr_utils import get_ocr_engine

load_dotenv()

LLM_API_KEY = (
    os.getenv("LLM_API_KEY", "").strip() or os.getenv("GROQ_KEY", "").strip() or None
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_VISION_MODEL = os.getenv("LLM_VISION_MODEL", "qwen/qwen3.6-27b")
REQUEST_LIMIT = 5

logger = logging.getLogger(__name__)

_MIME_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def detect_mime_type(image_bytes: bytes) -> str:
    """Infer the image MIME type from its magic bytes (JPEG if unknown)."""
    for signature, mime_type in _MIME_SIGNATURES:
        if image_bytes.startswith(signature):
            return mime_type
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")


class ImageOCRIngestor(BaseIngestor):
    @cached_property
    def engine(self):
        return get_ocr_engine()

    def __init__(self, type="img", accepted_formats=None, name="OCR", use_api=True):
        try:
            super().__init__(type, accepted_formats, name)
            self.client = None
            if LLM_API_KEY:
                self.client = OpenAI(
                    api_key=LLM_API_KEY,
                    base_url=LLM_BASE_URL,
                )
            self.use_api = use_api
        except IngestionError as e:
            raise e
        except Exception as e:
            raise IngestionError(
                f"Failed to initialize ImageOCRIngestor: {str(e)}"
            ) from e

    def extract_text(self, file):
        try:
            parts = []
            ocr_text = self.ocr_extract(file).strip()
            if ocr_text:
                parts.append(ocr_text)

            caption = self._try_caption(file)
            if caption:
                parts.append(f"Image Description: {caption}")

            text = ", ".join(parts)
            if not text:
                raise IngestionError(
                    f"Image {file.file_name} produced no text via OCR or API captioning."
                )
            return text, self.extract_metadata(file)
        except IngestionError as e:
            raise e
        except Exception as e:
            raise IngestionError(f"Failed to extract text via OCR: {str(e)}") from e

    def _try_caption(self, file: IngestableFile) -> str | None:
        """Best-effort caption: returns None instead of failing the ingestion.

        Captions are an enrichment on top of OCR, so a missing API key or a
        failing API call must not discard text that OCR already extracted.
        """
        if not self.use_api or self.client is None:
            return None
        try:
            return (self.api_caption(file) or "").strip() or None
        except IngestionError as e:
            logger.warning("Skipping caption for %s: %s", file.file_name, e)
            return None

    def ocr_extract(self, file):
        try:
            file.file_obj.seek(0)
            file_bytes = file.file_obj.read()
            result, _ = self.engine(file_bytes)
            if result:
                text = " ".join([item[1] for item in result])
            else:
                text = ""
            return text
        except IngestionError as e:
            raise e
        except Exception as e:
            raise IngestionError(f"Failed to extract text via OCR: {str(e)}") from e

    def api_caption(self, file: IngestableFile, timeout: int | None = None):
        """Calls a vision model to generate image captions."""
        if not self.client:
            raise IngestionError("API key not provided.")
        if timeout is not None and timeout <= 0:
            raise IngestionError(
                "Request limit reached. Failed to fetch image captions via API."
            )

        if timeout is None:
            timeout = REQUEST_LIMIT  # set timeout value for each request

        timeout -= 1

        try:
            file.file_obj.seek(0)
            image_bytes = file.file_obj.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = detect_mime_type(image_bytes)

            response = self.client.chat.completions.create(
                model=LLM_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                    Describe this image in detail as if creating a searchable caption for it. Write 3-5 natural, flowing sentences that cover:

                                    - What the image shows overall (scene, setting, location type, indoor/outdoor, time of day, weather if visible)
                                    - Any people, animals, or objects present, including notable colors, brands, or distinguishing details
                                    - Any actions, activities, or events taking place
                                    - Any visible text, numbers, signs, or labels — transcribe them exactly as written
                                    - The overall mood, style, or context (e.g. candid photo, screenshot, document scan, receipt, artwork)

                                    Be specific and use concrete nouns and descriptive adjectives rather than vague language, since this description will be used for semantic search — someone should be able to find this image later by describing what's in it.

                                    Do not include any preamble, headers, labels, or formatting. Do not use JSON or markdown. Do not include phrases like "the image shows" or "this is a picture of" — start directly with the description itself.

                                    If the image is a document, receipt, or screenshot rather than a photo, prioritize transcribing all visible text accurately over describing visual style.
                            """,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=300,
                extra_body={
                    "reasoning_effort": "none",
                    "reasoning_format": "hidden",
                },
            )
            text = response.choices[0].message.content
            return text

        except RateLimitError:
            # Handles 429 errors gracefully if you scan multiple images too fast

            print("rate limit reached. Waiting 5 seconds before retry...")
            time.sleep(5)
            return self.api_caption(file, timeout=timeout)

        except Exception as e:
            raise IngestionError(
                f"Failed to fetch image captions via API: {str(e)}"
            ) from e
