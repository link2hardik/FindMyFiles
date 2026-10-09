import json
from pathlib import Path

from backend.chunker.recursive_chunker import RecursiveChunker
from backend.ingestors.audio_ingestor import AudioIngestor
from backend.ingestors.image_ingestor import ImageOCRIngestor
from backend.ingestors.pdf_ingestor import PdfIngestor
from backend.ingestors.text_ingestor import TextIngestor
from backend.vector_store.local_vec_store import ChromaDBVectorStore
from eval.scripts.eval_app import EvalApp


CONFIG_PATH = Path(__file__).with_name("configs.json")


def generate_apps(config_path: Path = CONFIG_PATH):
    with config_path.open() as config_file:
        configurations = json.load(config_file)

    for configuration in configurations:
        accepted_formats = configuration["accepted_formats"]
        ingestors = [
            AudioIngestor(
                accepted_formats=[
                    extension
                    for extension, kind in accepted_formats.items()
                    if kind == "audio"
                ]
            ),
            ImageOCRIngestor(
                accepted_formats=[
                    extension
                    for extension, kind in accepted_formats.items()
                    if kind == "image"
                ]
            ),
            TextIngestor(
                accepted_formats=[
                    extension
                    for extension, kind in accepted_formats.items()
                    if kind == "text"
                ]
            ),
            PdfIngestor(
                accepted_formats=[
                    extension
                    for extension, kind in accepted_formats.items()
                    if kind == "pdf"
                ]
            ),
        ]

        work_dir = configuration["work_dir"]
        yield EvalApp(
            name=configuration["name"],
            work_dir=work_dir,
            description=configuration.get("description", ""),
            ingestors=ingestors,
            chunker=RecursiveChunker(
                chunk_size=configuration["chunk_size"],
                overlap=configuration["overlap"],
            ),
            vector_store=ChromaDBVectorStore(work_dir),
        )


app_configs = generate_apps()