from backend.chunker import BaseChunker
from backend.filestore import BaseFileStore, IngestableFile
from backend.filestore.local_filestore import LocalSQLiteFileStore
from backend.app_state import AppState
from backend.ingestors import BaseIngestor, IngestionError
from backend.vector_store import BaseVectorStore
from backend.vector_store.local_vec_store import ChromaDBVectorStore
import shutil


class EvalApp:
    names = set()

    def __init__(
        self,
        name: str,
        work_dir: str,
        description: str = "",
        ingestors: list[BaseIngestor] | None = None,
        chunker: BaseChunker | None = None,
        filestore: BaseFileStore | None = None,
        vector_store: BaseVectorStore | None = None,
    ):

        if not ingestors or chunker is None:
            raise ValueError("Ingestors and chunker must be defined.")

        if name in EvalApp.names:
            raise ValueError()
        EvalApp.names.add(name)
        self.name = name
        self.desc = description
        self.file_map = {}
        self.ingestors = ingestors
        self.chunker = chunker
        self.work_dir = work_dir
        self.app_state = AppState(db_path=self.work_dir)
        self.filestore = (
            filestore
            if filestore is not None
            else LocalSQLiteFileStore(path=self.work_dir)
        )
        self.vector_store = (
            vector_store
            if vector_store is not None
            else ChromaDBVectorStore(path=self.work_dir)
        )

    def add_file(self, file: IngestableFile, manifest_id):
        ingestable_file = file
        if ingestable_file.extension not in BaseIngestor.all_formats:
            return

        file_id = self.filestore.store(ingestable_file)
        self.file_map[file_id] = manifest_id

        app_state_id = self.app_state.insert_file(
            file_name=ingestable_file.file_name,
            file_type=ingestable_file.extension,
            file_status="Storage Complete",
            file_id=file_id,
        )

        file_obj = self.filestore.get(file_id)
        ingestor = BaseIngestor.ingestor_map.get(file_obj.extension)
        if ingestor is None:
            raise IngestionError(f"No ingestor found for {file_obj.extension}")

        text, metadata = ingestor.extract_text(file_obj)
        self.app_state.update_file(app_state_id, "Ingestion Complete")

        chunks = self.chunker.split_text(text)
        self.app_state.update_file(app_state_id, "Chunking Complete")

        self.vector_store.add(chunks, [metadata] * len(chunks))
        self.app_state.update_file(app_state_id, "Embedding Complete")
        self.app_state.update_file(app_state_id, "Ingestion Successful")

    def clean_directory(self):
        shutil.rmtree(self.work_dir)

    def close(self) -> None:
        self.app_state.close()
        self.filestore.close()
        self.vector_store.close()

    def search(self, q: str, k: int):
        result = self.vector_store.get(q, k)
        return result


## SAMPLE IMPLIMENTATION

# from backend.ingestors.audio_ingestor import AudioIngestor
# from backend.ingestors.image_ingestor import ImageOCRIngestor
# from backend.ingestors.text_ingestor import TextIngestor
# from backend.ingestors.pdf_ingestor import PdfIngestor
# from backend.chunker.recursive_chunker import RecursiveChunker
# from backend.vector_store.local_vec_store import ChromaDBVectorStore

# path = "backend/eval/data/work"
# app_1 = EvalApp(name="demo",
#                 description="demo app",
#                 ingestors=[
#                     AudioIngestor(accepted_formats=["wav"]),
#                     ImageOCRIngestor(accepted_formats=["jpg"]),
#                     TextIngestor(accepted_formats=["md","txt"]),
#                     PdfIngestor(accepted_formats="pdf")
#                 ],
#                 chunker=RecursiveChunker(chunk_size=512,overlap=64),
#                 vector_store = ChromaDBVectorStore(path),
#                 )
