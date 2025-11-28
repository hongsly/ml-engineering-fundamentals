from pathlib import Path

from src.data_loader import CorpusLoader
from src.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def build_index():

    pdf_paths = list(RAW_DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs")
    # pdf_paths = pdf_paths[:2]  # TODO: remove
    corpus_loader = CorpusLoader()
    corpus_loader.parse_pdfs(pdf_paths)
    print(f"Parsing complete")
    corpus_loader.save_chunks(PROCESSED_DATA_DIR / "chunks.jsonl")
    print(f"Saved {len(corpus_loader.get_chunks())} chunks")
    print(f"Statistics: {corpus_loader.get_statistics()}")

    chunks = corpus_loader.get_chunks()
    vector_store = VectorStore()
    vector_store.add_chunks(chunks)
    vector_store.save(PROCESSED_DATA_DIR / "rag_index.faiss")
    print(f"Saved index to {PROCESSED_DATA_DIR / 'rag_index.faiss'}")

if __name__ == "__main__":
    build_index()