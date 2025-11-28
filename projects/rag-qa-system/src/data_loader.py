import json
from pathlib import Path

import arxiv
import pymupdf4llm
from src.utils import Chunk, ChunkMetadata, chunk_text
from tqdm import tqdm


class _PDFDocument:
    def __init__(self, path: Path):
        """Initialize with a PDF path.
        Converts the PDF to markdown and stores the markdown text.

        Args:
            path: Path to the PDF file.
        """
        self.path = path
        try:
            self.md_text = pymupdf4llm.to_markdown(str(self.path))
        except Exception as e:
            print(f"Error parsing PDF {self.path}: {e}")
            self.md_text = ""

    def get_name(self) -> str:
        return self.path.stem

    def get_arxiv_id(self) -> str:
        return self.get_name().split("_")[0]


class CorpusLoader:
    def __init__(self):
        self.chunks = None
        self.arxivClient = arxiv.Client()

    def parse_pdfs(
        self, pdf_paths: list[Path], chunk_size: int = 500, overlap: int = 50
    ):
        """Parse PDFs and create chunks."""
        pdf_documents = [
            _PDFDocument(path) for path in tqdm(pdf_paths, desc="Parsing PDFs")
        ]
        arxiv_ids = [doc.get_arxiv_id() for doc in pdf_documents]
        print("Fetching arxiv metadata...")
        metadata_list = [self._fetch_arxiv_metadata(id) for id in tqdm(arxiv_ids)]
        self.chunks = [
            chunk
            for i, pdf_document in enumerate(pdf_documents)
            for chunk in chunk_text(
                pdf_document.md_text,
                chunk_size=chunk_size,
                overlap=overlap,
                parent_document_name=pdf_document.get_name(),
                metadata=metadata_list[i],
            )
        ]

    def save_chunks(self, output_path: Path):
        if self.chunks is None:
            raise ValueError("Chunks not loaded")
        with open(output_path, "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk) + "\n")

    def get_chunks(self) -> list[Chunk]:
        if self.chunks is None:
            raise ValueError("Chunks not loaded")
        return self.chunks

    def get_statistics(self) -> dict:
        if self.chunks is None:
            raise ValueError("Chunks not loaded")

        token_counts = [chunk["token_count"] for chunk in self.chunks]
        return {
            "num_chunks": len(self.chunks),
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "mean_tokens": sum(token_counts) / len(token_counts),
        }

    def _fetch_arxiv_metadata(self, arxiv_id: str) -> ChunkMetadata:
        search = arxiv.Search(id_list=[arxiv_id])
        r = next(self.arxivClient.results(search))
        if arxiv_id not in r.entry_id:
            print(f"WARINING: ArXiv ID mismatch: {arxiv_id} != {r.entry_id}")
        return {
            "arxiv_id": arxiv_id,
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "year": r.published.year,
            "url": r.pdf_url,
        }
