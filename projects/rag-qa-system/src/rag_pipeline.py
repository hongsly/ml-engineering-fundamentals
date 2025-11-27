import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from src.generator import Generator
from src.hybrid_search import HybridRetriever

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class RagAssistant:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        retrieval_mode: Literal["hybrid", "dense", "sparse", "none"] = "hybrid",
        top_k: int = 5,
    ):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found!",
                "Create a .env file with: OPENAI_API_KEY=<your_api_key>",
            )
        self.generator = Generator(api_key)
        self.model = model

        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.retriever = None
        if retrieval_mode != "none":
            self.retriever = HybridRetriever()
            self.retriever.load_from_file(
                DATA_DIR / "rag_index.faiss", DATA_DIR / "chunks.jsonl"
            )

    def query(self, query: str) -> str:
        context = None
        match self.retrieval_mode:
            case "hybrid":
                context = self.retriever.search_hybrid(query, self.top_k)
            case "dense":
                context = self.retriever.search_dense(query, self.top_k)
            case "sparse":
                context = self.retriever.search_sparse(query, self.top_k)
        answer = self.generator.generate(query, context, model=self.model)
        return answer
