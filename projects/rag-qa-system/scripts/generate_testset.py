from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from src.utils import EVAL_DATA_DIR, Chunk, get_openai_api_key, load_chunks_from_jsonl


def _convert_chunk_to_doc(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk["chunk_text"],
        metadata={**chunk["metadata"], "id": chunk["chunk_id"]},
    )


def _get_testset_generator() -> TestsetGenerator:
    get_openai_api_key()

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    return TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)


def generate_testset():
    """Generate a testset of 40 questions from the chunks."""
    chunks = load_chunks_from_jsonl()
    # TODO: cost concern
    documents = [_convert_chunk_to_doc(chunk) for chunk in chunks]

    generator = _get_testset_generator()

    dataset = generator.generate_with_langchain_docs(documents, testset_size=40)
    output_path = EVAL_DATA_DIR / "ragas_testset.jsonl"
    dataset.to_jsonl(str(output_path))


if __name__ == "__main__":
    generate_testset()
