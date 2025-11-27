import pytest
from src.generator import Generator

@pytest.fixture
def generator():
    return Generator(api_key="fake_key")

def test_get_prompt_without_context(generator):
    prompt = generator._get_prompt("What is RAG?", context=None)

    print(f"Prompt: {prompt}")
    assert "<question>What is RAG?</question>" in prompt
    assert "<documents>" not in prompt

def test_get_prompt_with_context(generator):
    context = [
        {"chunk_id": "doc1_chunk0", "chunk_text": "RAG is retrieval-augmented generation."},
        {"chunk_id": "doc2_chunk1", "chunk_text": "It combines search with LLMs."}
    ]

    prompt = generator._get_prompt("What is RAG?", context=context)

    print(f"Prompt: {prompt}")
    assert "<question>What is RAG?</question>" in prompt
    assert "<documents>" in prompt
    assert '<document id="doc1_chunk0">RAG is retrieval-augmented generation.</document>' in prompt
    assert '<document id="doc2_chunk1">It combines search with LLMs.</document>' in prompt
    assert "<documents>" in prompt

