# Day 31: Ragas Testset Generation - Cost Analysis & Planning

**Date**: 2025-11-27 (Week 5, Day 4 / Day 31)
**Context**: RAG Q&A System - Evaluation planning, Ragas cost discovery
**Time**: ~1 hour planning + knowledge check
**Knowledge Check**: 94% (A)

---

## Session Overview

**Implementation & Planning**: Completed ArXiv metadata implementation and created test generation script. Pivoted to cost analysis and sampling strategy after discovering Ragas TestsetGenerator costs during testing.

**Key Discovery**: Ragas TestsetGenerator is much more expensive than initially estimated due to multi-phase knowledge graph construction ($10-15 vs $0.13).

**What Was Completed**:
- ✅ Added ArXiv metadata (title, authors, year, URL) to all chunks
- ✅ Regenerated all 1500+ chunks with enriched metadata
- ✅ Created `scripts/generate_testset.py` with full Ragas integration
- ✅ Researched Ragas 0.3.9 API and Ollama support
- ✅ Discovered cost explosion and designed sampling solution
- ✅ Knowledge check: 94% (A)

---

## Project Context Questions

### Q1: Why No Reranking/Cross-Encoder?

**User Question**: "Why are we not doing reranking/cross encoder?"

**Answer**: **Scope decision for time-boxed portfolio project**
- Time constraint: ~12 hours total, Day 4-5 are evaluation + deployment
- Complexity: Cross-encoder adds another component (ColBERTv2, BGE-reranker, or API)
- Cost: Would require local model inference or additional API calls
- Diminishing returns: Hybrid retrieval already achieves 80% Precision@5

**Listed in future improvements** (project-plan.md:377):
> "1. Add reranking with cross-encoder (retrieve 20 → rerank → top 5)"

**Interview narrative**:
> "For the initial implementation, I focused on hybrid retrieval with RRF fusion which achieved 80% precision. If I were to scale this system, the next step would be adding a cross-encoder reranking stage—retrieve 20 candidates with hybrid search, then rerank to top-5 with a cross-encoder like BGE-reranker. This would likely push precision to 85-90% but adds latency (~50ms per query) and complexity."

---

### Q2: How Was 500 Token Chunk Size Determined?

**User Question**: "How was the 500 chunk size determined? Is it the standard size?"

**Answer**: **Yes, 500 tokens is a common default in RAG systems**

**Rationale**:
- 500 tokens ≈ 1-2 paragraphs of dense academic text
- **Not too large**: 1000+ tokens dilute relevance (multiple topics in one chunk)
- **Not too small**: 128 tokens lose context (incomplete thoughts)
- **Overlap (50 tokens)**: Prevents splitting related sentences across chunks

**Industry standards**:
- Short chunks (256-512): QA, factual lookup
- Medium chunks (512-1024): General RAG ✅ (our choice)
- Long chunks (1024-2048): Long-form generation, summarization

**Interview talking point**:
> "I used 500-token chunks with 50-token overlap, a standard choice for academic papers. In production, I'd A/B test chunk sizes (256, 512, 1024) and measure retrieval recall to optimize for the specific query distribution."

---

### Q3: Should We Try RAPTOR?

**User Question**: "Shall we try other chunking method e.g. RAPTOR? (maybe later after main project complete?)"

**Answer**: **Not for main project, but great post-project experiment!**

**RAPTOR complexity**:
- Builds hierarchical tree of summaries (recursive clustering + summarization)
- Requires LLM calls to generate summaries ($$$ + time)
- Retrieval happens at multiple levels (leaf chunks + intermediate summaries)
- Implementation: +2-3 hours minimum

**Recommendation**:
- ✅ **For main project**: Stick with fixed-size chunking (500 tokens)
- 📝 **After project**: Experiment with chunking strategies
  - Fixed-size (baseline)
  - Semantic chunking (split on topic boundaries)
  - RAPTOR (hierarchical summaries)
  - Sentence-window retrieval

---

## Ragas TestsetGenerator: API & Cost Analysis

### Ragas 0.3.9 API

**User Question**: "Do I use generate_with_langchain_docs / generate_with_llamaindex_docs?"

**Answer**: Use `generate_with_langchain_docs()` (for raw chunks → Langchain Documents)

**Full code**:
```python
from langchain.schema import Document
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Convert chunks to Langchain Documents
documents = [
    Document(
        page_content=chunk["chunk_text"],
        metadata={
            "chunk_id": chunk["chunk_id"],
            "arxiv_id": chunk["metadata"]["arxiv_id"],
            "title": chunk["metadata"]["title"],
        }
    )
    for chunk in chunks
]

# Initialize with gpt-4o-mini
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
)

# Generate
dataset = generator.generate_with_langchain_docs(
    documents,
    testset_size=40
)
```

---

### Can We Use Ollama?

**User Question**: "Is it possible to use Ollama for ragas?"

**Answer**: **Not reliably ❌**

**Known issues** (GitHub #762, #1456, #1576, #2055):
1. **Missing `agenerate_prompt` support** → ValueError: "No nodes that satisfied the given filter"
2. **Gets stuck** at "Generating common concepts" step → Tokenization endpoint becomes unresponsive
3. **No working solution** → Ragas maintainers recommend: "Use ChatOpenAI instead"

**Why it fails**:
- TestsetGenerator requires async generation methods
- Ollama via LangChain doesn't support this
- Knowledge graph construction requires complex multi-phase processing

**Recommendation**: **Use OpenAI gpt-4o-mini** ($0.50 total for project, proven to work)

---

### Cost Discovery: The $10-15 Problem

**What User Found**:
- Ran TestsetGenerator on **200 chunks**
- Observed **200K tokens** → **$0.70** (just SummaryExtractor phase!)
- More phases to follow: NER, theme extraction, query generation

**Original estimate** (my error):
- 1500 chunks → ~$0.11 corpus + ~$0.024 generation = ~$0.13 ❌

**Reality** (user's finding):
- 1500 chunks → ~$5.25 (SummaryExtractor) × 2-3 (all phases) = **$10-15** 😱

**Why the discrepancy?**
- I underestimated Ragas' multi-phase knowledge graph construction
- Each phase processes the entire corpus multiple times
- Not just simple question generation

**User demonstrated senior MLE cost awareness** by catching this early! ⭐

---

## Revised Strategy: Sampling

**Problem**: 1500 chunks → $10-15 (over budget)

**Solution**: Sample representative chunks

### Sampling Strategy

```python
def sample_chunks_by_paper(chunks: list[Chunk], n: int = 250) -> list[Chunk]:
    """Sample chunks uniformly across papers for representative coverage."""
    from collections import defaultdict
    import random

    by_paper = defaultdict(list)
    for chunk in chunks:
        arxiv_id = chunk["metadata"]["arxiv_id"]
        by_paper[arxiv_id].append(chunk)

    num_papers = len(by_paper)
    per_paper = n // num_papers

    sampled = []
    for paper_chunks in by_paper.values():
        if len(paper_chunks) <= per_paper:
            sampled.extend(paper_chunks)
        else:
            # Take first, last, and random middle chunks
            indices = [0, len(paper_chunks)-1]  # First and last
            middle = random.sample(range(1, len(paper_chunks)-1), per_paper-2)
            indices.extend(middle)
            sampled.extend([paper_chunks[i] for i in sorted(indices)])

    return sampled[:n]
```

**Coverage**:
- 250 chunks across 32 papers = **7-8 chunks per paper**
- First + last + middle chunks → Representative coverage
- Spans all topics in corpus

**Cost**:
- 250 chunks → ~$0.85 (all phases) ✅
- Generate 20 Ragas questions
- Total: 10 manual + 20 Ragas = **30 questions**
- **Total cost**: ~$1.25 (generation + evaluation)

**Cost comparison**:
```
Original plan: 1500 chunks, 40 questions → $10-15 ❌
Sampling:      250 chunks, 20 questions → $1.25 ✅ (8× cheaper)
Manual only:   0 chunks, 10 questions → $0.50 ✅ (cheapest but limited)
```

---

## Manual vs Ragas Test Format

**User Question**: "This is very different format, right?"

**Answer**: Yes, very different!

### Format Comparison

**Manual format** (test_questions.json):
```json
{
  "id": "factual_1",
  "question": "What is Retrieval-Augmented Generation?",
  "type": "factual",
  "expected_sources": ["Lewis et al., 2020"]
}
```

**Ragas format** (ragas_testset.jsonl):
```json
{
  "user_input": "What role does the University of Washington play...",
  "reference_contexts": ["<full chunk text>", "<another chunk text>"],
  "reference": "The University of Washington is affiliated with...",
  "persona_name": "NLP Research Scientist",
  "query_style": "PERFECT_GRAMMAR",
  "query_length": "LONG",
  "synthesizer_name": "single_hop_specific_query_synthesizer"
}
```

### Key Differences

| Field | Manual | Ragas | Purpose |
|-------|--------|-------|---------|
| Question | `question` | `user_input` | The query |
| Ground truth answer | ❌ None | ✅ `reference` | Expected answer |
| Ground truth contexts | ❌ `expected_sources` (vague) | ✅ `reference_contexts` (exact chunks) | Which chunks should be retrieved |
| Metadata | Basic `type` | Rich (persona, style, length, synthesizer) | Query characteristics |

### Which Metrics Need Ground Truth?

**User's excellent question**: "Why does Context Precision need ground truth? Recall yes, but precision?"

**Corrected answer**:

**Must have ground truth**:
- **Context Recall**: Can't measure recall without knowing what should have been retrieved
  - Formula: # ground truth contexts retrieved / # ground truth contexts
  - Need to know denominator!

**Better with ground truth** (but not required):
- **Context Precision**: Can work via LLM-as-judge, but ground truth is more objective
  - With ground truth: Direct comparison (retrieved ∩ ground truth) / retrieved
  - Without ground truth: LLM judges relevance (subjective)

**Don't need ground truth**:
- **Faithfulness**: LLM judges if answer is grounded in retrieved context
- **Answer Relevance**: LLM judges if answer addresses question

**User caught my error** - Context Precision doesn't strictly require ground truth (LLM-as-judge works), but having it makes evaluation more objective! ⭐

---

## Interview Talking Points

### Cost-Conscious Engineering ⭐

> "I initially planned to generate 40 synthetic questions from all 1,500 chunks, but after running a small test, I realized Ragas' knowledge graph construction would cost $10-15 just for test generation. So I sampled 250 representative chunks (7-8 per paper) and generated 20 questions for $1.25 instead. Combined with 10 manual edge cases, I had 30 diverse questions for evaluation at 8× lower cost. This demonstrates cost-conscious engineering while maintaining evaluation quality."

### Chunking Strategy

> "I used 500-token chunks with 50-token overlap, a standard choice for academic papers. This balances semantic completeness (1-2 paragraphs) with retrieval precision. In production, I'd A/B test different chunk sizes (256, 512, 1024) and measure retrieval recall to optimize for the specific query distribution."

### Reranking Trade-off

> "For the initial implementation, I focused on hybrid retrieval with RRF fusion which achieved 80% precision—sufficient for a portfolio project. If I were to scale this system, the next step would be adding a cross-encoder reranking stage to push precision to 85-90%, though this adds ~50ms latency per query."

---

## Key Formulas

**Sampling ratio**:
```
Chunks per paper = Total sampled / Number of papers
Coverage = (First + Last + Random middle) per paper
```

**Cost estimation**:
```
Cost per chunk = Observed tokens / Chunks tested × Price per token
Total cost = (Chunks / Tested chunks) × Observed cost × Phase multiplier
```

**Example** (Day 31):
```
Observed: 200 chunks → 200K tokens → $0.70 (SummaryExtractor only)
Per chunk: 200K / 200 = 1000 tokens/chunk
Full corpus: 1500 chunks × 1000 tokens = 1.5M tokens
Phase multiplier: 2-3× (SummaryExtractor + NER + themes + generation)
Total: 1.5M × $0.15/1M × 2.5 = $0.56 (if just generation)
BUT Ragas has more complex processing → observed $0.70 for 200 chunks
Extrapolated: $0.70 / 200 × 1500 = $5.25 (just one phase)
All phases: $5.25 × 2-3 = $10-15
```

---

## Common Pitfalls

### 1. Underestimating Ragas Costs
❌ **Wrong**: Assume simple question generation ($0.11 for 1500 chunks)
✅ **Correct**: Test on small sample first, observe multi-phase knowledge graph costs ($10-15 for 1500 chunks)

### 2. Ollama for TestsetGenerator
❌ **Wrong**: Try to use Ollama to save API costs
✅ **Correct**: Use OpenAI (proven to work), catch cost issues via sampling instead

### 3. All-or-nothing Test Generation
❌ **Wrong**: Generate from all 1500 chunks or skip Ragas entirely
✅ **Correct**: Sample representative subset (250 chunks, 7-8 per paper)

### 4. Ignoring Ground Truth Advantages
❌ **Wrong**: "Manual questions are simpler, let's just use those"
✅ **Correct**: Ragas format with ground truth enables objective Context Recall measurement

---

## Implementation Details (Day 31)

### ArXiv Metadata Enhancement ✅ COMPLETED

**What was added**:
- Title, authors, year, URL for each paper
- Extracted via ArXiv API during chunk generation
- Enriched all 1500+ chunks with metadata

**Impact**:
- Better citations in generated answers
- Enables provenance tracking
- Supports future features (filter by year, author search)

### scripts/generate_testset.py ✅ COMPLETED

**TODO**: Add sampling function (deferred to Day 32)

---

## Deferred to Day 32

1. **Implement sampling**: Add `sample_chunks_by_paper()` to generate_testset.py (15 min)

2. **Generate Ragas questions**: Run on 250 sampled chunks (~$0.85, 30 min)

3. **Actual Ragas evaluation**: How do the metrics perform in practice? (45 min)

4. **Retrieval metrics**: Measure actual Recall@K, MRR, NDCG (30 min)

---

**Status**: Day 31 complete! Implementation + cost analysis done, ready for evaluation tomorrow (Day 32).

**Key Achievements**:
- ✅ ArXiv metadata added to all chunks
- ✅ scripts/generate_testset.py created with Ragas integration
- ✅ User caught cost explosion early ($10-15 vs $1.25) - senior MLE cost awareness! ⭐
- ✅ Sampling strategy designed (250 chunks → 8× cost savings)

**Files Created**:
- `scripts/generate_testset.py` (Ragas TestsetGenerator integration)
- All chunks regenerated with ArXiv metadata

**Next (Day 32)**:
1. Implement sampling in generate_testset.py - 15 min
2. Generate 20 Ragas questions from 250 sampled chunks - 30 min (~$0.85)
3. Create evaluation/evaluate_rag.py - 45 min
4. Create evaluation/evaluate_retrieval.py - 30 min
5. Run comprehensive evaluation on 30 questions - 30 min (~$0.40)
6. Error analysis and reporting - 15 min
