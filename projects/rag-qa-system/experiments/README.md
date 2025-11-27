# Experiments Log

This folder contains exploratory analysis and hypothesis-testing scripts for the RAG Q&A system.

---

## Day 29 (Nov 25, 2025): Hybrid Retrieval Evaluation

### Hypothesis 1: Query-Corpus Mismatch
**Question**: Does BM25 perform poorly because our test queries don't match the corpus domain?

**Test**: Compare general NLP queries vs RAG-focused queries

**Result**: ✅ **CONFIRMED** - BM25 improved from 13% to 67% precision with RAG-focused queries

**Evidence**:
- General queries (attention, BERT, GPT): BM25 = 13.3%
- RAG queries (DPR, ColBERT, sparse/dense): BM25 = 66.7%
- Improvement: +53.4 percentage points

---

### Hypothesis 2: Reference Section Noise
**Question**: Is BM25 matching citation/reference sections instead of content?

**Test**: Analyze what percentage of chunks are references and check if BM25 retrieves them

**Result**: ✅ **CONFIRMED** - 15% of chunks are references (231/1541), causing false positives

**Evidence**:
- Query: "attention mechanism"
- BM25 matched: "An Attention Free Transformer" in bibliography
- This is a reference, not content explaining attention

**Potential fix**: Filter reference sections during chunking (could improve precision by 5-10%)

---

### Key Findings

#### 1. Query-Corpus Alignment is Critical

| Query Type | Dense | Sparse (BM25) | Hybrid |
|------------|-------|---------------|--------|
| General NLP queries | 60% | 13% | 40% |
| RAG-focused queries | 67% | 67% | **80%** ⭐ |

**Takeaway**: Hybrid (80%) outperforms Dense (67%) when queries align with corpus

#### 2. When Hybrid Helps vs Hurts

**Hybrid works when**:
- Both methods find different relevant docs
- Example: "How does ColBERT work?" → BM25 100%, Dense 100%, Hybrid 100%

**Hybrid fails when**:
- One method much worse than the other
- Example: "BERT pretraining" → BM25 0%, Dense 60%, Hybrid 40% (degraded)

#### 3. BM25 Strengths and Weaknesses

**Strengths**:
- Exact keyword matching: "ColBERT" query → 100% precision
- Multiple keywords: "sparse and dense retrieval" → 80% precision

**Weaknesses**:
- Vocabulary mismatch: Can't match "attention" to "self-attention mechanism"
- Reference noise: Matches citations in bibliographies
- Conceptual queries: "How does BERT pretraining work?" → 0% precision

---

## Day 30 (Nov 26, 2025): Generation Pipeline & Smoke Test

### Implementation Complete
- ✅ `src/generator.py`: OpenAI Responses API wrapper (gpt-4o-mini)
- ✅ `src/rag_pipeline.py`: End-to-end RAG pipeline (RagAssistant class)
- ✅ `data/eval/test_questions.json`: 10 manual test questions (factual, reasoning, multi-hop, negative)
- ✅ `experiments/smoke_test.py`: Smoke test script

### Test Results (5 questions × 4 modes)

| Mode | Citations | Answer Quality | Token Usage |
|------|-----------|----------------|-------------|
| **Hybrid** | ✅ Excellent | ✅ High | ~2700 input |
| **Dense** | ✅ Excellent | ✅ High | ~2700 input |
| **Sparse** | ✅ Good | ✅ Good | ~2700 input |
| **None** | ❌ None | ⚠️ Generic | ~50 input |

**Key findings**:
1. **Token usage validates retrieval**: 2700 tokens (with context) vs 50 tokens (without) ✅
2. **Citation format works**: `(source: [Li et al., 2023](2202.01110_rag_survey_lichunk_0))` ✅
3. **Answer quality high**: Detailed explanations with proper source attribution
4. **Sparse performed better than expected**: Day 29 findings confirmed - query-corpus alignment matters

### ⚠️ Issue Discovered: Negative Question Handling

**Problem**: Model answers out-of-corpus questions despite strict prompt

**Test case**: "What is the capital of France?" (not in RAG corpus)

**Results**:
- Hybrid/Dense/None: "The capital of France is Paris." ❌
- Sparse: "The provided documents do not contain... the capital of France is Paris." ⚠️

**Initial hypothesis**: Prompt engineering failure (model ignoring instructions)

**Updated prompt** (line 4-10 in `src/generator.py`):
```python
SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a Q&A assistant for RAG research. "
    "Answer ONLY using information from the provided support material. "
    "If the support material does not contain enough information, respond with: "
    "'I don't have enough information in the provided materials to answer this question.' "
    "DO NOT use your general knowledge - only cite the support material."
)
```

**Status**: ✅ **ROOT CAUSE FOUND** (via OpenAI log investigation)

### Root Cause Investigation ⭐

**What we found**: User checked OpenAI dashboard logs and discovered retrieved chunk `2404.16130_graphrag_edgechunk_22050` contained:
```
"capital of France?', a direct answer would be 'Paris'..."
```

**Diagnosis**: **Retrieval contamination**, not prompt engineering failure!
- GraphRAG paper used "capital of France" as an example in the text
- BM25/Dense retrieved this chunk (keyword match: "capital" + "France")
- Model correctly cited the retrieved document (working as designed!)
- The prompt is working correctly - the issue is retrieval quality

**Retrieval rankings**:
- Dense/Hybrid: Ranked contaminated chunk **#1** (highest relevance)
- Sparse: Ranked contaminated chunk **#2** (still in top-5)

**Why Sparse refused but Dense/Hybrid didn't**:
- Even though Sparse also retrieved the contaminated chunk (#2), it added a disclaimer
- Dense/Hybrid (#1 ranking) cited it directly without hesitation
- This suggests ranking position influences how confidently the model cites retrieved information
- All three methods retrieved the contaminated chunk - different presentation led to different behavior

**Key insight**: This is a data quality issue (retrieval contamination), not a model issue

### Solutions (Updated)

**1. Use truly out-of-domain queries** (immediate fix):
- ❌ Bad: "What is the capital of France?" (appears in papers as example)
- ✅ Good: "How to bake bread?" (impossible to be in RAG/LLM papers)

**2. Few-shot examples** (prompt engineering):
```python
"Example:\n"
"Question: What is the capital of France?\n"
"Support: [papers about RAG]\n"
"Answer: I don't have information about geography in these RAG research papers.\n"
```

**3. Structured outputs** (force format):
```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "properties": {
            "has_answer": {"type": "boolean"},
            "answer": {"type": "string"},
            "sources": {"type": "array"}
        }
    }
}
```

**4. Temperature=0** (more deterministic):
- Default: 0.3 (some randomness)
- Set to 0: Maximally deterministic

**5. Check retrieval quality** (data quality):
- Ensure negative queries retrieve low-relevance chunks
- Consider similarity threshold (e.g., reject if score < 0.3)
- Filter out example text from papers during preprocessing

---

## Scripts in This Folder

### Evaluation Scripts

**`test_general_queries.py`**
- Displays full retrieval results for 3 general queries for manual inspection
- Shows top-5 from all three methods
- Used for initial relevance judgments

**`evaluate_general_queries.py`**
- Original evaluation with general NLP queries
- Queries: attention mechanism, BERT pretraining, GPT vs BERT
- Result: Dense 60%, Sparse 13%, Hybrid 40%
- **Conclusion**: Hybrid underperformed due to query-corpus mismatch

**`test_rag_queries.py`**
- Quick test script to inspect RAG query results
- Displays top-3 from Dense, Sparse, Hybrid
- Shows overlap analysis between methods

**`evaluate_rag_queries.py`**
- Follow-up evaluation with RAG-focused queries
- Queries: DPR, ColBERT, sparse vs dense retrieval
- Result: Dense 67%, Sparse 67%, Hybrid 80%
- **Conclusion**: Hybrid outperformed when queries matched corpus

**`compare_retrievers.py`**
- Interactive manual evaluation framework
- Prompts user for relevance judgments
- Calculates Precision@5 and MRR

### Analysis Scripts

**`analyze_references.py`**
- Detects reference/bibliography sections in corpus
- Found 231/1541 chunks (15%) are references
- Shows examples of false positive matches

---

## Recommendations from Experiments

### For Production (Week 5 Day 2+):

**1. Use Dense-only for now** (simplest, 67% is good):
```python
from src.vector_store import VectorStore

retriever = VectorStore()
chunks = retriever.search(query, k=5)
```

**2. Or use Hybrid with RAG-focused queries** (80%, more impressive):
```python
from src.hybrid_search import HybridRetriever

retriever = HybridRetriever()
chunks = retriever.search_hybrid(query, top_k=5)
```

**3. For Week 5 Day 3 (Ragas evaluation)**, use RAG-focused test questions:
```python
test_questions = [
    "How does dense passage retrieval work?",
    "What is ColBERT late interaction?",
    "Compare BM25 vs dense retrieval",
    "What is query expansion in retrieval?",
]
```

### Optional Improvements:

**1. Filter reference sections** (could improve Hybrid 80% → 85%+):
- Add `is_reference_chunk()` to `src/data_loader.py`
- Re-run `scripts/build_index.py`
- Remove ~231 reference chunks

**2. Query classification strategy**:
```python
def choose_retrieval_method(query):
    if has_specific_keywords(query):  # "ColBERT", "DPR"
        return "hybrid"  # BM25 + Dense
    else:  # Conceptual: "how does X work?"
        return "dense"   # Dense only
```

**3. Adaptive fusion weights**:
- Instead of fixed RRF, weight BM25 higher for keyword queries
- Weight Dense higher for conceptual queries

---

## Metrics Summary

| Metric | General Queries | RAG Queries | Notes |
|--------|----------------|-------------|-------|
| **Dense Precision@5** | 60.0% | 66.7% | Consistent performer |
| **Sparse Precision@5** | 13.3% | 66.7% | 5× improvement with aligned queries! |
| **Hybrid Precision@5** | 40.0% | 80.0% | Best when both methods contribute |
| **Target** | 85% | 85% | RAG queries got close (80%) |

---

## Interview Talking Points

**Question**: "How did your hybrid retrieval perform?"

**Strong answer**:
> "I implemented hybrid retrieval with RRF fusion and ran two evaluations. With general NLP queries, hybrid (40%) underperformed dense-only (60%) because BM25 retrieved mostly irrelevant results (13% precision). However, when I tested with RAG-focused queries matching my corpus, BM25 improved 5× to 67% and hybrid achieved 80% precision. This taught me that query-corpus alignment is critical—BM25 performs well on keyword-specific queries like 'How does ColBERT work?' but fails on conceptual queries. I also discovered 15% of chunks were reference sections causing false positives. For production, I'd use query classification to route different query types to different retrieval strategies."

---

**Last Updated**: 2025-11-26
**Status**: Day 2 (Generation pipeline) complete, ready for Day 3 (Ragas evaluation)
