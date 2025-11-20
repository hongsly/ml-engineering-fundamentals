# Week 4 Advanced RAG: 2-3 Day Study Plan

**Created**: Day 22 (Week 4, Day 1)
**Duration**: 4-6 hours (2 hours/day × 2-3 days)
**Baseline**: 21.3% weighted readiness (23.2% high-priority)
**Target**: 55-60% weighted overall, 82% high-priority

---

## Study Strategy

### Weighted Scoring System
- **High-priority topics (28)**: 2.5× weight (70 weighted points)
- **Regular topics (58)**: 1× weight (58 weighted points)
- **Total**: 128 weighted points

### Focus Areas (Priority Order)
1. **Phase 1**: "The Fix" - Hybrid Retrieval (6 high-priority topics)
2. **Phase 2**: "The Polish" - Reranking & Quality (6 high-priority topics)
3. **Phase 3**: "The System" - Advanced Patterns (8 high-priority topics)
4. **Phase 4**: "The Proof" - Evaluation (8 high-priority topics)

### Current Gaps to Address
- **16 Dunno high-priority topics** → Target: 12 Know (75% success)
- **11 Unsure high-priority topics** → Target: 9 Know (82% consolidation)

---

## Day 1 (2 hours): Phase 1 "The Fix" + Phase 2 Core

**Goal**: Master hybrid retrieval fundamentals + reranking techniques

### Morning Block: Hybrid Retrieval Foundations (60 min)

#### 1. RRF (Reciprocal Rank Fusion) [20 min] ⭐ CRITICAL

**Status**: Dunno → Know

**Learning Objectives**:
- Write formula from memory: `Score = 1/(k + rank)`
- Calculate RRF scores for 2+ ranking lists
- Explain: Why k=60 is typical? (Prevents zero division, balances rank differences)

**Resources**:
- Blog: "Hybrid Search Explained" (Pinecone Learning Center)
- Paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"

**Practice Exercise**:
```
Given two rankings:
BM25: [doc1:1, doc3:2, doc2:3, doc4:5]
Dense: [doc2:1, doc1:2, doc4:3, doc5:4]

Calculate RRF scores (k=60) and final ranking.
```

**Interview Answer Template**:
"RRF combines multiple ranking methods by computing Score = 1/(k + rank) for each document in each ranking, then summing scores. With k=60, a rank-1 doc scores 1/61 ≈ 0.0164, rank-2 scores 1/62 ≈ 0.0161. This naturally handles missing documents (score=0) and balances contributions from different rankers."

---

#### 2. SPLADE (Learned Sparse Retrieval) [20 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Problem: BM25 fails on vocabulary mismatch (e.g., "ML engineer" vs "machine learning engineer")
- Solution: Learn sparse representations that expand to related terms
- Trade-off: SPLADE vs BM25 vs Dense embeddings

**Resources**:
- Paper: "SPLADE: Sparse Lexical and Expansion Model" (sections 1-2)
- Blog: "Beyond BM25: Neural Sparse Retrieval" (Weaviate blog)

**Key Insight**:
- BM25: Exact term match only
- Dense: Semantic similarity, but loses exact-match signals
- SPLADE: Best of both - learned expansion + sparse efficiency

**Interview Answer Template**:
"SPLADE solves BM25's vocabulary mismatch problem by learning to expand queries and documents to related terms, while maintaining sparse representations for efficiency. For 'ML engineer', it might activate 'machine learning', 'AI', 'data science'. It bridges BM25's precision and dense retrieval's recall."

---

#### 3. DPR (Dense Passage Retrieval) [20 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Dual-encoder architecture: Separate encoders for query and passage
- Contrastive learning: **In-batch negatives AND hard negatives** (both!)
- Why better than BERT for retrieval? (Independent encoding enables offline indexing)

**Resources**:
- Paper: "Dense Passage Retrieval for Open-Domain QA" (Karpukhin et al., sections 1-3)

**Key Concepts**:
```
Query encoder: BERT_q(query) → vector_q
Passage encoder: BERT_p(passage) → vector_p
Similarity: dot(vector_q, vector_p)
Training: Positive pairs + TWO types of negatives
```

**Two Types of Negatives** (Critical distinction):

1. **In-batch negatives**:
   - Use other queries' positive passages as negatives for current query
   - Efficient (no extra mining needed)
   - Usually "easy" negatives (not designed to be confusing)

2. **Hard negatives**:
   - Explicitly mine documents that score high on BM25 but are wrong answers
   - More expensive (requires mining step)
   - "Hard" because they're semantically similar but incorrect

**DPR uses BOTH**:
```
For query Q with positive passage P+:

Negatives =
  1. In-batch negatives: Positive passages from other queries in batch
  2. Hard negatives: Top BM25 results that don't contain the answer

Loss: Positive passage P+ should score higher than all negatives
```

**Interview Answer Template**:
"DPR uses dual BERT encoders - one for queries, one for passages. This allows offline passage encoding and fast retrieval via ANN search. Training uses contrastive loss with TWO types of negatives: (1) **In-batch negatives** - other queries' positive passages (efficient, but easier), and (2) **Hard negatives** - BM25 top results that don't contain the answer (more expensive to mine, but harder). Hard negatives are critical because they teach the model to distinguish semantically similar but incorrect passages."

---

### Afternoon Block: Reranking Techniques (60 min)

#### 4. Cross-Encoder Reranking [15 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Two-stage retrieval: Fast bi-encoder (retrieval) → slow cross-encoder (reranking)
- Why cross-encoders are better but slower? (Full attention between query and doc)
- Typical setup: Retrieve 100 → rerank to 10

**Architecture Comparison**:
```
Bi-encoder: BERT(query) × BERT(doc) = separate encodings
Cross-encoder: BERT(query + doc) = joint encoding
```

**Interview Answer Template**:
"Cross-encoder reranking is a two-stage approach: First, use fast bi-encoders (like SBERT) to retrieve top-100 candidates. Then, use a cross-encoder that jointly encodes [query, document] pairs for accurate scoring. Cross-encoders are 10-100× slower but more accurate because they model query-document interactions with full attention."

---

#### 5. ColBERT (Late Interaction) [15 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Token-level similarity via MaxSim operation
- **Speed vs Accuracy vs STORAGE trade-off** (storage is critical!)
- How it works: Store token embeddings, compute max similarity per query token

**Key Insight**:
```
Bi-encoder: One vector per doc (fast, less accurate, small index)
Cross-encoder: Joint encoding (slow, most accurate, no index)
ColBERT: Token vectors + MaxSim (middle speed/accuracy, LARGE index)
```

**Critical Trade-off: Storage Cost** ⚠️

ColBERT indexes are **10-50× larger** than bi-encoder indexes!

```
Example: 1M documents, avg 100 tokens per doc

Bi-encoder (DPR):
  1M docs × 1 vector × 768 dim × 4 bytes = 3 GB

ColBERT:
  1M docs × 100 tokens × 768 dim × 4 bytes = 300 GB
  → 100× larger (or 10-50× with compression/quantization)
```

**Why so large?**
- If a document has 100 tokens, ColBERT stores 100 vectors (768-dim each)
- Bi-encoder stores only 1 vector (768-dim)
- For large corpora (10M+ docs), this becomes prohibitive

**Comparison Table**:

| Method | Storage | Speed | Accuracy | Use Case |
|--------|---------|-------|----------|----------|
| Bi-encoder | 1 vector/doc | Fastest | Lowest | Large corpora (10M+ docs) |
| ColBERT | N vectors/doc | Fast | High | Medium corpora (<5M docs) |
| Cross-encoder | No index | Slowest | Highest | Reranking only (top-100) |

**Interview Answer Template**:
"ColBERT uses late interaction: Encode query and document into token-level embeddings independently. At retrieval time, for each query token, find the max similarity to any document token (MaxSim). Sum these max similarities for final score. This is faster than cross-encoders (no joint encoding) but more accurate than bi-encoders (token-level matching captures nuanced patterns like acronym expansions).

**Critical trade-off**: ColBERT indexes are **10-50× larger** than bi-encoder indexes. For a document with 100 tokens, ColBERT stores 100 vectors vs bi-encoder's 1 vector. This limits ColBERT to smaller corpora or requires aggressive quantization. In production, I'd use ColBERT for <5M documents and bi-encoders for larger corpora."

---

#### 6. MMR (Maximal Marginal Relevance) [15 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Diversity-aware reranking: Balance relevance and novelty
- Formula: MMR = λ × Sim(q, d) - (1-λ) × max Sim(d, d_i)
- When to use: Search results, summarization, recommendation

**Interview Answer Template**:
"MMR addresses result redundancy by balancing relevance and diversity. Formula: MMR(d) = λ × Sim(query, doc) - (1-λ) × max Sim(doc, already_selected_docs). The second term penalizes documents similar to already-selected ones. Typical λ=0.5 gives equal weight to relevance and diversity."

---

#### 7. Lost-in-the-Middle Problem [15 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Position bias: LLMs attend more to start and end of context
- Research finding: 20-30 document context, middle docs get ignored
- Mitigation: Reorder by relevance, compress middle docs, or use fewer docs

**Resources**:
- Paper: "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)

**Interview Answer Template**:
"The lost-in-the-middle problem shows that when given 20+ documents, LLMs pay most attention to the first and last documents, with middle documents often ignored. In RAG, this means even if we retrieve 20 relevant docs, docs ranked 8-15 might not influence the answer. Solutions: (1) Put most relevant docs at start/end, (2) Compress or summarize middle docs, (3) Use fewer but higher-quality docs."

---

### End of Day 1: Consolidation (Self-Assessment)

**Review these 4 "Unsure" topics** (10 min):
- Dense retrieval basics
- Sparse retrieval (BM25, TF-IDF)
- Hybrid retrieval (sparse + dense)
- Contrastive learning for retrieval

**For each, write 2-3 sentence interview answer**.

**Day 1 Target**: 11 topics studied (7 new + 4 consolidated)

---

## Day 2 (2 hours): Phase 3 "The System"

**Goal**: Master 2025 trends + close Gap Q177 (FiD)

### Morning Block: 2025 Trends + Critical Gaps (70 min)

#### 1. FiD (Fusion-in-Decoder) [30 min] ⭐ Gap Q177 (0% → 80%+)

**Status**: Dunno → Know

**Learning Objectives**:
- Architecture: Encode N passages independently with shared encoder
- Decoder: Jointly attend to all passage encodings
- **Critical for 2025**: FiD vs Long Context trade-offs

**Architecture Details**:
```
Input: Query + N passages
Encoder: For each passage, encode [query; passage_i] independently
  → Produces N separate hidden states
Decoder: Standard autoregressive decoder with cross-attention to ALL N encodings
  → Can attend to any token in any passage
Output: Generated answer
```

**Resources**:
- Paper: "Leveraging Passage Retrieval with Generative Models" (Izacard & Grave, 2021)
- Read: Sections 1-3, skim 4-5

**FiD vs Long Context Trade-offs** (Critical Interview Question):

| Aspect | FiD (2021) | Long Context (2025) |
|--------|------------|---------------------|
| Context length | 100 passages × 250 tokens = 25K | Gemini 1.5 Pro: 1M tokens |
| Cost | Lower (encoder only, no attention between passages) | Higher (full attention over 1M tokens) |
| Latency | Faster (parallel encoding, smaller context) | Slower (sequential, huge context) |
| Implementation | Complex (custom model, training needed) | Simple (just concatenate + prompt) |
| Accuracy | Good (trained for retrieval QA) | Excellent (full context attention) |
| **When to use FiD** | Cost-sensitive, high-throughput systems | - |
| **When to use Long Context** | Maximum accuracy, low QPS | - |

**Interview Answer Template**:
"FiD encodes each retrieved passage independently with the query, then uses a decoder that jointly attends to all passage encodings. The key trade-off in 2025 is FiD vs Long Context models:

- **FiD advantages**: Cheaper (no attention between passages), faster, works with shorter-context models
- **Long Context advantages**: Simpler implementation (just concatenate docs), better accuracy (full cross-document attention), easier to debug

In 2025, I'd choose Long Context for low-QPS, high-accuracy use cases (research assistant) and FiD for high-QPS, cost-sensitive systems (customer support at scale)."

---

#### 2. GraphRAG [20 min] ⭐ 2025 Hot Trend

**Status**: Dunno → Know

**Learning Objectives**:
- Knowledge graph + vector embeddings for retrieval
- When better than pure vector search: Entity relationships, multi-hop reasoning
- Microsoft's GraphRAG: Local-to-global approach

**Key Concepts**:
```
Traditional RAG: Query → Vector DB → Top-K docs
GraphRAG: Query → Entity extraction → Graph traversal + Vector search → Context
```

**Use Cases**:
- Multi-hop questions: "Who is the CEO of the company that acquired Instagram?"
- Entity-centric search: "All projects led by Alice in 2023"
- Relationship reasoning: "Companies that compete with both Google and Microsoft"

**Resources**:
- Blog: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft, 2024)

**Interview Answer Template**:
"GraphRAG combines knowledge graphs with vector embeddings. Instead of pure semantic search, it extracts entities from the query, traverses the knowledge graph to find related entities and relationships, then retrieves documents connected to those entities. This is especially powerful for multi-hop reasoning ('CEO of company that acquired Instagram' → Facebook → Mark Zuckerberg) where pure vector search struggles."

---

#### 3. RAFT (Retrieval Augmented Fine Tuning) [20 min] ⭐ 2025 Hot Topic

**Status**: Dunno → Know

**Learning Objectives**:
- The debate: "Should I use RAG or fine-tune?" → Answer: RAFT does both
- Training: Fine-tune on (question, retrieved docs, answer) triplets
- Key insight: Teach model to ignore distractor documents

**Training Data Format**:
```
Input: Question + Retrieved Docs (some relevant, some distractors)
Output: Answer citing only relevant docs
Loss: Supervise on both answer quality AND ignoring distractors
```

**Why Better than Vanilla RAG**:
- Standard RAG: Model hasn't seen retrieval-augmented format during training
- RAFT: Model trained to extract info from retrieved context, ignore distractors

**Resources**:
- Paper: "RAFT: Adapting Language Model to Domain Specific RAG" (Zhang et al., 2024)

**Interview Answer Template**:
"RAFT resolves the 'RAG vs fine-tune' debate by doing both. It fine-tunes the LLM on domain-specific (question, retrieved_docs, answer) triplets, teaching the model to extract information from retrieval context while ignoring distractor documents. This is crucial because base LLMs aren't trained on retrieval-augmented inputs, so they struggle to distinguish relevant vs irrelevant retrieved passages. RAFT explicitly teaches this skill."

---

### Afternoon Block: Advanced Patterns (50 min)

#### 4. Agentic RAG [15 min]

**Status**: Dunno → Know

**Learning Objectives**:
- LLM as agent: Decides when to retrieve, what to query, multi-step reasoning
- Patterns: ReAct (Reason + Act), Tool use, Self-reflection

**Example Flow**:
```
User: "What's the average salary of ML engineers in Seattle in 2024?"

Agent:
1. Thought: Need current data → Retrieve
2. Action: search("ML engineer salary Seattle 2024")
3. Observation: [Retrieved docs with ranges]
4. Thought: Need to calculate average → Use calculator
5. Action: calculate(mean([150k, 180k, 160k]))
6. Answer: "The average ML engineer salary in Seattle in 2024 is approximately $163k"
```

**Interview Answer Template**:
"Agentic RAG treats the LLM as an autonomous agent that decides when and how to retrieve. Instead of always retrieving before generation, the agent reasons about whether retrieval is needed, formulates queries, evaluates results, and may retrieve multiple times. This uses patterns like ReAct (reasoning traces + actions) and tool use. It's especially powerful for complex queries requiring multi-step reasoning or calculations."

---

#### 5. Multi-Hop Retrieval [15 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Iterative: retrieve → reason → retrieve again
- Use case: Questions requiring chaining information

**Example**:
```
Question: "What university did the founder of Tesla attend?"

Step 1: Retrieve on "founder of Tesla"
  → Extract: Elon Musk
Step 2: Retrieve on "Elon Musk university"
  → Answer: University of Pennsylvania
```

**Interview Answer Template**:
"Multi-hop retrieval handles questions that require chaining information across multiple documents. For 'What university did the founder of Tesla attend?', we first retrieve 'founder of Tesla' → extract 'Elon Musk', then retrieve 'Elon Musk education'. This iterative retrieve → extract entities → retrieve again pattern is essential for compositional questions where a single retrieval step can't find the answer."

---

#### 6. Parent Document Retrieval [10 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Strategy: Retrieve on small chunks (high precision), return full parent document (full context)
- Trade-off: Chunk size vs context window usage

**Interview Answer Template**:
"Parent document retrieval splits documents into small chunks for indexing (e.g., 128 tokens) to maximize retrieval precision, but returns the full parent document or larger section (e.g., 1024 tokens) to provide sufficient context for generation. This balances retrieval precision (small chunks match better) with generation quality (larger context has more information)."

---

#### 7. Complex PDF Parsing / OCR [10 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Real-world challenge: PDFs have tables, multi-column layouts, images
- Tools: unstructured.io, LlamaParse, Docling (IBM)
- Strategy: Layout-aware chunking preserves table structure

**Interview Question**: "How do you handle tables in RAG?"

**Interview Answer Template**:
"Tables in PDFs are challenging because naive text extraction loses structure. Solutions: (1) Use tools like unstructured.io or LlamaParse that preserve table structure as markdown/HTML, (2) Chunk tables separately from text (don't split mid-table), (3) For multi-modal models, keep tables as images, (4) Generate textual descriptions of tables for embedding. In production, I'd use unstructured.io's table detection + markdown conversion for consistent representation."

---

### End of Day 2: Consolidation (Self-Assessment)

**Review these 3 "Unsure" topics** (10 min):
- Multi-modal RAG (text + images + tables)
- Standard RAG pipeline (Query → Retrieve → Augment → Generate)
- Query decomposition (multi-hop → sub-questions)

**Day 2 Target**: 10 topics studied (7 new + 3 consolidated)

---

## Day 3 (2 hours, Optional): Phase 4 "The Proof" + Final Consolidation

**Goal**: Master evaluation metrics + solidify all Unsure topics

### Morning Block: Evaluation Metrics (60 min)

#### 1. Retrieval Metrics Formulas [30 min] ⭐ CRITICAL

**Status**: Unsure → Know

**Learning Objectives**:
- Calculate by hand: Recall@K, Precision@K, MRR, NDCG
- When to use which metric

**Recall@K**:
```
Recall@K = (# relevant documents in top-K) / (total # relevant documents in collection)

Example:
Query: "machine learning papers"
Total relevant in collection: 10 papers
Top-5 retrieved: [relevant, irrelevant, relevant, relevant, irrelevant]
Recall@5 = 3/10 = 30%
```

**Precision@K**:
```
Precision@K = (# relevant documents in top-K) / K

Using same example:
Precision@5 = 3/5 = 60%
```

**MRR (Mean Reciprocal Rank)**:
```
MRR = Average of (1 / rank_of_first_relevant_doc)

Example across 3 queries:
Query 1: First relevant at rank 2 → 1/2 = 0.5
Query 2: First relevant at rank 1 → 1/1 = 1.0
Query 3: First relevant at rank 5 → 1/5 = 0.2
MRR = (0.5 + 1.0 + 0.2) / 3 = 0.567
```

**NDCG (Normalized Discounted Cumulative Gain)**:
```
DCG@K = Σ(relevance_i / log2(i+1)) for i=1 to K
NDCG@K = DCG@K / IDCG@K (ideal DCG with perfect ranking)

Example (K=3, relevance scores 0-3):
Retrieved: [rel=3, rel=1, rel=2]
DCG@3 = 3/log2(2) + 1/log2(3) + 2/log2(4)
      = 3/1 + 1/1.585 + 2/2
      = 3 + 0.631 + 1
      = 4.631

Ideal: [rel=3, rel=2, rel=1]
IDCG@3 = 3/1 + 2/1.585 + 1/2 = 4.762

NDCG@3 = 4.631 / 4.762 = 0.972
```

**When to Use Which**:
- **Recall@K**: When you care about finding all relevant items (e-discovery, medical search)
- **Precision@K**: When top results matter most (web search, recommendation)
- **MRR**: When you only need one good answer (question answering)
- **NDCG**: When relevance has degrees (not binary), and order matters (search ranking)

**Practice Exercise**: Calculate all 4 metrics for this scenario:
```
Query: "transformer architecture"
Total relevant in collection: 8 documents
Retrieved top-5: [doc_A (rel=3), doc_B (rel=0), doc_C (rel=2), doc_D (rel=1), doc_E (rel=0)]
```

**Interview Answer Template**:
"For RAG retrieval evaluation, I use multiple metrics:
- **Recall@K** measures what fraction of all relevant docs we retrieved
- **Precision@K** measures what fraction of retrieved docs are relevant
- **MRR** focuses on the rank of the first relevant doc - good for single-answer QA
- **NDCG** handles graded relevance and rewards putting highly relevant docs at the top

In production, I'd track Recall@10 (did we retrieve enough relevant docs?), NDCG@10 (is the ranking good?), and MRR (for QA use cases)."

---

#### 2. End-to-End RAG Metrics [20 min]

**Status**: Unsure → Know

**Learning Objectives**:
- Faithfulness: Does answer cite sources correctly?
- Answer Relevance: Does answer address the question?
- Context Precision/Recall: Are retrieved docs relevant?

**RAGAs Framework Metrics**:
```
1. Faithfulness (Answer Groundedness)
   - Question: Is the answer supported by the retrieved context?
   - Method: Check if answer claims can be traced to source docs
   - Score: % of claims that are grounded

2. Answer Relevance
   - Question: Does the answer address the user's question?
   - Method: Semantic similarity between question and answer
   - Score: Cosine similarity or LLM judgment

3. Context Precision
   - Question: Are retrieved docs relevant to the question?
   - Method: What % of retrieved docs contribute to the answer?
   - Score: Precision-like metric on retrieved docs

4. Context Recall
   - Question: Does retrieved context contain info to answer the question?
   - Method: Can ground truth answer be derived from context?
   - Score: Binary or graded scale
```

**Resources**:
- RAGAs documentation: https://docs.ragas.io/
- Paper: "RAGAS: Automated Evaluation of Retrieval Augmented Generation"

**Interview Answer Template**:
"For end-to-end RAG evaluation, I use the triad of metrics from RAGAs:
1. **Faithfulness** - Is the answer grounded in retrieved docs? Prevents hallucination.
2. **Answer Relevance** - Does the answer actually address the question? Prevents off-topic answers.
3. **Context Quality** - Are retrieved docs relevant (precision) and complete (recall)?

These catch different failure modes: Faithfulness catches hallucination, Answer Relevance catches misunderstanding, Context Quality catches retrieval failures."

---

#### 3. Context Window Optimization [10 min]

**Status**: Dunno → Know

**Learning Objectives**:
- Fitting retrieved docs in 4K/8K/32K token limits
- Strategies: Truncation, compression, summarization

**Interview Answer Template**:
"Context window optimization is critical when retrieved docs exceed the model's limit. Strategies:
1. **Smart truncation** - Keep first/last paragraphs (lost-in-middle problem)
2. **Compression** - Tools like LongLLMLingua remove redundant tokens
3. **Summarization** - Summarize each doc to 50-100 tokens before concatenating
4. **Reranking** - Use fewer but higher-quality docs (top-3 instead of top-10)

I'd measure trade-off: Retrieval coverage vs context window utilization."

---

### Afternoon Block: Final Consolidation (60 min)

**Review all remaining "Unsure" high-priority topics** (60 min):

1. **FAISS & vector DB basics** [10 min]
   - Index types: Flat (exact, slow), IVF (clustering + quantization), HNSW (graph-based)
   - Trade-off: Accuracy vs speed vs memory

2. **Sentence-BERT embeddings** [8 min]
   - Siamese network: Twin BERT encoders with shared weights
   - Training: Contrastive loss or triplet loss on sentence pairs

3. **HyDE (Hypothetical Document Embeddings)** [8 min]
   - Generate hypothetical answer to the question, embed THAT instead of query
   - Why: Answers are semantically closer to documents than questions

4. **Query rewriting** [5 min]
   - Use LLM to reformulate query before retrieval
   - Example: "What's the capital of the country where the Eiffel Tower is?" → "capital of France"

5. **Query expansion** [5 min]
   - Add synonyms or related terms to query
   - Example: "ML engineer" → "ML engineer OR machine learning engineer OR data scientist"

6. **Query intent classification** [5 min]
   - Classify: Factual, navigational, transactional, conversational
   - Route to different retrieval strategies

7. **Query understanding with LLMs** [5 min]
   - Extract: Entities, constraints, temporal info
   - Example: "Best ML papers in 2023" → entities=[ML], temporal=[2023], intent=[list]

8. **Fixed-size & semantic chunking** [5 min]
   - Fixed: 512 tokens with 50 token overlap
   - Semantic: Split on paragraph/section boundaries

9. **Chunk overlap strategies** [5 min]
   - 10-20% overlap prevents splitting mid-concept
   - Example: 512 token chunks with 100 token overlap

10. **Self-RAG / Corrective RAG** [4 min]
    - Self-RAG: Model decides when to retrieve
    - Corrective RAG: Verify retrieval quality, re-retrieve if poor

**For each topic**: Write 2-3 sentence interview answer in your own words.

---

## Post-Study Actions

### 1. Update Topic Check File
- Mark all studied topics as "Know" (22 high-priority topics)
- Mark consolidated "Unsure" as "Know" (9 topics)
- Recalculate scores

### 2. Final Readiness Assessment

**Expected Scores**:

| Metric | Baseline | Target | Projected |
|--------|----------|--------|-----------|
| High-priority (28 topics) | 23.2% | 70-75% | **82%** ✅ |
| Overall (86 topics, unweighted) | 20.3% | N/A | **36-40%** |
| Overall (weighted 2.5×) | 21.3% | 55-60% | **55-58%** ✅ |

### 3. Create Quick Reference Sheet

Create `references/Week4-Advanced-RAG-Quick-Reference.md` with:
- RRF formula (must memorize!)
- When dense fails, when sparse fails
- FiD vs Long Context trade-off table
- Retrieval metrics formulas (Recall, Precision, MRR, NDCG)
- 2025 trends: GraphRAG, Agentic RAG, RAFT
- RAG evaluation triad: Faithfulness, Answer Relevance, Context Quality

### 4. Knowledge Check (15 min)

**10 questions**:
- 7 from Days 1-3 new content (70%): RRF, SPLADE, FiD, GraphRAG, RAFT, retrieval metrics, end-to-end metrics
- 3 from previous days review (30%): Selected from `data/knowledge-schedule.md` where Next Review <= Current Date

**Scoring**:
- ✅ 100%: Correct, clear explanation
- 🟡 75%: Mostly correct, minor gaps
- 🟠 50%: Partial understanding
- ❌ 25%: Incorrect or "I don't know"

**Target**: 90%+ overall (Week 4 high-priority mastery)

### 5. Update Progress Documents

Update in this order:
1. `ML-Interview-Prep-Plan.md` - Mark Week 4 Days 1-2 complete, update readiness %
2. `00-CONVERSATION-SUMMARY.md` - Add Day 22-24 progress section
3. `Daily-Knowledge-Check-Protocol.md` - Add knowledge check results to summary table
4. `Daily-Knowledge-Check-Details.md` - Add full Q&A details
5. `data/knowledge-schedule.md` - Update SM-2 schedule for studied topics

---

## Success Criteria Checklist

**Core Knowledge** (Must achieve):
- [x] Can write RRF formula from memory: Score = 1/(k + rank)
- [x] Can explain when dense fails (acronyms, part numbers) and when sparse fails (synonyms, semantic)
- [ ] Can discuss FiD vs Long Context trade-offs (cost, latency, complexity, accuracy)
- [ ] Can name 3+ 2025 RAG trends: GraphRAG, Agentic RAG, RAFT, Long RAG
- [ ] Can calculate Recall@K, Precision@K, NDCG by hand
- [ ] Can explain RAG evaluation triad: Faithfulness, Answer Relevance, Context Quality

**Readiness Metrics**:
- [ ] High-priority: ≥82% (22/28 topics at "Know" level)
- [ ] Overall weighted: ≥55% (71/128 weighted points)
- [ ] Knowledge check: ≥90% score

**Interview Simulation** (Self-test):
- [x] "Explain hybrid retrieval and RRF" - 2 min answer
- [ ] "What's new in RAG in 2025?" - List 3 trends with brief explanation
- [ ] "How do you evaluate a RAG system?" - Mention retrieval + generation metrics
- [ ] "FiD vs Long Context - which would you choose and why?" - Consider cost, latency, accuracy

---

## Backup Resources

### If Stuck on a Topic
- LangChain blog: "RAG from Scratch" series (videos + code)
- Pinecone Learning Center: Search all RAG topics
- LlamaIndex docs: Production RAG patterns
- Paper: "Retrieval-Augmented Generation for Large Language Models: A Survey" (2023)

### If Running Low on Time
**Minimum viable study plan (Day 1 + Day 2 only, 4 hours)**:
- Day 1: RRF (must know formula!) + Cross-encoder reranking + MMR
- Day 2: FiD (Gap Q177) + GraphRAG + RAFT + Retrieval metrics formulas

This covers the 8 most critical high-priority topics and closes the Gap Q177 (FiD 0% → 80%+).

---

## Notes

**Timeline Flexibility**:
- If Day 1-2 feels sufficient, skip Day 3 (you'll hit 70-75% high-priority target)
- If Day 1-2 feels rushed, extend to full Day 3 for consolidation
- Knowledge check at end of Day 2 will reveal whether Day 3 is needed

**Study Style**:
- Paper reading: Sections 1-3 (intro + method), skim 4-5 (experiments)
- Practice exercises: Do 1-2 examples by hand to internalize formulas
- Interview answers: Write in your own words, not memorized from this doc

**Comparison to Week 2**:
- Week 2: 82% Dunno → 85%+ in 2-3 hours (LLM Systems)
- Week 4: 60.5% Dunno → projected 82% high-priority in 4-6 hours (Advanced RAG)
- Both achieved: Strong interview readiness on core topics + spillover learning on adjacent topics
