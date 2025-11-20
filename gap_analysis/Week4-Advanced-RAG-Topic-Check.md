# Week 4: Advanced RAG - Topic Coverage Check

**Date**: Day 22 (Week 4, Day 1)
**Purpose**: Comprehensive topic inventory before beginning Advanced RAG study
**Context**: Gap analysis showed RAG at 30% (Q177: FiD 0%, Q178: Internet-augmented 75%, Q179: Hybrid retrieval 30%). Target: 75% interview readiness by end of Week 4.

---

## Instructions

For each subtopic, mark your current knowledge level:
- ✅ **Know**: Can explain confidently in an interview (2-3 min explanation)
- 🟡 **Unsure**: Heard of it, vague understanding, need review
- ❌ **Dunno**: No idea, never learned, or completely forgot

**Scoring**: After completing the assessment, calculate percentages for each category and overall.

---

## Category 1: Retrieval Techniques (11 topics)

**Cross-reference with gap analysis**: Q179 (Hybrid retrieval - 30%)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Dense retrieval basics | x |  | ☐ | Vector embeddings, semantic search, cosine similarity |
| Sparse retrieval (BM25, TF-IDF) | x |  | ☐ | Keyword-based methods, term frequency scoring |
| SPLADE (Learned Sparse) ⭐ | x | ☐ |  | **2025 trend** - Solves vocabulary mismatch, keeps exact-match benefits |
| **Hybrid retrieval (sparse + dense)** ⭐ | ☐ | x | ☐ | **Gap Q179** - Combining keyword + semantic, fusion strategies |
| Reciprocal Rank Fusion (RRF) ⭐ | x | ☐ |  | Score fusion algorithm: 1/(k + rank) - Must know formula! |
| Late interaction models (ColBERT) | x | ☐ |  | Token-level similarity, MaxSim operation |
| DPR (Dense Passage Retrieval) ⭐ | x | ☐ |  | Dual-encoder architecture, contrastive learning |
| Contrastive learning for retrieval | x |  | ☐ | In-batch negatives, hard negative mining |
| Multi-vector retrieval | ☐ | ☐ | x | Multiple embeddings per document |
| Query expansion techniques | ☐ | ☐ | x | Query rewriting, pseudo-relevance feedback |
| Approximate Nearest Neighbor (ANN) | ☐ | ☐ | x | HNSW, IVF, Product Quantization for scale |

**Category 1 Score**: Know: _/11 | Unsure: _/11 | Dunno: _/11

---

## Category 2: Chunking Strategies & Document Processing (10 topics)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Fixed-size chunking ⭐ | x | ☐ | ☐ | Simple 256/512/1024 token chunks |
| Semantic chunking ⭐ | ☐ | x | ☐ | Split by paragraph, sentence boundaries |
| Propositional / Context-aware chunking | ☐ | x | ☐ | LLM-based chunking with document context |
| Recursive chunking | ☐ | x | ☐ | Hierarchical splitting with overlap |
| Chunk overlap strategies ⭐ | ☐ | x | ☐ | 10-20% overlap, sliding windows |
| Document structure preservation | ☐ | x | ☐ | Maintain headers, sections, metadata |
| Chunk size vs retrieval quality | ☐ | x | ☐ | Trade-offs: precision vs context |
| Multi-resolution chunking | ☐ | ☐ | x | Store chunks at multiple granularities |
| Token vs character-based chunking | ☐ | x | ☐ | Alignment with model tokenizer |
| **Complex PDF parsing / OCR** ⭐ | ☐ | ☐ | x | Real-world: tables, columns, unstructured.io, LlamaParse |

**Category 2 Score**: Know: _/10 | Unsure: _/10 | Dunno: _/10

---

## Category 3: Embedding Models & Optimization (9 topics)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Sentence-BERT / Sentence Transformers ⭐ | ☐ | x |  | Siamese networks for sentence embeddings |
| OpenAI embeddings (ada-002, text-embedding-3) | ☐ | ☐ | x | Commercial embedding APIs |
| Domain-specific embedding fine-tuning | x | ☐ |  | Contrastive fine-tuning on domain data |
| Embedding dimension trade-offs | ☐ | x | ☐ | 384 vs 768 vs 1536 dimensions |
| Matryoshka embeddings | ☐ | ☐ | x | Variable-dimension embeddings |
| Multilingual embeddings | ☐ | x | ☐ | Cross-lingual retrieval (mBERT, XLM-R) |
| Embedding compression | ☐ | x |  | Quantization for storage/speed |
| Late interaction vs bi-encoders | x | ☐ |  | ColBERT vs SBERT trade-offs |
| Contextualized embeddings | ☐ | ☐ | x | Query-aware vs static embeddings |

**Category 3 Score**: Know: _/9 | Unsure: _/9 | Dunno: _/9

---

## Category 4: Vector Databases & Storage (7 topics)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Vector database landscape | ☐ | ☐ | x | Pinecone, Weaviate, Qdrant, Milvus, Chroma |
| FAISS (Facebook AI Similarity Search) ⭐ | ☐ | ☐ | x | Open-source ANN library, index types |
| Index types | ☐ | ☐ | x | Flat, IVF, HNSW, PQ trade-offs |
| Metadata filtering | ☐ | ☐ | x | Pre-filter vs post-filter strategies |
| Hybrid search in vector DBs | ☐ | ☐ | x | Combining vector + keyword + filters |
| Sharding & replication | ☐ | x | ☐ | Scaling vector search horizontally |
| Vector DB vs traditional DB | ☐ | ☐ | x | When to use which |

**Category 4 Score**: Know: _/7 | Unsure: _/7 | Dunno: _/7

---

## Category 5: Query Processing (8 topics)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Query rewriting / reformulation ⭐ | ☐ | x | ☐ | LLM-based query enhancement |
| Query decomposition | ☐ | x | ☐ | Multi-hop → sub-questions (related to FiD) |
| Query expansion | ☐ | x | ☐ | Adding synonyms, related terms |
| HyDE (Hypothetical Document Embeddings) ⭐ | ☐ | ☐ | x | Generate hypothetical answer, embed that |
| Step-back prompting | ☐ | ☐ | x | Abstract question before retrieval |
| Query intent classification | ☐ | x | ☐ | Factual vs conversational vs multi-hop |
| Query understanding with LLMs | ☐ | x | ☐ | Extract entities, intent, constraints |
| Multi-query retrieval | ☐ | ☐ | x | Generate multiple query variants |

**Category 5 Score**: Know: _/8 | Unsure: _/8 | Dunno: _/8

---

## Category 6: Reranking Techniques (7 topics)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Cross-encoder reranking ⭐ | x | ☐ |  | BERT-based query-document scoring |
| Two-stage retrieval (retrieve → rerank) | x |  | ☐ | Fast bi-encoder → slow cross-encoder |
| Reranking with LLMs | ☐ | x | ☐ | Use LLM to score relevance |
| Maximal Marginal Relevance (MMR) ⭐ | x | ☐ |  | Diversity-aware reranking |
| Cohere Rerank API | ☐ | ☐ | x | Commercial reranking service |
| Lost-in-the-middle problem ⭐ | x | ☐ |  | Position bias in long contexts |
| Relevance score calibration | ☐ | ☐ | x | Converting scores to probabilities |

**Category 6 Score**: Know: _/7 | Unsure: _/7 | Dunno: _/7

---

## Category 7: Context Management (6 topics)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Context window optimization ⭐ | ☐ | ☐ | x | Fitting retrieved docs in 4K/8K/32K tokens |
| Context compression | ☐ | x | ☐ | Removing redundant information |
| Context ranking & selection | ☐ | ☐ | x | Top-K selection strategies |
| Long-context models (Claude, GPT-4) | ☐ | x | ☐ | 100K+ token windows |
| Context stuffing vs summarization | ☐ | x |  | Trade-offs for long docs |
| Prompt template design | ☐ | x | ☐ | System + context + query structure |

**Category 7 Score**: Know: _/6 | Unsure: _/6 | Dunno: _/6

---

## Category 8: RAG Evaluation (9 topics)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Retrieval metrics ⭐ | ☐ | x | ☐ | Recall@K, Precision@K, MRR, NDCG |
| Generation metrics | ☐ | x | ☐ | BLEU, ROUGE, BERTScore |
| End-to-end RAG metrics ⭐ | ☐ | x | ☐ | Faithfulness, answer relevance |
| RAGAs framework | ☐ | ☐ | x | Automated RAG evaluation |
| Human evaluation protocols | ☐ | x | ☐ | Annotation guidelines, inter-rater reliability |
| Failure mode analysis | ☐ | x | ☐ | No relevant docs, wrong docs, hallucination |
| Latency vs quality trade-offs | ☐ | x | ☐ | P50/P95/P99 latency budgets |
| A/B testing RAG systems | ☐ | x |  | Online evaluation strategies |
| Groundedness / Attribution | ☐ | x | ☐ | Verifying citations, source attribution |

**Category 8 Score**: Know: _/9 | Unsure: _/9 | Dunno: _/9

---

## Category 9: Production RAG Architecture (11 topics)

**Cross-reference with gap analysis**: Q177 (Fusion-in-Decoder - 0%), Q178 (Internet-augmented LLMs - 75%)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Standard RAG pipeline ⭐ | ☐ | x | ☐ | Query → Retrieve → Augment → Generate |
| **Fusion-in-Decoder (FiD)** ⭐ | ☐ | ☐ | x | **Gap Q177 (0%)** - Encode passages independently, decode jointly. **2025 note**: Discuss FiD vs Long Context trade-offs! |
| Self-RAG | ☐ | ☐ | x | Model decides when to retrieve |
| Corrective RAG (CRAG) | ☐ | ☐ | x | Verify retrieval quality, re-retrieve if needed |
| Adaptive RAG | ☐ | ☐ | x | Dynamic retrieval strategy based on query |
| **RAFT (Retrieval Augmented Fine Tuning)** ⭐ | ☐ | ☐ | x | **2025 hot topic** - Fine-tune model to be better at RAG (ignore distractors) |
| **Agentic RAG** ⭐ | ☐ | ☐ | x | **2025 trend** - Autonomous agents, multi-step reasoning |
| **GraphRAG** ⭐ | ☐ | ☐ | x | **2025 trend** - Knowledge graph-based retrieval |
| Streaming RAG | ☐ | ☐ | x | Real-time document updates |
| Multi-modal RAG ⭐ | ☐ | x | ☐ | Text + images + tables |
| RAG with function calling | ☐ | ☐ | x | LLM calls APIs during generation |

**Category 9 Score**: Know: _/11 | Unsure: _/11 | Dunno: _/11

---

## Category 10: Advanced Patterns & Optimization (8 topics)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Multi-hop retrieval ⭐ | ☐ | ☐ | x | Iterative retrieval (retrieve → reason → retrieve) |
| **Long RAG** | ☐ | ☐ | x | **2025 trend** - Handling full documents vs chunks |
| Parent document retrieval ⭐ | ☐ | ☐ | x | Retrieve small chunks, return full context |
| Recursive retrieval | ☐ | ☐ | x | Hierarchical document structure |
| Query routing | ☐ | ☐ | x | Route to different indexes/strategies |
| Caching strategies | ☐ | ☐ | x | Query cache, embedding cache, result cache |
| Prompt compression | ☐ | ☐ | x | LongLLMLingua, AutoCompressors |
| Cold start for new documents | ☐ | ☐ | x | Indexing latency, temporary boosting |

**Category 10 Score**: Know: _/8 | Unsure: _/8 | Dunno: _/8

---

## High-Priority Subset (28 Topics)

These topics are marked with ⭐ above and represent interview-critical knowledge.

### Recommended Study Order (Attack in 4 phases)

**Phase 1: "The Fix" - Hybrid Retrieval Foundations** (Must master first)
1. **Hybrid retrieval (sparse + dense)** - **Gap Q179 (30%)**
2. **Reciprocal Rank Fusion (RRF)** - Must know formula: Score = 1/(k + rank)
3. Sparse retrieval (BM25, TF-IDF) - When dense fails (part numbers, acronyms)
4. Dense retrieval basics - When sparse fails (synonyms, semantic concepts)
5. **SPLADE (Learned Sparse)** - Modern bridge between sparse and dense
6. DPR (Dense Passage Retrieval) - Foundation for dense retrieval

**Phase 2: "The Polish" - Reranking & Retrieval Quality** (Shows practical expertise)
7. **Cross-encoder reranking** - Two-stage retrieval pattern
8. Late interaction models (ColBERT) - Token-level similarity
9. **MMR (Maximal Marginal Relevance)** - Diversity-aware reranking
10. **Lost-in-the-middle problem** - Position bias in long contexts
11. Sentence-BERT embeddings - Bi-encoder foundation
12. FAISS & vector DB basics - Production deployment

**Phase 3: "The System" - Advanced Patterns** (Shows seniority, 2025 relevance)
13. **GraphRAG** - **2025 hot trend** - Knowledge graph-based retrieval
14. **Agentic RAG** - **2025 hot trend** - Autonomous multi-step reasoning
15. **RAFT (Retrieval Augmented Fine Tuning)** - **2025 hot trend** - RAG + fine-tuning hybrid
16. **Fusion-in-Decoder (FiD)** - **Gap Q177 (0%)** + discuss Long Context trade-offs
17. Multi-hop retrieval - Iterative retrieve → reason → retrieve
18. Parent document retrieval - Retrieve small chunks, return full context
19. **Complex PDF parsing / OCR** - Real-world interview question: "How do you handle tables?"
20. Multi-modal RAG - Text + images + tables

**Phase 4: "The Proof" - Evaluation & Production** (Shows you can measure success)
21. **Retrieval metrics** - Recall@K, Precision@K, NDCG - Must know formulas
22. **End-to-end RAG metrics** - Faithfulness, answer relevance (RAGAs)
23. Context window optimization - Fitting docs in 4K/8K/32K tokens
24. Fixed-size & semantic chunking - Foundational preprocessing
25. Chunk overlap strategies - 10-20% overlap, sliding windows
26. Query rewriting / HyDE - Generate hypothetical answer, embed that
27. Standard RAG pipeline - Query → Retrieve → Augment → Generate
28. Self-RAG / Corrective RAG - Model decides when to retrieve

**High-Priority Score**: Know: _/28 | Unsure: _/28 | Dunno: _/28

---

## Overall Summary Scorecard

| Category | Know | Unsure | Dunno | % Know |
|----------|------|--------|-------|--------|
| 1. Retrieval Techniques (11) | _/11 | _/11 | _/11 | ___% |
| 2. Chunking & Doc Processing (10) | _/10 | _/10 | _/10 | ___% |
| 3. Embedding Models (9) | _/9 | _/9 | _/9 | ___% |
| 4. Vector Databases (7) | _/7 | _/7 | _/7 | ___% |
| 5. Query Processing (8) | _/8 | _/8 | _/8 | ___% |
| 6. Reranking (7) | _/7 | _/7 | _/7 | ___% |
| 7. Context Management (6) | _/6 | _/6 | _/6 | ___% |
| 8. RAG Evaluation (9) | _/9 | _/9 | _/9 | ___% |
| 9. Production Architecture (11) | _/11 | _/11 | _/11 | ___% |
| 10. Advanced Patterns (8) | _/8 | _/8 | _/8 | ___% |
| **TOTAL (76 topics)** | **_/76** | **_/76** | **_/76** | **___%** |
| **High-Priority (28 topics)** | **_/28** | **_/28** | **_/28** | **___%** |

**Overall Readiness**: ___% (Target: 75% by end of Week 4)

---

## Gap Analysis Cross-Reference

### From Day 6 Gap Analysis (ML-Theory-Questions.md Q177-179)

| Question | Topic | Pre-Study Score | Notes |
|----------|-------|-----------------|-------|
| Q177 | Fusion-in-Decoder (FiD) | 0% (Dunno) | Critical gap - encode passages independently, decode jointly |
| Q178 | Internet-augmented LLMs | 75% (Mostly correct) | Understand retrieval + generation, need architecture details |
| Q179 | Hybrid retrieval | 30% (Conceptual only) | Know it combines sparse + dense, need RRF/fusion strategies |

**Target Post-Study Scores**: Q177: 80%+, Q178: 90%+, Q179: 85%+

---

## Post-Study Validation

**To be completed after Week 4 Day 1-2 study session**

### Re-Assessment Results

| Category | Pre-Study % Know | Post-Study % Know | Improvement |
|----------|------------------|-------------------|-------------|
| Overall (72 topics) | ___% | ___% | +___% |
| High-Priority (24) | ___% | ___% | +___% |
| Category 1: Retrieval | ___% | ___% | +___% |
| Category 9: Production | ___% | ___% | +___% |

**Gap Closure**:
- Q177 (FiD): 0% → ___%
- Q178 (Internet-augmented): 75% → ___%
- Q179 (Hybrid retrieval): 30% → ___%

**Target Achievement**: ☐ Met 75% overall readiness | ☐ All high-priority topics ≥70%

---

## Recommended Study Resources

### Essential Papers
1. **DPR**: "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)
2. **FiD**: "Leveraging Passage Retrieval with Generative Models for Open Domain QA" (Izacard & Grave, 2021) ← **Gap Q177**
3. **SPLADE**: "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking" (Formal et al., 2021)
4. **ColBERT**: "Contextualized Late Interaction over BERT" (Khattab & Zaharia, 2020)
5. **Self-RAG**: "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2023)
6. **RAFT**: "RAFT: Adapting Language Model to Domain Specific RAG" (Zhang et al., 2024)
7. **GraphRAG**: "From Local to Global: A Graph RAG Approach" (Microsoft, 2024)

### Key Blog Posts / Tutorials
- LangChain RAG from Scratch: Multi-query, RAG-Fusion, CRAG, Self-RAG
- Pinecone Learning Center: Retrieval, reranking, evaluation
- LlamaIndex guides: Production RAG patterns, evaluation frameworks
- RAGAs documentation: Automated evaluation metrics

### Hands-On Practice (Prioritized)
1. **Phase 1 Focus**: Implement hybrid retrieval (BM25 + dense embeddings + RRF formula)
2. **Phase 2 Focus**: Build two-stage retrieval (bi-encoder → cross-encoder reranking)
3. **Phase 3 Focus**: Compare FiD vs Long Context for multi-document QA
4. **Phase 4 Focus**: Benchmark retrieval metrics (Recall@K, NDCG) on your implementation

### Key Interview Answers to Prepare
- **When does dense fail?** Exact part numbers, acronyms, proper nouns, new terms
- **When does sparse fail?** Synonyms, semantic concepts, paraphrasing
- **RRF formula on whiteboard**: Score = 1/(k + rank), typical k=60
- **FiD vs Long Context trade-offs**: FiD cheaper/faster but harder to implement vs Gemini 1.5 Pro 1M token window
- **Real-world PDF problem**: "How do you handle tables?" → unstructured.io, LlamaParse, layout-aware chunking

---

## Notes

**Format Reference**: This topic check follows the structure of `Week2-LLM-Systems-Topic-Check.md` (76 subtopics, 83% post-study achievement)

**Success Pattern from Week 2**: Pre-study assessment (82% dunno) → focused study on weak areas → post-study validation (83% target achieved) → Week 2 LLM Systems: 30% → 85%+

**Expected Outcome**: Similar pattern for Week 4 RAG: 30% baseline → identify critical gaps (especially FiD, Hybrid retrieval, GraphRAG, RAFT, SPLADE) → focused 2-3 hour study following 4-phase priority order → 75%+ interview readiness

**Update vs Original Version**: Added 4 topics based on 2025 trends and practical gaps:
1. SPLADE (learned sparse retrieval) - bridges sparse/dense
2. Propositional/context-aware chunking - LLM-based chunking
3. RAFT (Retrieval Augmented Fine Tuning) - hot 2025 topic
4. Complex PDF parsing / OCR - real-world interview question

Total: 72 → 76 topics, High-Priority: 24 → 28 topics
