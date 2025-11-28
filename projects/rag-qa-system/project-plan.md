# RAG Q&A System - Project Plan

**Created**: 2025-11-22 (Day 26, Week 4 Day 5)
**Timeline**: Option B - Full Project 4 (~12 hours over 2 weeks)
**Data Source**: ArXiv Papers (RAG/LLM domain)

---

## Problem Statement

Build a production-quality RAG system for question-answering over recent ArXiv papers on RAG and LLM techniques. Demonstrate:
- Hybrid retrieval (dense + sparse + RRF fusion)
- Automated evaluation with Ragas framework
- Docker deployment to cloud
- Complete senior MLE portfolio piece

---

## Architecture Overview

```
ArXiv Papers (PDF) → Chunking (500 tokens, 50 overlap)
                           ↓
              Sentence-BERT Embeddings (384-dim)
                           ↓
                    ┌──────┴──────┐
                    ↓             ↓
              FAISS Index      BM25 Index
             (Dense retrieval) (Sparse retrieval)
                    ↓             ↓
                    └──────┬──────┘
                           ↓
                  RRF Fusion (k=60)
               Score = Σ 1/(k + rank_i)
                           ↓
                    Top-K Documents
                           ↓
              GPT-3.5-turbo + Context
                           ↓
                  Generated Answer + Citations
```

---

## Tech Stack Decisions

### Core Components
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
  - Why: Free, fast, 384-dim works well, proven for RAG
  - Alternative considered: OpenAI embeddings (too expensive for portfolio)

- **Dense Retrieval**: FAISS (local index)
  - Why: Fast, battle-tested, good for <100K docs
  - Alternative: Chroma (heavier dependency)

- **Sparse Retrieval**: rank-bm25 library
  - Why: Pure Python, no server needed, sufficient for portfolio
  - Alternative: Elasticsearch (overkill for 20-30 papers)

- **Fusion**: Hand-coded RRF
  - Why: Simple (5 lines), demonstrates understanding
  - Formula: Score = Σ 1/(60 + rank_i)

- **LLM**: OpenAI API (gpt-3.5-turbo)
  - Why: Reliable, fast, cheap ($0.50 for 5K queries)
  - Alternative: Ollama (slower, local hassle for portfolio)

- **Evaluation**: Ragas + manual metrics
  - Ragas: Context precision, recall, faithfulness, answer relevance
  - Manual: Recall@K, MRR, NDCG for retrieval

- **Deployment**: Docker + Streamlit Cloud
  - Why: Free hosting, easy to share, professional
  - Alternative: AWS Lambda (more complex)

### Development Tools
- **Version Control**: Git + GitHub
- **Environment**: Python 3.10+ with venv
- **CI/CD**: GitHub Actions (linting + tests)
- **Monitoring**: Simple logging to file

---

## Data Source: ArXiv Papers

### Target Papers (20-30 papers on RAG/LLMs)

**Search queries on arxiv.org:**
1. "Retrieval Augmented Generation" (2023-2024)
2. "RAG evaluation" OR "RAG metrics"
3. "Hybrid retrieval" OR "dense sparse retrieval"
4. "Query rewriting" OR "Query decomposition"
5. "LLM hallucination" OR "Faithfulness"

**Recommended papers to download** (pick 20-30):
- FiD (Fusion-in-Decoder) - Izacard et al.
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
- Lost in the Middle (Liu et al., 2023)
- RAFT (Gorilla paper, 2024)
- ColBERT: Efficient and Effective Passage Search (Khattab & Zaharia, 2020)
- Ragas: Automated Evaluation of RAG (2023)
- Self-RAG (Asai et al., 2023)
- RAPTOR: Recursive Abstractive Processing (2024)
- GraphRAG (Microsoft, 2024)
- Recent survey papers on RAG (2024)
- Papers on query rewriting/decomposition
- Papers on reranking strategies
- Papers on long-context vs. RAG

**Download strategy**:
- Use arxiv.org search + filter by date (2023-2024)
- Download PDFs to `data/raw/`
- Total size: ~50-100 MB (acceptable)

---

## Implementation Timeline

### Weekend (Light Sessions)

**Day 26 (Sat, Nov 23) - 30 min** ✅
- [x] Create project structure
- [x] Write this project-plan.md
- [x] Decision: Option B confirmed

**Day 27 (Sun, Nov 24) - 30 min**
- [x] Download 20-30 ArXiv papers (PDFs to `data/raw/`)
- [x] Create folder structure:
  ```
  rag-qa-system/
  ├── data/
  │   ├── raw/              # PDFs
  │   ├── processed/        # Chunks (JSON)
  │   └── eval/             # Test questions
  ├── src/
  ├── evaluation/
  ├── tests/
  └── outputs/
  ```
- [x] Create `requirements.txt` stub (list libraries, don't install yet)

### Week 5 (Main Implementation - 12 hours)

**Day 1 (Mon, Nov 24, Week 4 Day 7) - 2 hours**
- [x] `src/data_loader.py`: Parse PDFs, chunk by 500 tokens with 50 overlap
- [x] `src/vector_store.py`: Generate embeddings, build FAISS index, save to disk
- [x] Test: Search for 1 query, verify top-5 results
- [x] Commit: "Add data loading and embedding generation"

**Day 2 (Mon, Nov 25, Week 5 Day 1 / Day 29) - 2.5 hours** ✅ **COMPLETE**
- [x] `src/sparse_retrieval.py`: BM25 with NLTK tokenization
- [x] `src/hybrid_search.py`: RRF fusion (k=60, retrieve 4×k candidates)
- [x] Evaluation framework: Precision@5, MRR
- [x] Test: Compare dense vs BM25 vs hybrid on 2 query sets
  - General NLP queries: Hybrid 40% < Dense 60% (query-corpus mismatch)
  - RAG-focused queries: Hybrid 80% > Dense 67% ✅ (aligned queries)
- [x] Key finding: Query-corpus alignment is critical for BM25 performance
- [x] Folder reorganization: Created `experiments/` for analysis scripts
- [x] Commit: "Add hybrid retrieval with RRF fusion"

**Decision**: Use **Hybrid (RRF)** for production - performs better (80% vs 67%) with RAG-focused queries
**Rationale**: User queries will be RAG-related (e.g., "How does ColBERT work?"), not general NLP
**See**: `references/day29-hybrid-retrieval-findings.md` for details

**Day 3 (Wed, Nov 26, Week 5 Day 2) - 2 hours** ✅ **COMPLETE**
- [x] `src/generator.py`: OpenAI Responses API wrapper (gpt-4o-mini, prompt engineering)
- [x] `src/rag_pipeline.py`: End-to-end pipeline (RagAssistant with 4 modes)
- [x] Create test question set (10 questions in `data/eval/test_questions.json`)
  - 3 simple factual ✅
  - 3 complex reasoning ✅
  - 2 multi-hop ✅
  - 2 negative (not in corpus) ✅
- [x] Smoke test: 5 questions × 4 modes = 20 tests
  - Results: Citations excellent, token usage validated (2700 vs 50)
  - Issue discovered: Negative question handling (retrieval contamination)
- [x] Commit: "Add generation and end-to-end pipeline"

**Key Decision**: Used gpt-4o-mini ($0.15 input / $0.60 output) instead of gpt-3.5-turbo - 3× cheaper + better quality

**Day 4 (Thu, Nov 27, Week 5 Day 3) - 1 hour** ✅ **PLANNING & COST ANALYSIS**
- [x] Add ArXiv metadata to chunks (title, authors, year, URL) - 30 min
- [x] Regenerate chunks with metadata (`scripts/build_index.py`)
- [x] Researched Ragas 0.3.9 API (`generate_with_langchain_docs`, gpt-4o-mini setup)
- [x] Investigated Ollama support (not reliable - missing `agenerate_prompt`)
- [x] **Critical discovery**: Ragas cost explosion ($10-15 for 1500 chunks)
  - Test run: 200 chunks → 200K tokens → $0.70 (SummaryExtractor phase only)
  - Extrapolated: 1500 chunks × 2-3 phases = $10-15 total
- [x] **Solution**: Sampling strategy (250 chunks, 7-8 per paper → $1.25 total)
- [x] Analyzed manual vs Ragas test format differences
- [x] Discussed ground truth requirements for metrics
- [x] **Daily knowledge check**: 94% (A) - Excellent overdue item retention
- [ ] Implementation deferred to Day 5 (evaluation code, run metrics)

**Key Decision**: Sample 250 representative chunks instead of all 1500 → 8× cost savings ($1.25 vs $10-15)

**Day 5 (Fri, Nov 28, Week 5 Day 4) - 3 hours** - **EVALUATION IMPLEMENTATION** (moved from Day 4)
- [ ] Implement sampling: `sample_chunks_by_paper()` (250 chunks, 7-8 per paper) - 15 min
- [ ] Generate 20 Ragas questions from sampled chunks - 30 min (~$0.85)
- [ ] Create `evaluation/evaluate_rag.py`: Ragas metrics (context precision, recall, faithfulness, answer relevance) - 45 min
- [ ] Create `evaluation/evaluate_retrieval.py`: Recall@K, MRR, NDCG - 30 min
- [ ] Run full evaluation on 30 questions (10 manual + 20 Ragas) - 30 min (~$0.40)
- [ ] Create `evaluation/error_analysis.py`: Categorize failures - 15 min
- [ ] Generate evaluation reports and summary
- [ ] Commit: "Add comprehensive Ragas evaluation with sampling"

**Total cost**: ~$1.25 (generation + evaluation)

**Day 6 (Sat, Nov 29, Week 5 Day 5) - 3 hours** - **DEPLOYMENT & DOCUMENTATION** (moved from Day 5)
- [ ] `Dockerfile`: Containerize application
- [ ] `docker-compose.yml`: Multi-service setup (optional)
- [ ] `app.py`: Streamlit UI with citations
- [ ] `.github/workflows/ci.yml`: Basic CI (linting, tests)
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces
- [ ] Write comprehensive `README.md`:
  - Problem, architecture, tech stack
  - Results (retrieval metrics, Ragas scores)
  - How to run locally and in Docker
  - Future improvements
- [ ] Push to GitHub
- [ ] Commit: "Add deployment and documentation"

---

## Code Structure (Full Project 4)

```
rag-qa-system/
├── README.md                      # Comprehensive documentation
├── requirements.txt              # All dependencies with versions
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Optional: multi-service
├── .env.example                  # API keys template
├── .gitignore                    # Don't commit data, .env
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions (lint + test)
├── data/
│   ├── raw/                      # 20-30 ArXiv PDFs
│   ├── processed/                # Chunked docs (JSON lines)
│   └── eval/                     # Test question sets
│       └── test_questions.json   # 10 test questions with ground truth
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # PDF parsing, chunking
│   ├── embeddings.py             # Sentence-BERT wrapper
│   ├── vector_store.py           # FAISS operations
│   ├── retriever.py              # Dense + BM25 + RRF fusion
│   ├── generator.py              # OpenAI API wrapper
│   ├── rag_pipeline.py           # End-to-end pipeline
│   └── api.py                    # FastAPI endpoint (optional)
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_retrieval.py    # Recall@K, MRR, NDCG
│   ├── evaluate_rag.py           # Ragas integration
│   ├── error_analysis.py         # Failure categorization
│   └── cost_analysis.py          # API cost tracking
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py          # Unit tests
│   └── test_api.py               # API tests
├── app.py                        # Streamlit UI
├── notebooks/                    # Optional
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_comparison.ipynb
│   └── 03_retrieval_tuning.ipynb
└── outputs/
    ├── eval_results/             # Evaluation metrics and reports
    │   ├── retrieval_metrics.json
    │   ├── ragas_scores.json
    │   └── error_analysis.json
    └── logs/                     # Query logs and monitoring
```

**Estimated lines of code**: ~800 lines (excluding notebooks)

---

## Requirements (Libraries to Install)

```txt
# Core dependencies
sentence-transformers==2.2.2
faiss-cpu==1.7.4
rank-bm25==0.2.2
openai==1.3.0
python-dotenv==1.0.0

# PDF parsing
PyPDF2==3.0.1
pdfplumber==0.10.3

# Evaluation
ragas==0.1.0
datasets==2.14.0

# API and UI
fastapi==0.104.1
uvicorn==0.24.0
streamlit==1.28.0

# Development
pytest==7.4.3
black==23.11.0
flake8==6.1.0

# Utilities
pandas==2.1.3
numpy==1.26.2
tqdm==4.66.1
```

---

## Evaluation Plan

### Test Question Set (10 questions)

**Simple Factual (3 questions)**:
1. "What is Retrieval-Augmented Generation?"
2. "Who proposed the FiD architecture?"
3. "What does RAFT stand for?"

**Complex Reasoning (3 questions)**:
4. "Why does hybrid retrieval (dense + sparse) outperform either approach alone?"
5. "How does ColBERT differ from traditional dense retrieval?"
6. "What are the trade-offs between long-context LLMs and RAG systems?"

**Multi-hop (2 questions)**:
7. "How do GraphRAG and FiD differ in their approach to multi-document reasoning?"
8. "What evaluation metrics are recommended for both retrieval and generation in RAG?"

**Negative (2 questions)**:
9. "What is the capital of France?" (not in corpus)
10. "How do you train a neural network?" (not in corpus)

### Metrics to Track

**Retrieval Metrics** (compare dense, BM25, hybrid):
- Recall@K (K=1,3,5,10): % of questions with correct doc in top-K
- MRR (Mean Reciprocal Rank): 1/rank of first correct doc
- NDCG: Normalized Discounted Cumulative Gain
- Precision@K: % of relevant docs in top-K

**Ragas Metrics** (automated LLM-as-judge):
- Context Precision: Are retrieved contexts relevant to question?
- Context Recall: Does retrieved context contain answer?
- Faithfulness: Is answer grounded in context (no hallucination)?
- Answer Relevance: Does answer address the question?
- Answer Correctness: Semantic similarity with ground truth

**Cost Metrics**:
- Total API calls (embeddings + generation + evaluation)
- Tokens used per query
- Cost per query, cost per 1K queries

### Expected Results

**Retrieval** (based on 99.2% RAG mastery):
- Dense-only: Recall@5 ≈ 70-80%
- BM25-only: Recall@5 ≈ 60-70%
- Hybrid+RRF: Recall@5 ≈ 85-95% ⭐ (best)

**Ragas Scores** (target):
- Context Precision: >0.85
- Context Recall: >0.90
- Faithfulness: >0.90
- Answer Relevance: >0.85

---

## Interview Talking Points

After building this, you can say:

**"I built a hybrid RAG system with automated evaluation for Q&A over recent research papers."**

**Architecture**:
- Problem: Answer questions over 20-30 ArXiv papers on RAG/LLMs with accurate citations
- Retrieval: Dense (sentence-BERT in FAISS) + sparse (BM25) + RRF fusion
- Key insight: Hybrid retrieval captures both semantic similarity (dense) and exact keyword matches (sparse)
- RRF formula: Score = Σ 1/(k + rank_i) with k=60 - principled fusion without learned weights
- Generation: GPT-3.5-turbo with retrieved context + prompt engineering for citations

**Evaluation rigor**:
- Automated: Ragas framework (LLM-as-judge) for context precision, faithfulness, answer relevance
- Manual: Recall@K, MRR, NDCG for retrieval quality
- Error analysis: Categorized failures (retrieval miss, poor generation, hallucination)
- Result: Hybrid+RRF achieved 90% Recall@5 (10-15% better than dense-only)

**Production readiness**:
- Docker containerization for reproducibility
- Deployed to Streamlit Cloud for demo
- CI/CD with GitHub Actions (linting, unit tests)
- Cost analysis: ~$0.02 per query (mostly generation, not retrieval)

**What I'd improve next**:
1. Add reranking with cross-encoder (retrieve 20 → rerank → top 5)
2. Multi-hop query decomposition for complex questions
3. Implement caching for repeated queries (reduce cost)
4. A/B test different embedding models (compare to OpenAI embeddings)

---

## Success Criteria

**Technical**:
- ✅ End-to-end RAG pipeline working
- ✅ Hybrid retrieval (dense + BM25 + RRF) implemented correctly
- ✅ Ragas evaluation framework integrated
- ✅ Recall@5 ≥ 85% on test questions
- ✅ Faithfulness score ≥ 0.90 (no hallucinations)
- ✅ Dockerized and deployed to cloud
- ✅ Clean GitHub repo with comprehensive README

**Portfolio**:
- ✅ Demonstrates senior MLE skills (evaluation rigor, production deployment)
- ✅ Shows RAG mastery (99.2% from Week 4 studies)
- ✅ Interview-ready: Can explain architecture, trade-offs, evaluation in 5 min
- ✅ Meets all updated Project 4 requirements from Project-Ideas.md

**Timeline**:
- ✅ Weekend: Documentation + data prep (~1 hour total)
- ✅ Week 5: Implementation + evaluation + deployment (~12 hours)
- ✅ End of Week 5: Complete portfolio piece on GitHub

---

## Notes

- This is Option B: Full Project 4 with all mandatory components
- Focus on quality over features - better to have working eval than broken reranking
- Document everything for interview storytelling
- If Week 5 runs long, cut optional notebooks - code + README are essential
- Cost estimate: ~$5-10 for OpenAI API during development (acceptable for portfolio)

---

**Next Steps**:
- **Tomorrow (Day 27)**: Download ArXiv papers, create folder structure (30 min)
- **Week 5 Day 1**: Start implementation with data loading (2 hours)

**Status**: Ready to begin! 🚀
