# Day 34 Quick Reference: Streamlit UI & Docker Deployment

**Date**: Nov 30, 2025 (Week 5 Day 6)
**Focus**: Testset regeneration, re-evaluation, UI, Docker containerization
**Time**: ~6 hours

---

## Testset Regeneration (Whole Documents)

### Root Cause Fixed
**Day 33 issue**: Generated testset from 500-token chunks → Ragas couldn't build knowledge graph → 46% bad questions

**Day 34 fix**: Use whole documents with PyMuPDFLoader
```python
# scripts/generate_testset.py
def load_and_clean_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path, mode="single")
    full_text = loader.load()[0].page_content
    # Remove references section
    cleaned_text = full_text[:ref_start] if ref_found else full_text
    return Document(page_content=cleaned_text, metadata={"source": pdf_path})
```

### Quality Improvement
| Metric | Before (Day 33) | After (Day 34) | Δ |
|--------|----------------|----------------|---|
| Overall quality | 5.5/10 | 8.5/10 | +55% |
| Unique questions | 54% | 97.5% | +80% |
| From reference sections | 46% | 0% | -100% |

**Final testset**: 32 Ragas + 10 manual = 42 questions total

---

## RAG Evaluation Results (42 Questions)

| Mode | Answer Correctness | Context Recall | Success Rate |
|------|-------------------|----------------|--------------|
| **HYBRID** | **66.9%** ⭐ | 83.3% | 52.4% ⭐ |
| **SPARSE** | 61.3% | **87.3%** ⭐ | 42.9% |
| **DENSE** | 51.9% ❌ | 69.4% ❌ | 23.8% ❌ |
| **NONE** | 43.5% | N/A | 31.0% |

**Key findings**:
- HYBRID best for answer quality (67%)
- SPARSE best for recall (87%)
- DENSE 3.7× worse retrieval failures (26% vs 7%)
- No retrieval = 43% (proves retrieval essential)

---

## Error Analysis Patterns

| Failure Type | HYBRID | SPARSE | DENSE |
|--------------|--------|--------|-------|
| **Success** | 52% ⭐ | 43% | 24% ❌ |
| Retrieval failure | 12% | 7% | **26%** ❌ |
| Generation failure | 10% | 12% | 19% |
| Ranking issue | 7% | 10% | 10% |

**Conclusion**: Dense embeddings are weak link on small corpus

---

## Streamlit UI Features

```python
# app.py - Key components
st.set_page_config(layout="wide")

# Sidebar: Retrieval mode, Top K, Model selection
retrieval_mode = st.sidebar.selectbox(["sparse", "hybrid", "dense"])
top_k = st.sidebar.slider(1, 10, 5)

# Main: Question input, example buttons, answer display
answer, context = assistant.query(
    question, model=model, retrieval_mode=retrieval_mode, top_k=top_k
)
st.markdown(answer)

# Retrieved chunks with scores
for chunk in context:
    st.expander(f"Chunk {i} - {chunk['metadata']['title']}")
```

**Features**:
- Mode selection (sparse/hybrid/dense)
- Model selection (OpenAI + local Ollama)
- Top K configuration
- 5 example questions
- Answer display with source attribution
- Retrieved chunks inspection

---

## Docker Basics

### Dockerfile Structure
```dockerfile
FROM python:3.10-slim           # Base image
WORKDIR /app                    # Working directory
COPY requirements.txt .         # Copy deps first (caching)
RUN pip install -r requirements.txt
COPY . .                        # Copy code
EXPOSE 8501                     # Document port
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0  # Listen on all interfaces
CMD ["streamlit", "run", "app.py"]
```

### Key Concepts
- **0.0.0.0**: Listen on all network interfaces (required for Docker)
- **Layer caching**: Order matters (stable → changing)
- **.dockerignore**: Exclude venv/, __pycache__, *.pdf

### Build & Run
```bash
# Build image
docker build -t rag-qa-system .

# Run container
docker run -d \
  --name rag-qa \
  -p 8501:8501 \
  -e OPENAI_API_KEY="sk-..." \
  rag-qa-system

# Check logs
docker logs -f rag-qa

# Open browser
http://localhost:8501
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  rag-qa-system:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data:ro
      - ./outputs:/app/outputs
```

---

## File Structure Changes

### Created EVAL_OUTPUT_DIR
```
data/eval/              # INPUT (test questions)
├── ragas_testset.jsonl
└── test_questions.json

outputs/eval_results/   # OUTPUT (results)
├── eval_results_*.json
├── response_dataset_*.jsonl
└── error_analysis_summary.json
```

**Benefits**:
- Clean separation of inputs/outputs
- Easier gitignore
- Standard ML project structure

---

## Docker Optimization Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Image size | 4.3 GB | 2.2 GB | -49% (2.1 GB saved) |
| Build time | ~5 min | ~3 min | -40% |

**Optimizations**:
1. `.dockerignore` excludes venv/, __pycache__, raw PDFs
2. Split `requirements-dev.txt` (pytest, black, flake8)
3. Use `python:3.10-slim` (vs full image)
4. `--no-cache-dir` for pip (don't cache downloads)

---

## Interview Talking Points

**Q: How did you deploy your RAG system?**

> "I containerized the system with Docker for reproducible deployment. I also optimized the image from 4.3GB to 2.2GB using .dockerignore to exclude the venv, caches, and raw PDFs. The Streamlit UI lets users compare retrieval modes (sparse/hybrid/dense) with configurable top-K and displays retrieved chunks with similarity scores for transparency."

**Q: What did you learn from the testset regeneration?**

> "I initially generated questions from 500-token chunks, which prevented Ragas from building a knowledge graph—46% of questions came from bibliography sections. I fixed this by using whole documents with PyMuPDFLoader and added regex to strip references. Quality improved from 5.5/10 to 8.5/10. This taught me that testset generation and retrieval chunking have different requirements—Ragas needs document structure for complex questions, while retrieval needs semantic coherence."

**Q: Which retrieval method performed best?**

> "HYBRID had the best answer quality (67% correctness, 52% success rate), but SPARSE had highest recall (87%). DENSE performed worst with 3.7× more retrieval failures (26% vs 7%). The key insight: on a small technical corpus (1,395 chunks), keyword matching (BM25) outperforms embeddings. Dense retrieval shines on larger corpora where semantic similarity matters more than exact term matching."

---

## Key Files Created/Updated

**Created**:
- `app.py` - Complete Streamlit UI
- `Dockerfile` - Container definition
- `docker-compose.yml` - Multi-service orchestration
- `.dockerignore` - Exclude unnecessary files
- `requirements-dev.txt` - Dev dependencies
- `README.md` - Comprehensive project documentation
- `references/Day34-Quick-Reference.md` - This file

**Updated**:
- `requirements.txt` - Fixed NumPy version, upgraded packages
- `src/utils.py` - Added EVAL_OUTPUT_DIR
- `evaluation/evaluate_rag.py` - Write to outputs/
- `experiments/analyze_errors.py` - Read from outputs/
- `data/eval/test_questions.json` - Added reference answers
- `experiments/README.md` - Day 34 documentation

---

## Commands Reference

```bash
# Evaluation
python -m evaluation.evaluate_rag
python -m experiments.analyze_errors

# Docker
docker build -t rag-qa-system .
docker run -d --name rag-qa -p 8501:8501 -e OPENAI_API_KEY="..." rag-qa-system
docker logs -f rag-qa
docker stop rag-qa && docker rm rag-qa

# Docker Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## Next Steps (Week 6)

1. ✅ RAG project complete (90% overall)
2. [ ] Push to GitHub (separate portfolio repo)
3. [ ] Optional: Regenerate testset with GPT-4o-mini (higher quality)
4. [ ] Optional: FastAPI + observability (LangSmith/Langfuse)
5. [ ] Start Phase 3: Portfolio Project 2 or Interview Prep

---

**Status**: Day 34 complete - RAG system production-ready with UI + Docker
