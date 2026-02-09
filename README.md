# ML Engineering Fundamentals

A 12-week study plan for ML engineering interview preparation through systematic gap analysis, targeted learning, and implementation practice.

**Approach**: Pre-study gap assessment → Targeted learning → Implementation from scratch → Spaced repetition (SM-2) → System design practice

---

## Progress Tracking

| Week | Focus | Study Hours | Days | Key Achievement |
|------|-------|-------------|------|-----------------|
| **Week 1** | Foundations + Gap Analysis | ~12h | 7 days | 189-question assessment, algorithm implementations |
| **Week 2** | LLM Systems + Statistics | ~15h | 7 days | Closed LLM systems gap (30% → 85%) |
| **Week 3** | PyTorch + ML Infra + System Design | ~18h | 7 days | System design practice, PyTorch from scratch |
| **Week 4** | Advanced RAG Theory | ~12h | 7 days | 99.2% RAG mastery, paper deep dives |
| **Week 5** | RAG Project Implementation | ~15h | 6 days | Production RAG system with evaluation & Docker |
| **Week 6** | Neural Network from Scratch | ~10h | 5 days (paused, resumed Feb 2026) | Forward/backward pass, gradient checking, SGD/Adam optimizers |

**Total Time Investment**: ~82 hours across 39 days
**Phase 1-3 Complete** (Weeks 1-6): Two portfolio projects + interview-ready fundamentals

## Portfolio Projects

- [**RAG Q&A System**](https://github.com/hongsly/rag-qa-system) - Production-ready hybrid retrieval system with RAGAS evaluation, error analysis, and Docker deployment. Key finding: Dense embeddings underperform on small technical corpus (3.7× worse retrieval failures vs sparse).

- **Neural Network from Scratch** (`projects/neural-network/`) - PyTorch-style NN framework in pure NumPy with verified backpropagation, gradient checking (rel. error < 1e-7), SGD/Adam optimizers. Key insight: Adam converges 35× better than SGD on XOR (loss 0.000032 vs 0.00093) despite 10× smaller learning rate.

## Strengths

- ✅ ML Fundamentals: Loss functions, optimization, regularization
- ✅ Classical ML: Regression, classification, clustering, ensembles
- ✅ Deep Learning: Neural networks, backprop, architectures
- ✅ NLP/Transformers: Attention, BERT, GPT, tokenization
- ✅ LLM Systems: Distributed training, parallelism, inference optimization
- ✅ RAG Systems: Hybrid retrieval (BM25 + dense), RAGAS evaluation, production deployment

## Tools & Technologies

* **ML Frameworks**: PyTorch, TensorFlow, scikit-learn
* **Languages**: Python, SQL
* **Data**: NumPy, Pandas, Jupyter
* **Infrastructure**: Docker, Kubernetes, Airflow, Kafka
* **Distributed Training**: Megatron-LM, DeepSpeed (ZeRO), FSDP
* **LLM Serving**: vLLM, Ray Serve
* **RAG Stack**:
  - Retrieval: Dense (DPR, Sentence-BERT), Sparse (BM25, SPLADE), Hybrid (RRF)
  - Reranking: Cross-encoder, ColBERT
  - Vector stores: FAISS
  - Evaluation: RAGAS (answer correctness, faithfulness, context recall)
  - Deployment: Streamlit, Docker

## Repository Structure

```
.
├── ML-Interview-Prep-Plan.md           # 12-week study plan
├── Daily-Knowledge-Check-Protocol.md   # SM-2 implementation
├── notebooks/                          # Jupyter implementations
├── references/                         # Daily notes and reference sheets
├── gap_analysis/                       # Area knowledge maps
└── projects/                           # Portfolio projects
    └── rag-qa-system/                  # ⚠️ Moved to github.com/hongsly/rag-qa-system
```

## Key Learnings

### Study Strategy
1. **Gap analysis first** - Don't study blindly, identify specific gaps
2. **Topic coverage checks** - Comprehensive inventory prevents missing critical subtopics
3. **Prioritize breadth over depth** - 80% understanding of 30 topics > 100% of 10 topics
4. **Spaced repetition works** - SM-2 maintained 95%+ retention on 40+ topics
5. **Implementation matters** - 60min → 15min improvement proves muscle memory

### Technical Insights
1. **LLM Training**: ZeRO Stage 3 enables 7B model training on 4×A100 (80GB each)
2. **Memory Bandwidth**: Single best GPU performance indicator (A100 1.73× faster than V100 ≈ bandwidth ratio)
3. **Inference Optimization**: KV-cache is O(n) vs O(n²), but costs 2×H×L×2×S bytes
4. **System Design**: Dynamic batching = 12-50× cost savings (not just scaling!)
5. **RAG Retrieval**: Cross-encoder 10× slower than bi-encoder, use two-stage pipeline
6. **RAG Evaluation**: Dense embeddings underperform on small corpus (3.7× worse retrieval failures vs sparse)
7. **Testset Generation**: Whole documents needed for quality questions, not isolated chunks (5.5/10 → 8.5/10)

## Papers Read

### Foundational (Reviewed)
- **Attention Is All You Need** (Vaswani et al., 2017)
- **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)

### LLM Systems Optimization
- **Megatron-LM**: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)
- **ZeRO**: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)
- **vLLM**: Efficient Memory Management for Large Language Model Serving (Kwon et al., 2023)

### Advanced RAG
- **RRF**: Reciprocal Rank Fusion (Cormack et al., 2009) - Combined ranking method
- **ColBERT**: Efficient and Effective Passage Search via Contextualized Late Interaction (Khattab & Zaharia, 2020) - *Skimmed*
- **SPLADE**: Sparse Lexical and Expansion Model for First Stage Ranking (Formal et al., 2021) - *Skimmed*
- **SPLADE v2**: Improving SPLADE with Regularization (Formal et al., 2022) - *Skimmed*
- **DPR**: Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020)
- **Sentence-BERT**: Sentence Embeddings using Siamese BERT Networks (Reimers & Gurevych, 2019) - *Skimmed*
- **MMR**: Maximal Marginal Relevance (Carbonell & Goldstein, 1998)
- **Lost in the Middle**: How Language Models Use Long Contexts (Liu et al., 2023)
- **ReAct**: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022) - *Skimmed*
- **RAFT**: Adapting Language Model to Domain Specific RAG (Zhang et al., 2024) - *Skimmed*
- **FiD**: Fusion-in-Decoder (Izacard & Grave, 2020)
- **GraphRAG**: Graph-based Retrieval Augmented Generation (Edge et al., 2024) - *Skimmed*
