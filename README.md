# ML Engineering Fundamentals

A comprehensive 12-week study plan for mastering ML engineering concepts through systematic learning, gap analysis, and implementation practice.

## Repository Purpose

This repository documents a systematic approach to ML engineering interview preparation, emphasizing:

1. **Pre-study gap assessment** - Identify knowledge gaps before diving into study
2. **Targeted learning** - Focus on high-impact topics with measurable outcomes
3. **Implementation from scratch** - Build muscle memory through coding
4. **Spaced repetition** - SM-2 algorithm for long-term retention
5. **System design practice** - Real-world ML system architecture

## Papers Read

### Foundational (Read 2019)
- **Attention Is All You Need** (Vaswani et al., 2017)
- **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)

### LLM Systems Optimization (Week 2)
- **Megatron-LM**: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019)
- **ZeRO**: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)
- **vLLM**: Efficient Memory Management for Large Language Model Serving (Kwon et al., 2023)

### Advanced RAG (Week 4)
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

## Progress Tracking

| Week | Focus | Study Hours | Days | Key Achievement | 
|------|-------|-------------|------|-----------------|
| **Week 1** | Foundations + Gap Analysis | ~12h | 7 days | 189-question assessment, algorithm implementations |
| **Week 2** | LLM Systems + Statistics | ~15h | 7 days | Closed LLM systems gap  |
| **Week 3** | PyTorch + ML Infra + System Design | ~18h | 7 days | System design practice |
| **Week 4** | Advanced RAG (In Progress) | ~6h | 2 days (so far) | RRF, ColBERT, DPR, Cross-encoder mastery |

**Total Time Investment**: ~51 hours across 23 days


## Strengths

- ✅ ML Fundamentals: Loss functions, optimization, regularization
- ✅ Classical ML: Regression, classification, clustering, ensembles
- ✅ Deep Learning: Neural networks, backprop, architectures
- ✅ NLP/Transformers: Attention, BERT, GPT, tokenization
- ✅ LLM Systems: Distributed training, parallelism, inference optimization

## Tools & Technologies

* **ML Frameworks**: PyTorch, TensorFlow, scikit-learn  
* **Languages**: Python, SQL  
* **Data**: NumPy, Pandas, Jupyter  
* **Infrastructure**: Docker, Kubernetes, Airflow, Kafka  
* **Distributed Training**: Megatron-LM, DeepSpeed (ZeRO), FSDP  
* **LLM Serving**: vLLM, Ray Serve  
* **RAG**: Dense retrieval (DPR, Sentence-BERT), Sparse retrieval (BM25, SPLADE), Hybrid (RRF), Reranking (Cross-encoder, ColBERT)

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
5. **RAG**: Cross-encoder 10× slower than bi-encoder, use two-stage pipeline

## Repository Structure

```
.
├── ML-Interview-Prep-Plan.md           # 12-week study plan
├── Daily-Knowledge-Check-Protocol.md   # SM-2 implementation
├── notebooks/                          # Jupyter implementations
├── references/                         # Daily notes and reference sheets
└── gap_analysis/                       # Area knowledge maps
```

---

**Study Approach**: Systematic gap analysis → Targeted learning → Implementation practice → Spaced repetition  
**Timeline**: 12 weeks (2-3 hours/day, 5-10 hours/week).  
**Created**: October 2024  
**Last Updated**: November 2024  
