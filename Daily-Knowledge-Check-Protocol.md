# Daily Knowledge Check Protocol

**Purpose**: Test knowledge retention using spaced repetition based on forgetting curve
**Frequency**: End of each study day
**Duration**: 10-15 minutes

---

## Protocol

### Timing
- **When**: After completing the day's work (theory + implementation)
- **Before documentation**: Run knowledge check before creating quick reference sheets

### Question Distribution

**70% Today's Content** (Fresh learning):
- Core concepts from today's theory
- Implementation details from today's algorithm
- Key formulas or techniques learned

**30% Previous Content** (Spaced repetition):
- Selected from `data/knowledge-schedule.md` where `Next Review <= Current Date`
- Randomly choose, slightly prioritize items with larger overdue
- SM-2 algorithm automatically determines optimal review timing

### Question Types

1. **Conceptual** (40%):
   - "Explain X"
   - "When would you use X vs Y?"
   - "What are the limitations of X?"

2. **Implementation** (40%):
   - "Walk me through how to implement X"
   - "What's the time complexity and why?"
   - "What edge cases need handling?"

3. **Applied/Scenario** (20%):
   - "Given scenario X, which approach would you use?"
   - "How would you debug this issue?"
   - "What metric would you use for this problem?"

### Scoring

**Per Question**:
- ✅ **100%**: Correct, clear explanation, can elaborate
- 🟡 **75%**: Mostly correct, minor gaps or unclear explanation
- 🟠 **50%**: Partial understanding, significant gaps
- ❌ **25%**: Incorrect or "I don't know"

**Overall Score Interpretation**:
- **90-100%**: Excellent retention, ready to move forward
- **80-89%**: Good retention, minor review needed
- **70-79%**: Adequate, consider reviewing weak areas before moving on
- **<70%**: Significant gaps, should review before proceeding

### Feedback Format

For each answer:
1. ✅/🟡/🟠/❌ Score with brief assessment
2. What was correct
3. What needs improvement (if applicable)
4. Interview-ready answer (if significantly different)

At the end:
- Overall score percentage
- Summary of strong areas
- Summary of areas needing review
- Recommendation (proceed / review specific topics / review day)
- **Update `data/knowledge-schedule.md`** with SM-2 calculations (EF, interval, next review)

---

## SM-2 Spaced Repetition Workflow

**Implemented**: 2025-11-09 (Day 13)

**File**: `data/knowledge-schedule.md`

**Algorithm**: SuperMemo 2 (SM-2) for optimal review scheduling

### Overview

SM-2 automatically calculates when to review each topic based on:
- **EF (Easiness Factor)**: How easy the topic is for you (1.3-2.5+, default 2.5)
- **I (Interval)**: Days until next review
- **n**: Number of successful consecutive reviews
- **Performance history**: Your actual scores over time

### Daily Routine

#### Before Knowledge Check (2-3 min):

1. **Read schedule file**:
   ```bash
   Read data/knowledge-schedule.md
   ```

2. **Identify review items**:
   - Filter for items where `Next Review <= Current Date`
   - Select 3-5 items with oldest/lowest next_review dates
   - Example: If current date is 2025-11-10:
     ```
     llm_megatron: Next Review = 2025-11-10 ✓ (due today)
     llm_tensor_parallel_comm: Next Review = 2025-11-10 ✓ (due today)
     stats_4.3: Next Review = 2025-11-24 ✗ (not due yet)
     ```

3. **Generate questions (10 total)**:
   - **30% review**: 3 questions from items identified in step 2
   - **70% new**: 7 questions from today's new content
   - Maintain mix of conceptual, implementation, applied questions

#### After Knowledge Check (3-5 min):

For each reviewed item, calculate new values:

**Step 1: Map score to quality**

| Score Range | Quality (q) | SM-2 Rating | Interval Action |
|-------------|-------------|-------------|-----------------|
| ≤60% | 1 | Again (fail) | Reset to n=1, I=1 |
| 61-79% | 2 | Hard | Increase slowly |
| 80-90% | 3 | Good | Normal increase |
| >90% | 4 | Easy | Large increase |
| 100% | 5 | Perfect | Large increase |

**Step 2: Calculate new EF (Easiness Factor)**

```
EF' = EF + (0.1 - (5-q)×(0.08 + (5-q)×0.02))
EF' = max(1.3, EF')  # Floor at 1.3
```

**Step 3: Calculate new interval**

⚠️ **CRITICAL**: Update I **BEFORE** incrementing n (order matters!)

```
If q = 1 (failed):
    I = 1 day
    n = 1
Else if n = 0 (first review):
    I = 1 day
    n = 1
Else if n = 1:
    I = 6 days
    n = 2
Else (n >= 2):
    I = I_previous × EF'
    n = n + 1
```

**Common mistake**: Setting I based on the NEW n value instead of OLD n value.
- ❌ Wrong: n=1→2, then set I=6×EF (uses n=2)
- ✅ Correct: n=1, set I=6, then n=2 (uses n=1 to determine I)

**Step 4: Update schedule**

```
Last Review = Current Date
Next Review = Current Date + I days
Append score to Score History
Write updated row to data/knowledge-schedule.md
```

### Example Calculation

**Scenario**: Reviewed `stats_5.3` (Gambler's ruin) on 2025-11-10

**Old values**: EF=2.5, n=2, I=6, Last Review=2025-11-04

**User score**: 95% → quality=4 (Easy)

**Calculations**:
```
Step 1: quality = 4 (Easy, since 95% > 90%)

Step 2: Calculate new EF
EF' = 2.5 + (0.1 - (5-4)×(0.08 + (5-4)×0.02))
    = 2.5 + (0.1 - 1×(0.08 + 1×0.02))
    = 2.5 + (0.1 - 0.10)
    = 2.5 + 0.0
    = 2.5 ✓

Step 3: Calculate new interval
Since q=4 (passed) and n=2 (current):
    I' = 6 × 2.5 = 15 days  (use OLD n=2 to determine formula)
    n' = 2 + 1 = 3          (increment after calculating I)

Step 4: Update values
Last Review = 2025-11-10
Next Review = 2025-11-10 + 15 = 2025-11-25
Score History = "75,95" (appended 95)
```

**Updated row in knowledge-schedule.md**:
```
| stats_5.3 | Gambler's ruin problem | Stats | 2.5 | 15 | 3 | 2025-11-10 | 2025-11-25 | 75,95 |
```

### Adding New Items Daily

**After each study and knowledge check session**, add newly studied topics to the schedule:

1. Identify 3-5 key concepts from today's study
  * Note: DO NOT lazily add one row per knowledge check question! Consider how to best organize the topics studied today. 
2. Create topic_id (format: `section_subtopic`, e.g., `llm_flash_attention`)
3. Consider the knowledge check as the first review and initialize with:
   ```
   EF = 2.5  # 2.5 is default; adjust according to initial knowledge check result
   n = 1 (not reviewed yet)
   I = 1
   Next Review: tomorrow
   Score History: knowledge check score
   ```
4. Add row to `data/knowledge-schedule.md`

### Special Cases

**Failure (score <=60%)**:
```
n = 1 (reset)
I = 1 day (review tomorrow)
EF decreases (formula applies)
```

**Perfect streak (multiple 100% scores)**:
```
EF gradually increases above 2.5
Intervals grow: 1→6→15→38→95 days (roughly 2.5× each time)
```

**Overdue review**:
- If you miss a review date, just review when you can
- Calculate EF/interval normally based on performance
- SM-2 is forgiving of delays

### Tips

1. **Show your work**: Always calculate EF/I explicitly, don't skip steps
2. **Verify calculations**: User can spot-check math if uncertain
3. **Be consistent**: Use same quality mapping every time
4. **Don't game it**: Rate honestly - SM-2 works best with accurate feedback
5. **Track patterns**: If same item fails multiple times, need deeper study
6. **Add new items daily**: Keep schedule updated with newly studied concepts

---

## Benefits

1. **Spaced repetition**: Reviews content at optimal intervals before forgetting
2. **Early gap detection**: Identifies weak areas before they become problems
3. **Interview simulation**: Mimics interview question patterns
4. **Confidence building**: Proves retention and improvement over time
5. **Adjustment signal**: Low scores indicate need to slow down or review
6. **Automated scheduling**: SM-2 algorithm handles review timing optimization
7. **Adaptive learning**: EF values adjust based on actual performance history (easier items → longer intervals, harder items → more frequent review)

---

## Integration with Study Plan

### Daily Routine:
```
1. Theory refresh (if applicable)
2. Implementation practice
3. ✅ KNOWLEDGE CHECK (10-15 min) ← Includes SM-2 schedule updates
4. Create quick reference sheet
5. Update progress files
```

### Weekly Review:
- Compare knowledge check scores across week
- Identify persistent weak areas
- Adjust study focus for next week

---

## Sample Knowledge Check (Day 4 Example)

### Today's Content (70% - Regularization, Regression Metrics, K-Means):

**Q1**: Explain when to use L1 vs L2 regularization.

**Q2**: Why is Elastic Net better for correlated variables?

**Q3**: What are R²'s main limitations and how do you address them?

**Q4**: Walk me through the K-Means algorithm steps.

**Q5**: Why are K-Means results unstable across runs?

**Q6**: What is inertia and how do you compute it?

**Q7**: Implement the K-Means assignment step in one line of NumPy.

### Previous Content (30% - Days 1-3):

**Q8** (Day 3): Explain bias correction in Adam optimizer. [1 day ago]

**Q9** (Day 3): When should you use Precision-Recall instead of AUC-ROC? [1 day ago]

**Q10** (Day 2): What's the gradient formula for logistic regression? [2 days ago]

**Total**: 10 questions, ~10-15 min

---

## Tracking Progress

### Knowledge Check Log:

| Day | Date | Overall Score | Strong Areas | Weak Areas | Action |
|-----|------|---------------|--------------|------------|--------|
| 1 | 2025-10-27 | N/A | - | Assessment day | - |
| 2 | 2025-10-29 | N/A | - | - | - |
| 3 | 2025-10-30 | 88% (B+) | Implementation, NumPy, bias correction | Explaining "why" (Adam, Precision-Recall) | Polish reasoning |
| 4 | 2025-10-31 | 95% (A) | Regularization, metrics, K-Means algorithm, retention | Inertia code formula (70%), K-Means "keep best" detail | Review inertia calculation |
| 5 | 2025-11-01 | **98.5% (A+)** | Boosting (perfect), parametric/non-parametric, **inertia resolved (95%)** | None identified | Continue momentum |
| 6 | 2025-11-03 | 66.5% | Gap analysis complete (189 questions) | LLM systems (30%), statistics (65%), RAG (30%) | Week 2 focus areas identified |
| 7 | 2025-11-04 | N/A | Topic coverage check (82% dunno) | - | Week 2 Day 1 begins |
| 8 | 2025-11-04 | **80.8% (B+)** | Megatron-LM basics (77%), Review retention (100%) | ZeRO Stage 3 (0%), ZeRO memory reductions (40%), Tensor parallel comm (70%) | Read ZeRO Section 5, reinforce Stage 3 |
| 9 | 2025-11-05 | **98% (A+)** | Backward pass (97.9%), ZeRO Stage 3 resolved (95%), f/g operators (100%), Review (95%) | Minor terminology precision | Continue Day 3-4 topics |
| 10 | 2025-11-06 | **99.0% (A+)** | Hardware bottlenecks (99.3%), Communication scaling (perfect), Review (98.3%), **User caught 3 errors!** | None - all concepts strong | Day 4: Inference optimization |
| 11 | 2025-11-07 | **97.5% (A+)** | All 6 inference techniques (97.9%), Memory formulas (100%), Trade-off analysis (perfect), Review (96.7%) | Speculative decoding batch reasoning (75%, clarified) | Continue Week 2 momentum |
| 12 | 2025-11-08 | **99.6% (A+)** | **Perfect calculations (100%)** - Parameters (GPT-2/3), Memory (7B/10B models), Batch size optimization, Chinchilla law, Review (97.7%) | None - minor rounding in KV-cache (820 vs 800 KB) | **Week 2 LLM Systems COMPLETE: 83% readiness** |
| 13 | 2025-11-09 | **99.5% (A+)** | **Regression diagnostics (99.3%)** - DW, BP, SW, VIF tests perfect, Covariance vs correlation (100%), Review (100%) | None - all topics >95%, caught error in Q6 (impossible correlation value) | Statistics Day 1: Diagnostics mastered |
| 14 | 2025-11-10 | **86.5% (B+/A-)** | MLE derivations (100%, 100%, 85%), Chi-square test (100%), Normal dist calculations (90%), Covariance review (100%) | T-test assumptions (70%), DW interpretation (75%), AUC for imbalanced data (70%) | Statistics Day 2: Strong fundamentals, minor gaps in test assumptions |
| 15 | 2025-11-11 | **92.0% (A-)** | Regularization/priors (90%), A/B testing (100%, 95%), Distributions (75%, 90%, 100%), CLT/LLN (70%), Review perfect (100%, 100%, 100%) | Binomial variance formula (75%), CLT concrete examples (70%) | Statistics Day 3: Strong fundamentals, perfect review retention |
| 16 | 2025-11-12 | **92.5% (A-)** | PyTorch basics (BCELoss 100%, training loop 95%, no_grad 95%, parameters 95%, backward 90%), Review excellent (KNN 95%, Boosting 100%, MLE 100%) | Training loop order (85% - both work), FSDP sync (70%), FSDP prefetch (60%) | PyTorch Day 1: Strong fundamentals, FSDP concepts need exposure |
| 17 | 2025-11-13 | **95.1% (A)** | Kafka (96.25% - consumer groups, ISR, acks), Feature Stores (93.3% - online/offline, point-in-time 100%), Review excellent (95% - parametric/non-parametric 100%, KV-cache 95%, quantization 85%) | Training-serving skew elaboration (85%), GPTQ vs AWQ usage (85% - research-level) | ML Infrastructure Day 1: Excellent first-day absorption, point-in-time correctness perfect |
| 18 | 2025-11-14 | **95.5% (A+)** | Airflow (97.5% - executors 100%, catchup 100%, idempotency 95%), Feature Store transformations (98.3% - batch vs streaming perfect), Review mixed (90% - Kafka 100%, Feature Store 100%, backward pass 70%) | Backward pass all-reduce locations (70% - concatenate vs sum confusion) | ML Infrastructure Day 2: Outstanding new content (97.9%), one conceptual gap in review |
| 19 | 2025-11-15 | **89.5% (B+/A-)** | Docker (image/container 95%, build/run 95%), Kubernetes (HPA 95%, resources 90%), Review perfect (regression 95%, bandwidth 100%, PyTorch 100%) | Multi-stage builds ML use case (70%), K8s Deployment definition (75%), GPU specifics (80%) | ML Infrastructure Day 3: Strong fundamentals (85.7%), perfect review retention (98.3%) |
| 20 | 2025-11-16 | **91% (A-/A)** | Data pipelines (100%), Bias handling (100%), A/B testing (100%), Two-tower (90%), Throughput calc (85% - revised), Review excellent (Airflow 100%, continuous batching 100%, VIF 85%) | GPU scaling calculation (50% - missed given numbers) | System Design Day 1: Mock interview 78/100 (B+), perfect conceptual understanding, correctly understood 1K predictions = 1 request |
| 21 | 2025-11-17 | **97.0% (A+)** | Cost analysis mastery, Feature engineering (20 features), Dynamic batching, Sliding windows, Manual review volumes, Class imbalance strategies, Review excellent (Kafka 100%, Docker 100%, HPA 100%) | Dynamic batching as first optimization (75% - clarified) | Week 3 complete: System design 85-90% ready |
| 22 | 2025-11-18 | **96.0% (A+)** | RAG fundamentals (98.9% - RRF, SPLADE, DPR, reranking, ColBERT, MMR), Review retention (88.3% - MLE 90%, Covariance 75%, Throughput 100%) | RRF ranking interpretation (90%), Covariance formulas missing (75%) | Week 4 Day 1 complete: Advanced RAG Day 1 mastery, 11 topics studied (7 new + 4 consolidated) |
| 23 | 2025-11-19 | **96.3% (A+)** | Day 2 Advanced RAG (98.6% - FiD 100%, GraphRAG 100%, RAFT 100%, Agentic 90%, Multi-hop 100%, Parent doc 100%, PDF parsing 100%), Review retention (91.7% - Breusch-Pagan 75%, f/g operators 100%, ColBERT 100%) | Breusch-Pagan test procedure (75% - missing method), Agentic RAG cost comparison (90% - implied but not stated) | Week 4 Day 2 complete: 7 new topics + 3 consolidations, Gap Q177 (FiD) closed 0%→100% |
| 26 | 2025-11-22 | **80.0% (B/B+)** | Review-only day (10 overdue topics - Stats fundamentals, LLM systems, RAG), Stats distributions (100%), Regularization priors (100%), Communication patterns (100%), SPLADE (100%), Speculative decoding (100%), CLT/LLN (100%), Shapiro-Wilk (100%) | Chi-square test statistic (75% - missing Σ(O-E)²/E formula), Durbin-Watson interpretation (75% - reversed 0/4 meanings), FSDP stream sync (50% - still weak, 3rd review, EF 1.45) | Day 26 (Week 4 Day 5): Weekend review session, FSDP remains challenging (65%→50%, reset to tomorrow), 7 perfect scores show strong retention |
| 27 | 2025-11-23 | **87.5% (A-/B+)** | Review-only day (10 overdue topics - PyTorch, ML fundamentals, Infrastructure, System design), PyTorch (no_grad 100%, BCELoss 100%), FSDP internals (100% - breakthrough!), Airflow executors (100%), GPU scaling (100%), Regression metrics formulas (all concepts right) | nn.Module super().__init__ (75% - forgot 3rd component), Adam formulas (50% - concepts yes, formulas missing), Docker/K8s examples (75% - concepts clear, details vague) | Day 27 (Week 4 Day 6): **FSDP breakthrough!** 50%→100% (3rd review success, EF 1.55), Adam optimizer needs formula review tomorrow (failed: 50%, reset to n=1) |
| 28 | 2025-11-24 | **97.0% (A+)** | Review day (10 due/overdue topics - LLM, ML, Stats, PyTorch, Kafka), **Adam optimizer mastered!** 50%→100% (formulas perfect), Memory calc/batch size/Chinchilla/activation mem (all 100%), Residual plots (100%), PyTorch training loop (100%), Kafka (100%) | ROC-AUC axes uncertainty (80% - FPR/TPR right, axis placement unsure), Regression assumptions (90% - residuals vs X/Y normality), User corrected 3 scoring errors (Q2/Q5/Q9) | Day 28 (Week 5 Day 1): **RAG project implementation** - 1541 chunks, FAISS IndexFlatIP, high-quality retrieval. Adam breakthrough 50%→100% (2nd review), 8/10 perfect scores |
| 29 | 2025-11-25 | **96.0% (A+)** | Hybrid retrieval Day 29 (BM25 implementation, RRF formula 100%, query-corpus alignment 100%, hybrid trade-offs 100%, reference noise 100%, evaluation metrics 100%), Review perfect (chunking 100%, A/B testing 100%), FAISS IVF/HNSW gap identified (75%) | FAISS IVF/HNSW (75% - not covered yet, will learn during scaling), tiktoken word-level elaboration (90% - could add more detail) | Day 29 (Week 5 Day 2): **Hybrid retrieval complete** - Query-corpus alignment insight (BM25: 13%→67%), **excellent Recall@K question** (monotonicity understanding), production decision: Hybrid 80% > Dense 67% for RAG queries |
| 30 | 2025-11-26 | **97.0% (A+)** | Generation pipeline Day 30 (prompt engineering 95%, negative handling 95%, token validation 100%, execution trace 95%, hybrid findings 100%), Review perfect (Day 29 retention 98.3%), **Outstanding log investigation** (checked OpenAI logs to find root cause) | Negative question handling (95% - retrieval contamination discovered, need different test query), Minor prompt polish (Q3: could list 2-3 solutions as requested) | Day 30 (Week 5 Day 3): **Generation pipeline complete** - OpenAI Responses API working, citations excellent, smoke test validated (2700 vs 50 token usage), **User investigated logs** (found "capital of France" in retrieved chunk), production-ready pipeline with 4 modes |

**Progress Trend**: Week 2-4 sustained excellence 🚀
- Day 3→5: +10.5% improvement over Week 1
- Day 5→8: Maintained momentum through gap analysis + topic check
- Day 8→9: +17.2% improvement (80.8% → 98%) - Outstanding!
- Day 9→10: +1.0% improvement (98% → 99%) - Sustained mastery level
- Day 10→11: -1.5% (99.0% → 97.5%) - Still A+ range, excellent retention
- Day 11→12: +2.1% improvement (97.5% → 99.6%) - **Perfect calculations**
- Day 12→13: -0.1% (99.6% → 99.5%) - **Sustained mastery level (99%+ range)**
- Day 13→14: -13.0% (99.5% → 86.5%) - Expected dip for new complex material (MLE, hypothesis testing)
- Day 14→15: +5.5% improvement (86.5% → 92.0%) - Recovery to A- range
- Day 15→16: +0.5% improvement (92.0% → 92.5%) - Steady A- range
- Day 16→17: +2.6% improvement (92.5% → 95.1%) - ML Infrastructure day 1
- Day 17→18: +0.4% improvement (95.1% → 95.5%) - Sustained A+ range
- Day 18→19: -6.0% (95.5% → 89.5%) - Expected dip for new infrastructure topics (Docker/K8s)
- Day 19→20: +1.5% (89.5% → 91%) - Recovery to A-/A range (revised after throughput calc correction)
- Day 20→21: +6.0% (91% → 97%) - Outstanding recovery to A+ range
- Day 21→22: -1.0% (97% → 96%) - Minor dip, still A+ range, excellent RAG absorption
- Day 22→23: +0.3% (96% → 96.3%) - Sustained A+ range, exceptional Day 2 content mastery (98.6%)
- Day 23→26: -16.3% (96.3% → 80.0%) - Review-only day (weekend), all previously learned topics, expected variance for review sessions
- Day 26→27: +7.5% (80.0% → 87.5%) - Continued review (10 overdue topics), FSDP breakthrough 50%→100%, Adam formula gap identified
- Day 27→28: +9.5% (87.5% → 97.0%) - Review day + RAG project, **Adam optimizer mastered** 50%→100% (2nd review, perfect formulas), 8/10 perfect scores, user caught 3 scoring errors
- Day 28→29: -1.0% (97.0% → 96.0%) - Minor dip, still A+ range, exceptional hybrid retrieval understanding (96.3% new content)
- Day 29→30: +1.0% (96.0% → 97.0%) - Back to A+ range, excellent generation pipeline absorption, **user investigated OpenAI logs** (root cause analysis)
- **Week 2 Days 1-5 average**: 92% across Days 8-12 (LLM Systems)
- **Week 2 Days 6-7 (Statistics)**: Average 93% (Day 13: 99.5%, Day 14: 86.5%)
- **Week 3 Days 1-5 average**: 93.5% across Days 15-19 (Statistics completion + ML Infrastructure deep dive)
- **Week 3 Days 6-7 (System Design)**: Day 20 91% + 78/100 mock, Day 21 97% - strong system design practice
- **ML Infrastructure (Day 17-18)**: Averaging 95.3% on brand new territory - excellent absorption
- **Day 10 highlight**: User caught 3 approximations/errors (bubble time formula, ranking, TP scaling)
- **Day 11 highlight**: User caught 2 major errors (communication volume per-device, speculative decoding ragged tensor problem)
- **Day 12 highlight**: Perfect calculations, caught blog post imprecision on PP activation memory, clarified gradient memory storage
- **Day 13 highlight**: Perfect diagnostics understanding, caught impossible correlation in test question (r=2)
- **Day 14 highlight**: Perfect MLE derivations (100%), identified σ̂² vs σ̂ distinction, user scored conservatively on ROC/AUC (70%)
- **Day 15 highlight**: Perfect review retention (100% on all 3 overdue items), Megatron 77%→100%, Precision-Recall 88-92%→100%
- **Day 16 highlight**: Removed duplicate topic (llm_tensor_parallel_comm), excellent PyTorch fundamentals (87.9%), perfect review retention (98.3%)
- **Day 17-19 highlight**: ML Infrastructure gap closure 0%→65% in 3 days (Kafka, Feature Stores, Airflow, Docker, K8s)
- **Day 20 highlight**: First system design practice - mock interview 78/100 (B+), perfect data pipeline/bias understanding (100%), **user caught throughput calculation error** (1K predictions = 1 request, not 10K QPS!), critical gap: reading given numbers in problem statements (50%)
- **Day 21 highlight**: Fraud detection system design - 97% (A+), perfect review retention (100%), **user caught 4 major errors in Day 21 reference doc**: messy dynamic batching calcs, missing batch time explanation, manual review threshold inconsistency, WRONG class imbalance math (downsampling + weights DON'T cancel out!)
- **Day 22 highlight**: Advanced RAG Day 1 - 96% (A+), nearly perfect new content (98.9%), excellent consolidation of 4 "unsure" topics, RRF formula mastered, ColBERT storage trade-off understood
- **Day 23 highlight**: Advanced RAG Day 2 - 96.3% (A+), **exceptional new content mastery (98.6%)** - FiD 100% (Gap Q177 closed 0%→100%), GraphRAG 100% (local+global modes), RAFT 100%, all 7 topics interview-ready, **user caught H1 scoring error** (H1 is implicit negation of H0)
- **Day 26 highlight**: Weekend review (10 overdue topics) - 80% (B/B+), **FSDP struggles continue** (3rd review: 65%→50%, EF dropped to 1.45, reset to review tomorrow), 7 perfect scores (Stats, LLM comm patterns, RAG), Communication patterns Day 25 study paid off (75%→100%)
- **Day 27 highlight**: Weekend review continued (10 overdue topics) - 87.5% (A-/B+), **FSDP BREAKTHROUGH!** (4th review: 50%→100%, EF recovered to 1.55), Perfect scores on PyTorch basics (100%, 100%), Airflow (100%), GPU scaling (100%), Adam optimizer failed (50% - formula gap, reset to n=1, due tomorrow)
- **Day 28 highlight**: Week 5 Day 1 (RAG project implementation + review) - 97% (A+), **ADAM OPTIMIZER MASTERED!** (2nd review: 50%→100%, all 5 formulas perfect, EF 2.06), **RAG project complete** (1541 chunks, FAISS IndexFlatIP, high-quality retrieval), 8/10 perfect scores, **user caught 3 scoring errors** (Q2 memory breakdown, Q5 activation "2" coefficient, Q9 training loop order - all valid corrections), 4 LLM topics now at 43-day intervals (100% streak)
- **Day 29 highlight**: Week 5 Day 2 (Hybrid retrieval + evaluation) - 96% (A+), **Exceptional new content mastery (96.3%)** - RRF formula perfect (100%), query-corpus alignment insight (BM25: 13%→67% with aligned queries), hybrid trade-offs (100%), reference noise discovery (15% of corpus), evaluation metrics reasoning (100%), **User asked excellent question** about Recall@K monotonicity (correctly understood Recall@K ≥ Recall@(K-1) always), Perfect review retention (100% on chunking, A/B testing)
- **Day 30 highlight**: Week 5 Day 3 (Generation pipeline + smoke test) - 97% (A+), **Outstanding root cause investigation** - user checked OpenAI logs and found "capital of France" in retrieved chunk (retrieval contamination, not prompt issue), Excellent execution trace (95%), Token usage validation perfect (100%), Prompt engineering solid (95%), **Production-ready citations** (format: `[Li et al., 2023](doc_id)`), Smoke test validated: 2700 vs 50 tokens proves retrieval working
- Review retention: 91.9% average (Day 22: 88.3%, Day 23: 91.7%, Day 26: 80.0%, Day 27: 87.5%, Day 28: 97.0%, Day 29: 100%, Day 30: 98.3%)

---

## Weak Items Tracking

**Purpose**: Track items needing additional review until mastered

**SM-2 Integration**: With SM-2 implemented (Day 13+), weak items are automatically tracked via EF (Easiness Factor):
- **EF < 1.8**: Struggling items (review every 1-3 days until improved)
- **EF 1.8-2.2**: Moderate difficulty (review every 3-8 days)
- **EF > 2.5**: Strong items (review every 12+ days)

Items with low scores (<60%) automatically reset to n=1, I=1 (review tomorrow). No manual tracking needed.

**Historical Weak Items** (pre-SM-2, for reference):

**Resolved Items** (scored 90%+ on retest):
- ✅ **Inertia computation** (Day 4: 70% → Day 5: 95%) - Resolved in 1 day!
- ✅ **ZeRO Stage 3 all-gather pattern** (Day 8: 0% → Day 9: 100%) - Resolved in 1 day!
- ✅ **ZeRO memory reductions** (Day 8: 40% → Day 9: 95%) - Resolved in 1 day! (Corrected scoring: "up to 4×/8×" was correct)
- ✅ **Tensor parallel all-reduce locations** (Day 8: 70% → Day 9: 100%) - Resolved in 1 day!
- ✅ **FSDP stream sync + prefetching** (Day 16: 65% → Day 26: 50% → Day 27: 100%) - Resolved after 4 reviews! (EF 1.45 → 1.55)
- ✅ **Adam optimizer bias correction formulas** (Day 27: 50% → Day 28: 100%) - Resolved in 1 day! All 5 formulas perfect (EF 1.96 → 2.06)
