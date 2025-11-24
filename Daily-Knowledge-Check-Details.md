# Daily Knowledge Check Details

## Day 4 Detailed Results (2025-10-31)

**Content Tested**:
- 70% Day 4: Regularization (L1/L2/Elastic Net), Regression metrics (MAE/RMSE/R²), K-Means
- 30% Review: Day 3 (Adam, Q-K-V), Day 2 (LogReg gradient)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | L1 vs L2 regularization | 4 | 100% ✅ | Perfect - use cases clear |
| Q2 | Elastic Net correlated vars | 4 | 95% ✅ | Excellent - minor wording |
| Q3 | R² limitation & Adjusted R² | 4 | 100% ✅ | Perfect - formula + reasoning |
| Q4 | MAE vs RMSE | 4 | 100% ✅ | Perfect - outliers + smoothness |
| Q5 | K-Means algorithm | 4 | 100% ✅ | Perfect - clear steps |
| Q6 | K-Means instability | 4 | 95% ✅ | Good - minor: "keep best" not "average" |
| Q7 | Inertia computation | 4 | 70% 🟡 | Concept OK, code wrong (all distances vs assigned) |
| Q8 | Bias correction (review) | 3 | 100% ✅ | Perfect retention from Day 3 |
| Q9 | Q-K-V attention (review) | 3 | 90% ✅ | Good - K definition slightly vague |
| Q10 | LogReg gradient (review) | 2 | 100% ✅ | Perfect retention from Day 2 |

**Overall Score**: 95% (A)

**Retention Analysis**:
- Day 4 content (Q1-Q7): 94%
- Day 3 review (Q8-Q9): 95%
- Day 2 review (Q10): 100%
- **Spaced repetition working excellently!**

**Key Insights**:
- Regularization/metrics understanding: Excellent (95-100%)
- K-Means algorithm: Perfect understanding
- K-Means implementation details: One gap (inertia code)
- Previous content retention: Outstanding (95%+)

**Action Items**:
1. 🔴 Review inertia calculation: `np.sum((X - centroids[labels]) ** 2)` vs all pairwise distances
2. Will resurface inertia in Day 7 knowledge check

**Recommendation**: ✅ Excellent performance, proceed to Day 5

---

## Day 5 Detailed Results (2025-11-01)

**Context**: Weekend day, limited time (<1 hour), theory-only session

**Content Tested**:
- 70% Day 5: Boosting (AdaBoost, Gradient Boost), Parametric vs Non-parametric
- 30% Review: Day 4 (L1/L2, inertia), Day 3 (Q-K-V)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | Boosting vs bagging | 5 | 95% ✅ | Great insight despite not directly in video |
| Q2 | Amount of say formula | 5 | 100% ✅ | Perfect formula + explanation |
| Q3 | Weight adjustment | 5 | 100% ✅ | Perfect formulas, both directions |
| Q4 | AdaBoost vs Gradient Boost | 5 | 100% ✅ | Clear distinction |
| Q5 | What trees predict (Gradient Boost) | 5 | 100% ✅ | Perfect: residuals |
| Q6 | Parametric vs non-parametric | 5 | 100% ✅ | Perfect definition + examples |
| Q7 | Is NN parametric? | 5 | 100% ✅ | Perfect reasoning |
| Q8 | L1 vs L2 (review) | 4 | 100% ✅ | Perfect retention |
| Q9 | Inertia (weak item) | 4 | 95% ✅ | **HUGE improvement 70%→95%!** |
| Q10 | Q-K-V (review) | 3 | 95% ✅ | Better K explanation than Day 4 |

**Overall Score**: 98.5% (A+)

**Retention Analysis**:
- Day 5 content (Q1-Q7): 99.3% - Outstanding absorption
- Day 4 review (Q8-Q9): 97.5% - Excellent retention
- Day 3 review (Q10): 95% - Solid retention

**Key Insights**:
- Boosting understanding: Near perfect (99.3%) despite limited time
- Parametric/non-parametric: Complete mastery (100%)
- **Weak item resolution**: Inertia 70%→95% in just 1 day (spectacular!)
- Q-K-V improvement: Better articulation than Day 4
- Learning efficiency: 98.5% score with <1 hour study time

**Improvements from Day 4**:
1. Inertia code: From conceptually wrong to nearly perfect ✅
2. K in Q-K-V: More precise language ("keywords/match against")
3. Overall retention: +3.5% despite busy weekend

**Action Items**:
- None! All items resolved or stable
- Continue current approach

**Recommendation**: ✅ Outstanding performance, ready for Day 6

---

## Day 8 Detailed Results (2025-11-04)

**Context**: Week 2, Day 1 - First day of LLM Systems study (Megatron-LM + ZeRO papers)

**Content Tested**:
- 70% Day 8 (Q1-Q10): Megatron-LM (Sections 1-3), ZeRO (Sections 1-4 superficial)
- 30% Review (Q11-Q13): Week 1 learned content (Days 3-5)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | Model vs data parallelism | 8 | 100% ✅ | Perfect - all key points |
| Q2 | Self-attention splitting | 8 | 80% 🟡 | Core correct, confusion on GEMM multiplier matrix |
| Q3 | MLP column/row-parallel pattern | 8 | 100% ✅ | Excellent - nailed non-linearity issue |
| Q4 | Cross-node degradation | 8 | 80% 🟡 | Correct but brief, needed NVLink vs InfiniBand details |
| Q5 | Tensor-parallel degree = 8 | 8 | 100% ✅ | Correct, good intuition on head divisibility |
| Q6 | All-reduce after attention | 8 | 80% 🟡 | Instinct right, reasoning uncertain |
| Q7 | Largest model trained | 8 | 100% ✅ | Perfect recall (8.3B, TP=8, 512 GPUs) |
| Q8 | ZeRO three stages | 8 | 40% 🟡 | Looked up answer, correct but needs reinforcement |
| Q9 | DP memory inefficiency | 8 | 100% ✅ | Perfect - model states replicated, computation parallel |
| Q10 | Pos+g+p parameter communication | 8 | 0% ❌ | Haven't read Section 5 yet (expected gap) |
| Q11 | Gradient Boosting mechanics (review) | 5 | 100% ✅ | Perfect retention from Day 5 |
| Q12 | L1 vs L2 use cases (review) | 4 | 100% ✅ | Perfect retention from Day 4 |
| Q13 | Adam bias correction (review) | 3 | 100% ✅ | Perfect retention from Day 3 |

**Overall Score**: 80.8% (10.5/13)

**Retention Analysis**:
- Day 8 content (Q1-Q10): 6.8/10 = **68%** - Solid first pass on new material
  - Megatron-LM (Q1-Q7): 5.4/7 = **77%** - Interview-ready for basics
  - ZeRO (Q8-Q10): 1.4/3 = **47%** - Expected gap, Section 5 not read yet
- Review content (Q11-Q13): 3/3 = **100%** - Outstanding Week 1 retention
- **Spaced repetition working perfectly** - 100% retention across 3-5 days

**Key Insights**:
- **Strong Week 1 retention**: Perfect scores on all review questions (Day 3→8: 5 days, Day 4→8: 4 days, Day 5→8: 3 days)
- **Megatron-LM fundamentals solid**: 77% on first read, can explain tensor parallelism, model/data parallelism, MLP patterns
- **ZeRO Stage 1-2 understood**: 100% on DP memory efficiency reasoning (Q9)
- **Expected gaps**: ZeRO Stage 3 (Q10: 0%), ZeRO formulas from memory (Q8: 40%)
- **Communication details need depth**: Q2, Q4, Q6 scored 80% - understand concepts but lack precision on NVLink speeds, all-reduce locations, W_O matrix dimensions

**Improvements Needed**:
1. **ZeRO Stage 3 mechanics** (Q10: 0%) - Read Section 5 tomorrow, focus on all-gather pattern
2. **ZeRO memory reductions** (Q8: 40%) - Redraw Figure 1 from memory without notes
3. **Tensor parallel communication** (Q2, Q6: 80%) - Diagram all-reduce locations in attention + MLP
4. **Cross-node details** (Q4: 80%) - Memorize: NVLink 600 GB/s, InfiniBand 12.5 GB/s (48× slower)

**Clarification Resolved During Session**:
- **W_O projection matrix**: Originally confused about "column-parallel" - corrected to **row-wise split**
  - W_O1: `[512, 1024]` (top 512 rows), W_O2: `[512, 1024]` (bottom 512 rows)
  - Y1 and Y2 are both `[batch, seq, 1024]` full-sized partial results
  - All-reduce sums Y1 + Y2 element-wise before dropout/residual

**Action Items**:
1. 🔴 **Day 9 priority**: Read ZeRO Section 5 (pages 10-11) for Q10 gap
2. 🟡 **Day 10 resurface**: ZeRO memory reductions without notes (Q8: 40%)
3. 🟡 **Day 11 resurface**: Tensor parallel all-reduce locations (Q2, Q6: 80%)
4. ✅ **Week 1 spaced repetition**: Continue current schedule (working perfectly)

**Recommendation**: ✅ Excellent performance for Day 1 of new material. 80.8% overall with 100% review retention shows strong learning momentum. Proceed to Day 9 with ZeRO Section 5 focus.

---

## Day 9 Detailed Results (2025-11-05)

**Context**: Week 2, Day 2 - Backward pass deep dive, ZeRO Stage 3, practice exercises

**Content Tested**:
- 70% Day 9: Backward pass (f/g operators), ZeRO Stage 3, 3D parallelism benefits, TP communication volume
- 30% Review: Day 8 weak items (ZeRO stages), Week 1 spaced repetition (Days 3-5)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | f and g operators | 9 | 100% ✅ | Perfect - concise and accurate |
| Q2 | Why all-reduce after W1 not W2 | 9 | 95% ✅ | Excellent reasoning, minor terminology |
| Q3 | ZeRO Stage 3 all-gather | 9 | 100% ✅ | Perfect understanding |
| Q4 | ZeRO Stage 3 overhead | 9 | 90% ✅ | 1.5× correct, minor: reduce-scatter vs scatter-reduce |
| Q5 | PP reduces network congestion | 9 | 100% ✅ | Perfect - staggers DP communication |
| Q6 | TP communication volume | 9 | 100% ✅ | Perfect - activation size not parameters |
| Q7 | 3D parallelism speedup factors | 9 | 100% ✅ | All three factors correct |
| Q8 | ZeRO three stages (weak item) | 8 | 95%→100% ✅ | **Scoring corrected**: "up to 4×/8×" was more precise! |
| Q9 | MP cross-node degradation (review) | 8 | 100% ✅ | Perfect retention from Day 8 |
| Q10 | AdaBoost vs Gradient Boost (review) | 5 | 90% ✅ | Good distinction, minor clarification on stumps |

**Overall Score**: 98% (9.8/10) - A+

**Retention Analysis**:
- Day 9 content (Q1-Q7): 97.9% - Outstanding absorption, mastery level
- Day 8 review (Q8-Q9): 97.5% - Excellent retention, weak item resolved
- Day 5 review (Q10): 90% - Solid retention across 4 days
- **Spaced repetition working perfectly**: 95%+ retention across 1-4 day intervals

**Key Insights**:
- **Backward pass mastery**: 100% on f/g operators, can explain all-reduce locations and reasoning
- **ZeRO Stage 3 resolved**: 0% → 100% in 1 day (Day 8 Q10 gap closed)
- **ZeRO memory formulas corrected**: User's "up to 4×/8×" was MORE accurate than initial grading (depends on Nd)
- **Weak item resolution rate**: 3/3 Day 8 weak items resolved in 1 day (100% success rate)
- **Week 1 retention exceptional**: 90-100% on content from 4-6 days ago

**Improvements from Day 8**:
1. **ZeRO Stage 3**: From 0% to 100% - Complete understanding of all-gather pattern
2. **ZeRO memory reductions**: From 40% to 95% - Corrected understanding of "up to Nd×"
3. **Tensor parallel communication**: From 70% to 100% - All-reduce locations in backward pass clear
4. **Overall**: +17.2% improvement (80.8% → 98%)

**Practice Exercises Completed** (Not formally scored, but logged):
- ✅ Derived backprop gradients for column-parallel layer (95% - missed GeLU derivative)
- ✅ Derived backprop gradients for row-parallel layer (100%)
- ✅ Traced complete MLP block gradient flow (98% - all 4 all-reduces identified)
- ✅ Strong scaling calculation (95% - equations correct, no numerical calc due to time)

**Action Items**:
- None! All Day 8 weak items resolved
- Continue Day 3-4 topics (Parallelism strategies, Inference optimization)

**Recommendation**: ✅ Outstanding performance! 98% demonstrates mastery of Day 1-2 material (Megatron-LM + ZeRO fundamentals). Ready for Day 3-4 topics (FSDP, Hardware, Inference optimization).

---

## Day 10 Detailed Results (2025-11-06)

**Content Tested**:
- 70% Day 10 (Week 2, Day 3): GPU hardware bottlenecks, Roofline model, Communication scaling, FSDP
- 30% Review: Day 8-9 (ZeRO stages, TP backward pass, 3D parallelism)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | Why 500 TFLOPS → 50 TFLOPS? | 10 | 100% ✅ | Perfect - memory-bound explanation |
| Q2 | Arithmetic intensity definition | 10 | 100% ✅ | FLOPs/byte, low = memory-bound |
| Q3 | Roofline model for LLMs | 10 | 100% ✅ | Slanted roof (memory-bound) |
| Q4 | Communication volume ranking | 10 | 95% 🟡 | Correct P vs B×S×H insight, ranking error (my fault) |
| Q5 | Why TP needs NVLink vs InfiniBand | 10 | 100% ✅ | High frequency requires high bandwidth |
| Q6 | FSDP advantage despite 1.5× comm | 10 | 95% 🟡 | N× memory reduction, could add "enables large models" |
| Q7 | PP bubble time formula | 10 | **100% + Bonus** ✅✅ | User caught approximation error! |
| Q8 | ZeRO Stage 2 vs Stage 3 | 8-9 | 100% ✅ | Perfect retention (6 days ago) |
| Q9 | TP backward all-reduce reason | 8-9 | 100% ✅ | Replicated input X requires sum |
| Q10 | 3D parallelism: TP within node | 8-9 | 100% ✅ | Hardware constraint (NVLink) |

**Overall Score**: 99.0% (9.9/10)
- Day 10 content (Q1-Q7): 99.3% (6.95/7)
- Review content (Q8-Q10): 98.3% (2.95/3)

**Exceptional User Contributions**:

1. **Bubble Time Formula Precision** (Q7):
   - User: "I think it is actually (N-1)/(M+N-1)"
   - ✅ **Correct!** My approximation was (N-1)/M (valid when M >> N)
   - Exact: 3/19 = 15.8%, Approximation: 3/16 = 18.75%
   - **Awarded bonus points for catching approximation**

2. **Communication Volume Ranking** (Q4):
   - User noted: "TP > FSDP ranking depends on P vs B×S×H"
   - ✅ **Outstanding insight!** Showed deep understanding
   - My error: Earlier said FSDP > TP, contradicted own calculation (TP: 1228 GB, FSDP: 1050 GB)

3. **TP Scaling Limitation**:
   - User: "I thought it is because standard is 8 GPU per node"
   - ✅ **Correct!** Hardware constraint is primary reason, not just theoretical overhead
   - Updated quick reference to emphasize hardware first

4. **Communication Volume Clarification**:
   - User asked: "Is 2P total data sent and received from all GPUs?"
   - ✅ **Excellent question!** Added Section 11 to quick reference explaining calculation

**Strengths**:
- ✅ Deep understanding of memory-bound vs compute-bound operations
- ✅ Correctly identified P vs B×S×H trade-off in communication volume
- ✅ Exceptional precision (caught 3 approximations/errors)
- ✅ Perfect retention of Week 1 and Week 2 Day 1-2 material (spaced repetition working)
- ✅ Interview-ready explanations (concise, accurate)

**Areas of Excellence**:
- Roofline model understanding (slanted roof = memory-bound)
- GPU hardware constraints (8 GPUs/node with NVLink)
- Communication pattern trade-offs (volume, frequency, bandwidth)
- Formula precision (exact vs approximate)

**Practice Exercises Logged** (not formally scored, tracked for completeness):
- ✅ Sketched 4 communication patterns (DP, FSDP, TP, PP)
- ✅ Big O scaling analysis for each strategy
- ✅ Synthesized trade-offs for interview-ready answers

**Action Items**:
- None! All concepts strong, no weak items
- Continue Day 4 topics (Inference optimization: KV-cache, quantization, continuous batching, speculative decoding)

**Recommendation**: ✅ Exceptional performance! 99% demonstrates mastery + ability to catch instructor errors. User shows deep understanding beyond rote memorization - can identify approximations, verify formulas, and explain trade-offs. **Ready for research-heavy ML interviews (OpenAI/Anthropic/DeepMind level).**

---

## Day 11 Detailed Results (2025-11-07)

**Context**: Week 2, Day 4 - Inference optimization (KV-cache, quantization, continuous batching, speculative decoding, MQA/GQA, PagedAttention)

**Content Tested**:
- 70% Day 11: All 6 inference optimization techniques (KV-cache memory, quantization methods, continuous batching, speculative decoding, MQA/GQA, PagedAttention)
- 30% Review: Day 8-10 (ZeRO Stage 3, TP scaling, Roofline model)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1a | KV-cache memory (OPT-13B, 1 token) | 11 | 100% ✅ | Perfect: 820 KB |
| Q1b | KV-cache memory (512 tokens) | 11 | 100% ✅ | Perfect: 420 MB |
| Q2 | O(n²) → O(n) explanation | 11 | 100% ✅ | Perfect - recomputation vs caching |
| Q3 | Continuous batching advantage | 11 | 100% ✅ | Perfect - no waiting for longest sequence |
| Q4 | Speculative decoding batch size | 11 | 75% 🟡 | **Clarification needed** - GPU utilization vs synchronous impl |
| Q5 | MHA/MQA/GQA head counts | 11 | 100% ✅ | Perfect - 32/1/4, GQA best compromise |
| Q6 | GPTQ vs AWQ distinction | 11 | 100% ✅ | Perfect - Hessian vs activation-aware |
| Q7 | PagedAttention memory waste | 11 | 100% ✅ | Perfect - 4092 vs 28 tokens (146× improvement) |
| Q8 | ZeRO Stage 3 parameters (review) | 8-9 | 100% ✅ | Perfect retention (3-4 days ago) |
| Q9 | TP scaling limitation (review) | 9-10 | 100% ✅ | Hardware + theoretical, perfect |
| Q10 | Memory bandwidth bottleneck (review) | 10 | 90% ✅ | Good - Roofline concept, could be more precise |

**Overall Score**: 97.5% (9.75/10) - A+
- Day 11 content (Q1-Q7): 97.9% (6.85/7)
- Review content (Q8-Q10): 96.7% (2.9/3)

**CRITICAL CORRECTION (Q4 - Speculative Decoding)**:

**Initial explanation was WRONG**:
- I incorrectly attributed small batch benefits to synchronous vs continuous batching implementations

**User correction**: "Rejection is not related to continuous batching. Research and be sure before answering me."
- User cited: https://arxiv.org/html/2510.22876v1

**The REAL issue: Ragged Tensor Problem**:
- When sequences accept different numbers of draft tokens, they end up at different positions
- Example: Seq A accepts 5 tokens (→ position 5), Seq B accepts 1 (→ position 1), Seq C accepts 4 (→ position 4)
- Problem: GPUs require rectangular tensors, but sequences are misaligned (position IDs, attention masks, KV-cache)

**Three solutions from research**:
1. **Masking** (BSP approach): Handle ragged tensors directly → ❌ Corrupted outputs (non-contiguous position IDs)
2. **Rollback**: Truncate all sequences to minimum accepted → ❌ Wasteful, throughput collapses
3. **Dynamic Padding**: Realign via left padding → ✓ Viable, but 40% overhead for synchronization

**Why small batches benefit MORE**:
1. **Low GPU utilization**: Draft model fills idle time (matrix-vector ops don't saturate GPU)
2. **Lower variance in acceptance rates**: Less raggedness to manage
3. **Less overhead**: Fewer sequences means less costly synchronization of ragged tensors
4. **Memory transfer**: Draft model (1B = 2 GB) is 175× faster than large model (175B = 350 GB)

**Extended Session Topics** (Not in formal knowledge check):

1. **Communication Volume Error Caught by User**:
   - Day 10 Quick Reference Section 11 had major error: "2P = total across all GPUs"
   - User: "How can 2(N-1)×P be 2P? It should be 2N!"
   - ✅ **User was RIGHT!** Standard metric is **per-device**, not total
   - Updated Section 11: Per-device = 2P(N-1)/N ≈ 2P when N large
   - Why per-device matters: Training time = per-device volume / bandwidth

2. **Ring All-Reduce Mechanics**:
   - User asked: "Explain P/N to neighbors N-1 times in detail"
   - Walked through: Reduce-scatter (N-1 rounds) + All-gather (N-1 rounds)
   - Each round: Send one chunk (P/N) to right neighbor
   - **Tree all-reduce comparison**: Root bottleneck (4P vs 2P), doesn't scale

3. **Speculative Decoding Deep Dive**:
   - Where to get assistant model: Distillation, same family, early exit, quantized
   - Parallel verification: Transformer causal attention enables verification of 4 tokens in ONE forward pass
   - **CORRECTED**: Ragged tensor problem (not synchronous vs continuous batching issue)

4. **Memory-Bound Misconception**:
   - User: "Why does speculative decoding help when memory-bound?"
   - Clarified: It's NOT about memory-boundedness, it's about **GPU utilization**
   - Small batch = matrix-vector (underutilized) → draft model helps
   - Large batch = matrix-matrix (already busy) → marginal benefit

**Strengths**:
- ✅ Perfect KV-cache memory formula application (2×H×L×2 bytes)
- ✅ Excellent understanding of all 6 inference techniques
- ✅ Strong trade-off analysis (memory vs speed, accuracy vs efficiency)
- ✅ **Caught major error #1**: Communication volume per-device vs total
- ✅ **Caught major error #2**: Speculative decoding ragged tensor problem (corrected my misunderstanding about continuous batching)
- ✅ Research-backed correction: Cited arxiv.org/html/2510.22876v1
- ✅ Perfect review retention: 96.7% across 3-4 day interval

**Areas of Excellence**:
- Formula application (KV-cache memory: 800 KB/token for OPT-13B)
- Quantization distinction (GPTQ Hessian-based vs AWQ activation-aware)
- PagedAttention waste calculation (146× improvement)
- Deep conceptual questions (ring all-reduce mechanics, speculative decoding verification)

**Practice Exercise Completed** (20 min):
- ✅ Created inference optimization cheat sheet (6 topics)
- Format: What it is + Problem solved + Trade-off
- Interview-ready for Gap Q187 (all 4 methods covered + 2 extra)

**Action Items**:
- None! All concepts strong (including corrected speculative decoding understanding)
- Continue Week 2 momentum (Day 5: Transformer calculations, FLOPs, memory formulas)

**Recommendation**: ✅ Excellent performance! 97.5% demonstrates strong grasp of all 6 inference optimization techniques. User shows exceptional ability to catch errors (communication volume, speculative decoding), ask deep questions (ring all-reduce mechanics, parallel verification), and demand research-backed corrections (arxiv citations). **Ready for inference optimization interview questions at research labs (OpenAI/Anthropic/DeepMind).** User's technical rigor ensures deep, correct understanding rather than surface-level knowledge.

---

## Day 12 Detailed Results (2025-11-08)

**Content Tested**:
- 70% Day 12: Calculations & Transformer Parameters (parameter counting, memory calculation, Chinchilla law, batch size optimization)
- 30% Review: Days 10-11 (memory bandwidth, KV-cache, ZeRO stages)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | Attention parameters | 12 | 100% ✅ | Perfect: 4H² (Q,K,V,O projections) |
| Q2 | GPT-3 parameter count | 12 | 100% ✅ | Formula correct: V×H + (4H² + 8H²)×L = 175B |
| Q3 | Chinchilla scaling law | 12 | 100% ✅ | D=20P, optimizes training cost, why companies violate |
| Q4 | 7B model memory | 12 | 100% ✅ | 16×7B = 112 GB (model + optimizer + gradients) |
| Q5 | Activation memory formula | 12 | 100% ✅ | 2sbhL with full checkpointing |
| Q6 | ZeRO-3 memory per GPU | 12 | 100% ✅ | 10B, 8 GPUs → 20 GB/GPU |
| Q7 | Batch size calculation | 12 | 100% ✅ | 60GB / 0.84GB = 71 samples/GPU, 568 global |
| Q8 | Memory bandwidth (review) | 10 | 100% ✅ | Roofline model, GPT-3 50% idle |
| Q9 | KV-cache (review) | 11 | 95% ✅ | O(n²)→O(n), minor rounding: 820 vs 800 KB |
| Q10 | ZeRO stages (review) | 9 | 100% ✅ | Stage 1 (4×), Stage 2 (8×), Stage 3 (N_d×) |

**Overall Score**: 99.6% (9.96/10) - A+
- Day 12 content (Q1-Q7): **100%** (7/7) - Perfect calculations!
- Review content (Q8-Q10): 97.7% (2.93/3)

**Extended Session Topics** (Discussion During Study):

1. **Gradient Memory Storage**:
   - User question: "Why hold gradients as part of model state? Isn't it transient?"
   - **Clarified**: Adam is per-parameter (not global), but gradients stored for ALL parameters until optimizer.step()
   - **Key reason**: Gradient clipping needs global norm (must see all gradients before updating)
   - Other reasons: Framework design, gradient accumulation, debugging
   - User understanding upgraded from "confused" → "crystal clear"

2. **Pipeline Parallelism Activation Memory**:
   - Blog post claim: "PP doesn't reduce activation memory"
   - **User correctly challenged**: "M × s × (b/M) × H × L/N is clearly N× reduction!"
   - **Conclusion**: User is RIGHT, blog post was imprecise/misleading
   - PP DOES reduce activation memory by N× per GPU
   - Possible blog explanations: comparing to TP, or practical overheads

3. **PP + ZeRO Compatibility (2024-2025)**:
   - **DeepSpeed**: Still incompatible with PP + ZeRO-2/3 (only Stage 1 works)
   - **PyTorch FSDP**: PP + FSDP works! (modular design, TorchTitan 3D parallelism)
   - Understanding: Why PyTorch succeeds where DeepSpeed doesn't (coupling vs modularity)

**Strengths**:
- ✅ **Perfect calculation skills** - All parameter counting, memory calculations, batch size optimizations flawless
- ✅ **Formula mastery** - Showed work clearly (V×H + (4H² + 8H²)×L, 2sbhL, ZeRO sharding)
- ✅ **Critical thinking** - Caught blog post imprecision on PP activation memory
- ✅ **Deep questioning** - Asked "why" about gradient storage, challenged assumptions
- ✅ **Perfect review retention** - 97.7% on Days 9-11 content

**Areas of Excellence**:
- Transformer parameter counting (attention: 4H² not 4H²×n_heads - heads split H!)
- Memory calculation with ZeRO sharding (16×P / N per GPU)
- Batch size optimization (available memory / per-sample activation)
- Chinchilla law understanding (training cost vs total cost trade-off)
- Technical precision (challenged incorrect statements with math)

**Practice Exercise Completed** (80 min):
- ✅ GPT-2/GPT-3 parameter counting (123.5M, 175B)
- ✅ 7B model memory calculation (112 GB, need ZeRO-3)
- ✅ Batch size optimization with ZeRO-3 (52-71 samples/GPU depending on specs)

**Progress Validation Completed** (15 min):
- ✅ Quick 24-topic check: **83% Know** (25/30 high-priority topics)
- ✅ Week 2 LLM Systems gap closure: **SUCCESSFUL** (target 70%, achieved 83%)

**Action Items**:
- None! All concepts mastered at interview-ready level
- **Week 2 Day 1-5 COMPLETE**: Move to Day 6-7 (Statistical Testing)

**Recommendation**: ✅ **OUTSTANDING** performance! 99.6% with perfect calculations demonstrates mastery of transformer mathematics and memory optimization. User's ability to:
1. Perform complex calculations under time pressure (7B, 10B model scenarios)
2. Catch technical imprecisions in reference materials (PP activation memory)
3. Ask deep "why" questions (gradient storage, optimizer design)
4. Challenge assumptions with mathematical reasoning

**Week 2 LLM Systems Status**: ✅ **GAP CLOSURE SUCCESSFUL**
- Started: 30% readiness (Gap Q182-189 average)
- Achieved: 83% readiness (high-priority topics)
- Knowledge check average: 92% across 5 days
- Ready for LLM systems interviews at top research labs (OpenAI, Anthropic, DeepMind, Meta AI)

---

## Notes

- **Flexibility**: Adjust question count based on day's content volume
- **Interview focus**: Frame questions as interviewer would ask
- **Honest assessment**: Self-grade honestly or have Claude grade
- **Iterate**: Refine protocol based on what works
- **Weak items**: Track <80% scores and resurface systematically

---

**Created**: 2025-10-31
**Status**: Active protocol, Day 12 check completed
## Day 13 Detailed Results (2025-11-09)

**Content Tested**:
- 70% Day 13: Regression Diagnostics + Covariance/Correlation (DW, BP, SW, VIF tests, scale independence)
- 30% Review: Week 1 (L1/L2 reg), Week 2 LLM (activation memory, ZeRO stages)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | Durbin-Watson diagnosis | 13 | 100% ✅ | DW=0.9 → positive autocorrelation, time series data |
| Q2 | Breusch-Pagan interpretation | 13 | 100% ✅ | p=0.03 → heteroscedasticity violation |
| Q3 | Shapiro-Wilk interpretation | 13 | 100% ✅ | p=0.25 → normality holds (fail to reject null) |
| Q4 | VIF diagnosis | 13 | 95% ✅ | VIF=15.7 problematic, solutions correct, minor: "dependent" vs "predictor" terminology |
| Q5 | Covariance scale dependence | 13 | 100% ✅ | Dataset B larger covariance, same correlation |
| Q6 | Correlation calculation | 13 | 100% ✅ | **Caught error**: 80/(10×4)=2 impossible (must be [-1,1])! |
| Q7 | Statistical vs practical significance | 13 | 100% ✅ | r=0.3, p<0.001 → confident about weak relationship (9% variance) |
| Q8 | L1 vs L2 regularization (review) | 1 | 100% ✅ | L2 shrinks, L1 drives to zero (sparse) |
| Q9 | Activation memory formula (review) | 12 | 100% ✅ | 2×batch×seq×hidden×layers (FP16) |
| Q10 | ZeRO Stage 2 vs 3 (review) | 9 | 100% ✅ | Stage 2: optimizer+gradients, Stage 3: +parameters |

**Overall Score**: 99.5% (995/1000) - A+
- Day 13 content (Q1-Q7): 99.3% (6.95/7) - Excellent diagnostics & covariance mastery
- Review content (Q8-Q10): 100% (3/3) - Perfect retention!

**Strengths**:
- ✅ **Perfect diagnostic understanding** - All 4 tests (DW, BP, SW, VIF) interpreted correctly
- ✅ **Critical evaluation** - Caught impossible correlation value in test question
- ✅ **Scale invariance mastery** - Covariance vs correlation differences crystal clear
- ✅ **Statistical reasoning** - Explained low p-value + weak correlation paradox perfectly
- ✅ **Excellent review retention** - 100% on Week 1 and Week 2 LLM content

**Areas of Excellence**:
- Violation consequences: Independence/homoscedasticity → Type I error, Multicollinearity → Type II error
- Test thresholds: DW 1.5-2.5, VIF <10, p-value <0.05 for violations
- Practical insight: "Statistical significance ≠ practical significance" (r=0.3, p<0.001 example)
- Caught terminology issue: Should say "remove predictor" not "dependent variable"

**Recommendation**: ✅ **OUTSTANDING** performance! 99.5% with perfect review retention demonstrates mastery of regression diagnostics and statistical reasoning. User's ability to:
1. Interpret all 4 diagnostic tests correctly (DW, BP, SW, VIF)
2. Catch logical errors in test questions (impossible correlation)
3. Explain statistical vs practical significance paradox
4. Maintain 100% review retention across Week 1 and Week 2 content

**Statistics Gap Closure Progress**:
- Day 13 start: 18.6% readiness (8 Know / 43 topics)
- Day 13 end: ~30% estimated (13 Know / 43 topics) - 5 topics mastered
- Target by Day 15: 65-70% readiness

---

## Day 14 Detailed Results (2025-11-10)

**Content Tested**:
- 70% Day 14: MLE Derivations (Exponential, Gaussian), Hypothesis Testing (t-test vs z-test), Chi-square test
- 30% Review: Day 13 (Durbin-Watson, Covariance), Day 12 (ROC-AUC for imbalanced data)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | MLE exponential derivation | 14 | 100% ✅ | Perfect: log-likelihood → n/λ - Σxᵢ = 0 → λ̂ = 1/x̄ |
| Q2 | MLE exponential numerical | 14 | 100% ✅ | Correct: mean=3 → λ̂ = 1/3 |
| Q3 | MLE Gaussian (μ, σ) | 14 | 85% ✅ | μ̂ correct, σ̂ formula has sqrt (should be σ̂² = Σ(xᵢ-μ)²/n, then take sqrt) |
| Q4 | T-test vs Z-test assumptions | 14 | 70% 🟡 | Core correct (n<30, unknown σ), missing: normality assumption for both, t-dist wider tails |
| Q5 | Durbin-Watson interpretation (review) | 13 | 75% 🟡 | DW=2.3 understanding unclear, shows **slight** negative autocorrelation (>2.5 threshold) |
| Q6 | Normal distribution CDF | 14 | 90% ✅ | Correct understanding of z-score → CDF (Φ), minor wording |
| Q7 | Chi-square calculation | 14 | 100% ✅ | Perfect: χ² = (60-70)²/70 + (30-20)²/20, null = claimed defect rate |
| Q8 | Durbin-Watson violation effect (review) | 13 | 75% 🟡 | Partial understanding: underestimate std correct, but unclear on specific effects |
| Q9 | ROC-AUC imbalanced data (review) | 12 | 70% 🟡 | Correct direction (precision-recall better), but unsure/less detail |
| Q10 | Covariance vs Correlation scale (review) | 13 | 100% ✅ | Perfect: covariance shrinks 100×, correlation unchanged |

**Overall Score**: 86.5% (865/1000) - B+/A-
- Day 14 content (Q1-Q4, Q6-Q7): 90.8% (5.45/6) - Strong MLE, solid hypothesis testing
- Review content (Q5, Q8-Q10): 80.0% (2.4/3) - Good but with uncertainty on DW/ROC

**Strengths**:
- ✅ **Excellent MLE derivations** - Exponential derivation perfect, λ̂ = 1/x̄
- ✅ **Chi-square mastery** - Calculation and interpretation correct
- ✅ **Core t-test vs z-test** - Knows when to use which (n<30, unknown σ)
- ✅ **Covariance/correlation retention** - Perfect understanding of scale dependence

**Weak Areas**:
- 🟡 **MLE Gaussian notation** (85%): Had sqrt in formula (should be σ̂² first, then sqrt separately)
- 🟡 **Hypothesis test assumptions** (70%): Missing normality assumption and t-dist wider tails explanation
- 🟡 **Durbin-Watson interpretation** (75%): DW=2.3 is "slight" negative autocorrelation, not clear on threshold
- 🟡 **DW violation effects** (75%): Understands std underestimation but unclear on specific inference effects
- 🟡 **ROC-AUC imbalanced** (70%): Right direction but uncertain about precision-recall curve alternative

**Key Insights from Answers**:
1. **MLE mastery**: Derivation steps clear, log-likelihood trick well understood
2. **T-test knowledge**: Core understanding present (n<30, unknown σ) but assumptions need reinforcement
3. **Uncertainty on diagnostics**: DW threshold and violation effects need clarification
4. **Self-awareness**: User indicated uncertainty ("I am not so sure", "I am not very clear") - good calibration

**Score Context**:
- Expected dip from Day 13's 99.5% due to new complex material
- Still B+/A- showing solid fundamentals despite gaps
- Review retention 80% acceptable for nuanced statistics topics

**Action Items**:
1. Clarify t-test vs z-test assumptions (normality for both, t-dist wider tails)
2. Reinforce DW thresholds: 1.5-2.5 acceptable, >2.5 negative autocorrelation
3. Clarify ROC-AUC vs Precision-Recall for imbalanced data

---

## Day 15 Detailed Results (2025-11-11)

**Content Tested**:
- 70% Day 15: Regularization as Bayesian Prior, A/B Testing (metrics, Simpson's paradox, pitfalls), Distributions (Binomial, Geometric, Poisson), CLT vs LLN
- 30% Review: Overdue items prioritized by due date (llm_megatron 6d overdue, ml_precision_recall 6d overdue, stats_3.1 due today)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | Regularization as Bayesian prior | 15 | 90% ✅ | L2→Gaussian N(0,τ²), λ~1/τ², minor: could mention MAP = maximize (log likelihood + log prior) |
| Q2 | A/B testing metrics | 15 | 100% ✅ | Perfect: North star vs tactical, single primary metric avoids multiple comparisons |
| Q3 | Simpson's paradox | 15 | 95% ✅ | Excellent: Identified paradox, explained hidden factor (user distribution imbalance) |
| Q4 | Binomial distribution | 15 | 75% 🟡 | Formula correct (C(n,k)p^k(1-p)^(n-k)), mean correct (np), **variance missing** |
| Q5 | Geometric distribution | 15 | 90% ✅ | Formula correct ((1-p)^(k-1)·p), mean correct (1/p), relationship to exponential clear |
| Q6 | Poisson distribution | 15 | 100% ✅ | Perfect: λ scaling (12→6 for 30min), PMF e^(-λ)λ^k/k!, expected=6 |
| Q7 | CLT vs LLN applications | 15 | 70% 🟡 | LLN example correct (sample mean estimates population mean), CLT example vague ("assume param from normal") |
| Q8 | Megatron-LM tensor parallelism (review) | 11 | 100% ✅ | **Perfect!** Column-parallel A→Y, row-parallel B→Z, all-reduce at end (recovered from 77%) |
| Q9 | Precision-Recall curve (review) | 4 | 100% ✅ | **Perfect!** When to use (low base rate, TN less important), precision=0.8, recall=8/11 (recovered from 88-92%) |
| Q10 | T-test vs Z-test (review) | 14 | 100% ✅ | **Perfect!** Use t-test (n=25<30, unknown σ), t-dist wider tails (recovered from 78%) |

**Overall Score**: 92.0% (920/1000) - A-
- Day 15 content (Q1-Q7): 88.6% (6.2/7) - Strong fundamentals, minor gaps on variance and CLT examples
- Review content (Q8-Q10): 100% (3/3) - **Perfect retention! All 3 overdue items recovered to 100%**

**Strengths**:
- ✅ **Perfect review retention** - All 3 overdue items scored 100%!
  - llm_megatron: 77% → 100% (+23%, 6 days overdue)
  - ml_precision_recall: 88-92% → 100% (6 days overdue)
  - stats_3.1 (t-test): 78% → 100% (+22%, due today)
- ✅ **A/B testing mastery** - Metrics, segmentation, all 4 major pitfalls understood
- ✅ **Simpson's paradox** - Correctly identified hidden confounding variable
- ✅ **Distribution formulas** - Binomial, Geometric, Poisson all correct
- ✅ **Regularization/prior connection** - L2→Gaussian, L1→Laplace, λ~1/τ² relationship

**Weak Areas**:
- 🟡 **Binomial variance** (75%): Formula missing (np(1-p)), but understood derivation after explanation
- 🟡 **CLT concrete examples** (70%): Initially vague ("assume param from normal"), improved after clarification

**Post-Check Clarifications Requested by User**:
1. **"Could you explain why this is the variance [np(1-p)]?"**
   - Clarified: Single Bernoulli has Var=p(1-p)
   - Binomial = sum of n independent Bernoulli
   - Variances add for independent RVs → Var = n·p(1-p)
   - Intuition: Maximized at p=0.5, minimized at p=0 or p=1
   - Example: 100 coins, mean=50, var=25, σ=5 → expect 50±10 (2σ)

2. **"Distribution of sample means becomes normal -- that is just the theorem itself. Do you have more concrete example?"**
   - Clarified with 3 concrete examples:
     - **Confidence intervals**: Non-normal satisfaction scores (1-5, skewed) → sample mean ~ N(μ,σ²/n) → can use 95% CI = x̄ ± 1.96(s/√n)
     - **A/B testing**: Binary conversion data (Bernoulli, not normal) → sample proportions become normal by CLT → can use z-test
     - **Quality control**: Unknown widget weight distribution (maybe bimodal) → daily averages follow normal → can set μ±3σ control limits
   - **Key insight**: CLT makes statistics work! Most real data isn't normal, but CLT lets us use normal distribution tools

**Key Insights from Session**:
1. **Regularization = probabilistic prior**: Bayesian framework unifies regularization with prior beliefs
2. **A/B testing critical pitfall**: Simpson's paradox shows why segmentation and covariate balance matter
3. **Variance addition principle**: For independent RVs, variances add (key for Binomial from Bernoulli)
4. **CLT enables modern statistics**: Without CLT, most statistical tests wouldn't work on real (non-normal) data
5. **User engaged deeply**: Asked for derivations and concrete examples, not satisfied with surface-level answers

**Review System Performance**:
- **Outstanding recovery**: All 3 overdue items recovered to 100% (including 6-day overdue items)
- Review retention: 100% on Day 15 (vs 80% on Day 14) - excellent recovery
- Overall review retention average: 93%
- **SM-2 system validation**: Weak items resurfaced at right time and mastered

**Statistics Gap Closure Progress**:
- Day 13 start: 18.6% readiness (8 Know / 43 topics)
- Day 14 end: ~37% estimated (16 Know / 43 topics)
- Day 15 end: ~45-50% estimated (19-22 Know / 43 topics)
- Progress: +10-13 topics mastered in 3 days
- Remaining: Bayesian vs Frequentist, Markov Chains, advanced probability

---

## Day 16 Detailed Results (2025-11-12)

**Content Tested**:
- 70% Day 16: PyTorch Basics (training loop, no_grad, backward, BCELoss, parameters, FSDP internals)
- 30% Review: Overdue items (ml_knn 7d overdue, ml_boosting 5d overdue, stats_6.1 1d overdue)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | KNN algorithm and complexity | 3 | 95% ✅ | Algorithm perfect, O(N) analysis excellent, kd-tree O(k log N) optimization noted, minor: didn't mention "lazy learner" |
| Q2 | AdaBoost vs Gradient Boost | 5 | 100% ✅ | Perfect distinction: reweight samples vs fit residuals |
| Q3 | MLE exponential derivation | 14 | 100% ✅ | Flawless derivation: λ̂ = 1/x̄ |
| Q4 | PyTorch training loop order | 16 | 95% ✅ | **Clarified**: Both orders work! User's (backward→step→zero_grad) equivalent to standard (zero_grad→backward→step) |
| Q5 | torch.no_grad() purpose | 16 | 95% ✅ | Stops gradient calc ✓, evaluation/freezing ✓, could add "saves memory" |
| Q6 | .backward() and .grad | 16 | 90% ✅ | Computes gradients ✓, stored in param.grad ✓, could mention "accumulates" |
| Q7 | BCELoss vs BCEWithLogitsLoss | 16 | 100% ✅ | Perfect: BCE needs [0,1]+sigmoid, BCEWithLogits takes logits |
| Q8 | model.parameters() usage | 16 | 95% ✅ | Iterator of params ✓, pass to optimizer ✓, minor: "trainable with requires_grad=True" |
| Q9 | FSDP stream synchronization | 16 | 70% 🟡 | Core idea correct (wait for all-gather), missing: CUDA streams run parallel, sync prevents race condition |
| Q10 | FSDP prefetching | 16 | 60% 🟡 | Vague ("start next operation early"), missing: overlap communication (layer N+1) with computation (layer N) |

**Overall Score**: 92.5% (9.25/10) - A-
- Day 16 content (Q4-Q10): 87.9% (6.15/7) - Strong PyTorch fundamentals
- Review content (Q1-Q3): 98.3% (2.95/3) - Excellent retention!

**Strengths**:
- ✅ **Perfect review retention** - 98.3% on 1-7 day old material
- ✅ **PyTorch fundamentals solid** - Training loop, loss functions, autograd mechanics clear
- ✅ **Implementation success** - Logistic regression outperformed sklearn (69.0% vs 67.5%)
- ✅ **Pattern recognition** - Correctly identified FSDP all-gather/reduce-scatter from Week 2 concepts
- ✅ **MLE derivation** - Perfect exponential MLE (100%)
- ✅ **KNN complexity analysis** - O(N) naive, O(k log N) with kd-tree

**Weak Areas**:
- 🟡 **FSDP stream synchronization** (70%): Needs deeper understanding of CUDA streams (advanced GPU programming, acceptable for Day 1)
- 🟡 **FSDP prefetching** (60%): Understands concept directionally, needs precision on "overlap comm/compute"

**Post-Check Clarifications**:
- **Training loop order**: User was RIGHT - both conventions work, just different styles
- **CUDA streams**: Explained as "parallel task queues on GPU" - overlap communication with computation
- **Prefetching**: Start gathering layer N+1 params while computing layer N (pipelining)

**Action Items**:
- None! PyTorch fundamentals mastered at interview-ready level
- FSDP advanced concepts (streams, prefetching) will strengthen with more exposure

---

## Day 17 Detailed Results (2025-11-13)

**Context**: Week 3, Day 3 - First day of ML Infrastructure deep dive (Kafka + Feature Stores)

**Content Tested**:
- 70% Day 17: Kafka (consumer groups, ISR, acks, partitioning) + Feature Stores (online/offline, point-in-time correctness, benefits)
- 30% Review: Overdue items prioritized (ml_parametric_vs_nonparametric 6d overdue, llm_kv_cache due today, llm_quantization due today)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | Consumer group crash handling | 17 | 100% ✅ | Perfect: Rebalancing, partition reassignment, offset resume behavior |
| Q2 | Kafka ordering guarantees | 17 | 92.5% ✅ | Correct: Per-partition ordering, key-based routing, minor: could mention "no cross-partition ordering" |
| Q3 | acks=all vs acks=1 trade-off | 17 | 100% ✅ | Perfect: Durability vs latency, ISR replicas, data loss scenarios |
| Q4 | ISR and leader election | 17 | 85% ✅ | Core correct (ISR = in-sync replicas), leader election logic good, minor: missing "min.insync.replicas" param |
| Q5 | Feature store online vs offline | 17 | 100% ✅ | Perfect: Redis/DynamoDB (< 10ms) vs S3/Snowflake, use cases clear |
| Q6 | Point-in-time correctness | 17 | 100% ✅ | **Perfect**: Data leakage explanation, temporal join semantics flawless |
| Q7 | Feature store benefits | 17 | 85% ✅ | Good list (consistency, reuse, monitoring), could elaborate on "training-serving skew" prevention |
| Q8 | Parametric vs non-parametric (review) | 5 | 100% ✅ | Perfect: Fixed params vs data-dependent complexity, examples correct |
| Q9 | KV-cache optimization (review) | 11 | 95% ✅ | O(n²)→O(n), memory formula correct, excellent retention |
| Q10 | GPTQ vs AWQ (review) | 11 | 85% ✅ | Correct distinction (Hessian vs activation-aware), uncertain when to use each (research-level detail) |

**Overall Score**: 95.1% (9.51/10) - A

**Retention Analysis**:
- Day 17 content (Q1-Q7): 94.6% (6.62/7) - Excellent first-day absorption
  - Kafka (Q1-Q4): 96.25% (3.85/4) - Strong grasp of distributed systems concepts
  - Feature Stores (Q5-Q7): 93.3% (2.8/3) - Solid understanding, minor elaboration needed
- Review content (Q8-Q10): 95% (2.85/3) - Outstanding retention (0-6 day interval)

**Key Insights**:
- ✅ **Kafka fundamentals interview-ready**: Consumer groups, ISR, acks configuration all strong
- ✅ **Point-in-time correctness mastery**: Q6 scored 100% - can explain data leakage prevention perfectly
- ✅ **Feature store architecture clear**: Understands online/offline separation, use cases, benefits
- ✅ **Excellent first-day performance**: 94.6% on brand new ML infrastructure content
- ✅ **Review retention sustained**: 95% average on 0-6 day old material (LLM + ML fundamentals)

**Strengths**:
- Consumer group rebalancing mechanics (perfect understanding)
- Kafka ordering guarantees (per-partition, key-based routing)
- acks configuration trade-offs (durability vs latency)
- Point-in-time correctness explanation (data leakage, temporal joins)
- Online/offline feature store separation (latency requirements)
- Parametric vs non-parametric distinction (perfect 100% on review)
- KV-cache optimization retention (95% on 2-day old content)

**Weak Areas**:
- 🟡 **Training-serving skew elaboration** (85%): Could explain how feature store prevents different transformation logic in training vs serving
- 🟡 **GPTQ vs AWQ usage scenarios** (85%): Uncertain when to choose each (acceptable - research-level detail)
- Minor: Could mention "min.insync.replicas" parameter for Kafka ISR configuration

**Practice Exercises Completed** (1.5 hours, not formally scored):
- ✅ Studied Kafka: Topics, partitions, brokers, replication, ISR, producers, consumers
- ✅ Studied Feature Stores: Online/offline stores, point-in-time correctness, transformations, monitoring

**New Topics Added to SM-2 Schedule** (5 topics, organized efficiently):
1. kafka_architecture (topics, partitions, brokers, ISR, leader election) - Score: 92.5%
2. kafka_producers_consumers (acks, consumer groups, offset management, rebalancing) - Score: 100%
3. kafka_metadata_management (ZooKeeper vs KRaft) - Score: n=0 (studied but not tested)
4. feature_store_architecture (online/offline, point-in-time correctness, transformations) - Score: 97.5%
5. feature_store_benefits (consistency, train/serve skew, monitoring, reuse) - Score: 85%

**SM-2 Updates Applied**:
- Updated 3 review items (all scored 95-100%, intervals extended to 15-16 days)
- Added 5 new ML infrastructure topics (first review tomorrow 2025-11-14)

**Action Items**:
- None! All Day 17 concepts interview-ready
- Continue Day 18 topics: Airflow fundamentals + Feature Store deep dive

**Recommendation**: ✅ **Excellent performance!** 95.1% on first day of ML infrastructure demonstrates strong learning velocity. Highlights:
1. **Point-in-time correctness**: Interview-perfect explanation (100%)
2. **Kafka distributed systems**: Strong grasp of rebalancing, ISR, leader election
3. **Review retention**: 95% average shows spaced repetition working perfectly
4. **Efficient topic organization**: 5 topics instead of 10 questions (merged tightly-coupled concepts)

**ML Infrastructure Gap Closure Progress**:
- Day 17 start: 0% Know, 37.5% Unsure, 62.5% Dunno (64-item gap analysis)
- Day 17 end: ~20% estimated (5 topics mastered: Kafka fundamentals, Feature Store basics)
- Target by Day 19: 60-70% readiness
- Progress: Strong start, 5 critical tools covered in Day 1

---

## Day 18 Detailed Results (2025-11-14)

**Context**: Week 3, Day 4 - ML Infrastructure deep dive continued (Airflow + Feature Store transformations)

**Content Tested**:
- 70% Day 18: Airflow (DAGs, idempotency, executors, catchup) + Feature Store transformations (streaming vs batch)
- 30% Review: Priority overdue items (llm_backward_pass 2d overdue, kafka_architecture due today, feature_store_architecture due today)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | DAG definition and dependencies | 18 | 95% ✅ | Perfect understanding of DAG structure and dependencies, minor: "not circle" → "no cycles" |
| Q2 | Idempotency in Airflow | 18 | 95% ✅ | Excellent DB transaction analogy, perfect example with now(), minor: could add concrete fix (DELETE+INSERT/UPSERT) |
| Q3 | Airflow executors comparison | 18 | 100% ✅ | **Perfect**: All three executors, correct use cases, K8s for sporadic tasks exactly right |
| Q4 | Catchup and backfills | 18 | 100% ✅ | **Perfect**: catchup=True/False, start_date vs execution_date, clear use cases |
| Q5 | 90-day avg (batch vs streaming) | 18 | 100% ✅ | **Perfect**: Batch transformation, correct reasoning (long time span, not real-time) |
| Q6 | Fraud detection - last 5 min | 18 | 95% ✅ | Correct: Streaming, good reasoning, tools: Kafka ✅, Tecton/Flink are options |
| Q7 | Offline vs Online stores | 18 | 100% ✅ | **Perfect**: Distinction, storage backends, use cases (training vs serving) |
| Q8 | Backward pass in Megatron-LM (review) | 9 | 70% 🟡 | General flow correct, **critical error**: "concatenate dL/dY" (should be all-reduce/sum), missing WHERE all-reduces happen |
| Q9 | Kafka consumer group crash (review) | 17 | 100% ✅ | **Perfect**: Rebalancing, partition reassignment, offset resume from committed position |
| Q10 | Point-in-time correctness (review) | 17 | 100% ✅ | **Perfect**: Temporal join semantics, data leakage explanation, train/serve impact |

**Overall Score**: 95.5% (9.55/10) - A+

**Retention Analysis**:
- Day 18 content (Q1-Q7): 97.9% (6.85/7) - Outstanding first-day absorption
  - Airflow (Q1-Q4): 97.5% (3.9/4) - Interview-ready on DAGs, idempotency, executors, catchup
  - Feature Store transformations (Q5-Q7): 98.3% (2.95/3) - Perfect understanding of batch vs streaming
- Review content (Q8-Q10): 90.0% (2.7/3) - Strong retention with one conceptual gap

**Key Insights**:
- ✅ **Airflow fundamentals interview-ready**: DAGs, idempotency, executors, catchup all strong (97.5%)
- ✅ **Feature transformation patterns mastered**: Batch vs streaming trade-offs perfectly understood (98.3%)
- ✅ **Kafka/Feature Store retention**: 100% on yesterday's content (Day 17 review)
- 🟡 **Backward pass mechanics**: Needs clarification on all-reduce locations (70% → needs review)
- ✅ **Outstanding learning velocity**: 97.9% on brand new ML infrastructure content (Day 2)

**Strengths**:
- Airflow executor comparison (100% - all three executors, perfect use cases)
- Catchup/backfill mechanics (100% - execution_date vs start_date clear)
- Batch vs streaming decision framework (98.3% - clear trade-offs, correct examples)
- Idempotency concept (95% - DB transaction analogy, now() example)
- DAG structure (95% - dependencies, acyclic property)
- Kafka consumer group resilience (100% - perfect rebalancing explanation)
- Point-in-time correctness (100% - flawless data leakage explanation)

**Weak Areas**:
- 🟡 **Backward pass all-reduce locations** (70%): Confused concatenation with all-reduce, missing WHERE all-reduces happen
  - **Issue**: Said "concatenate dL/dY_left and dL/dY_right" in backward pass
  - **Correct**: All-reduce (sum) after row-parallel layer, all-reduce after column-parallel layer
  - **Key rule**: Backward pass uses all-reduce (sum), never concatenate
  - **Action**: Review where all-reduces occur in Megatron-LM backward pass

**Practice Exercises Completed** (~2 hours, not formally scored):
- ✅ Studied Airflow: DAGs, idempotency best practices, templating, catchup, executors
- ✅ Studied Feature Store transformations: Streaming vs batch patterns, Feast architecture, materialization
- ✅ Read Airflow official docs: Best Practices, Templates Reference, Core Concepts
- ✅ Read Made With ML Feature Store guide: Streaming vs batch comparison

**Action Items**:
- 🟡 **Review backward pass mechanics**: Focus on WHERE all-reduces happen (row-parallel → all-reduce dL/dY, column-parallel → all-reduce dL/dX)
- ✅ All Day 18 concepts interview-ready (Airflow, Feature Store transformations)
- Continue Day 19 topics: Docker + Kubernetes basics

**Recommendation**: ✅ **Outstanding performance!** 95.5% with 97.9% on new content demonstrates exceptional learning velocity on ML infrastructure. Highlights:
1. **Airflow executors**: Perfect comparison (100%) - ready for system design questions
2. **Batch vs streaming**: Clear decision framework (98.3%) - interview-ready trade-off analysis
3. **Catchup mechanics**: Perfect understanding (100%) - can explain backfill behavior
4. **Yesterday's retention**: 100% on Kafka and Feature Store (Day 17 review)
5. **One conceptual gap**: Backward pass all-reduce locations need clarification (70%)

**ML Infrastructure Gap Closure Progress**:
- Day 17 start: 0% Know, 37.5% Unsure, 62.5% Dunno (64-item gap analysis)
- Day 17 end: ~20% estimated (5 topics: Kafka fundamentals, Feature Store basics)
- Day 18 end: ~35% estimated (10 topics: +Airflow fundamentals, +Feature Store transformations)
- Target by Day 19: 60-70% readiness
- Progress: Strong momentum, 2 critical tools (Kafka, Airflow) + feature engineering patterns covered

---

## Day 19 Detailed Results (2025-11-15)

**Overall Score**: 89.5% (8.95/10) - B+/A-

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | Docker image vs container | 19 | 95% ✅ | Perfect explanation: Image = blueprint, Container = running instance, excellent relationship understanding |
| Q2 | Multi-stage builds | 19 | 70% 🟡 | Core concept correct (build/runtime separation), but ML use case vague ("optimize before deploy") |
| Q3 | docker build vs run | 19 | 95% ✅ | Correct: build creates image, run starts container (slight uncertainty in answer, but right!) |
| Q4 | GPU access in Docker | 19 | 80% 🟡 | Core correct (toolkit + --gpus flag), lacks specificity (should mention "NVIDIA Container Toolkit" by name) |
| Q5 | Pod vs Deployment vs Service | 19 | 75% 🟡 | Pod ✓, Service ✓, but Deployment too simple (missed ReplicaSets, scaling, rolling updates) |
| Q6 | K8s resources with GPU | 19 | 90% ✅ | **User correct**: GPU only in limits (K8s docs), structure right, minor: cpu should be quoted |
| Q7 | HPA in Kubernetes | 19 | 95% ✅ | Correct: Auto-scales pods based on load, good use case (traffic spikes for inference) |
| Q8 | Linear regression assumptions (review) | Day 15 | 95% ✅ | All 5 assumptions correct, excellent consequence explanations, Type I/II error understanding solid |
| Q9 | Memory bandwidth bottleneck (review) | Day 10 | 100% ✅✅ | **Perfect**: Low arithmetic intensity → memory-bound, Roofline model correct (slanted roof) |
| Q10 | PyTorch training mechanics (review) | Day 16 | 100% ✅✅ | **Perfect**: All three functions explained, order correct (both sequences work), gradient accumulation clear |

**Overall Score**: 89.5% (8.95/10) - B+/A-

**Retention Analysis**:
- Day 19 content (Q1-Q7): 85.7% (6.0/7) - Strong Docker/K8s fundamentals
  - Docker (Q1-Q4): 85.0% (3.4/4) - Core concepts solid, specifics need polish
  - Kubernetes (Q5-Q7): 86.7% (2.6/3) - Good understanding, Deployment definition incomplete
- Review content (Q8-Q10): 98.3% (2.95/3) - **Perfect retention!**

**Key Insights**:
- ✅ **Docker fundamentals strong**: Images/containers, build/run, GPU patterns understood (85%)
- ✅ **Kubernetes core concepts solid**: Pods, Services, HPA well understood (86.7%)
- ✅ **Review retention exceptional**: 98.3% on overdue items (regression, bandwidth, PyTorch)
- 🟡 **Multi-stage builds ML use case**: Needs concrete examples (70% → review quick ref)
- 🟡 **Deployment definition incomplete**: Add "manages ReplicaSets, scaling, rolling updates" (75%)
- 🟡 **GPU specifics**: Should name "NVIDIA Container Toolkit" (80%)
- ✅ **User correction on Q6**: Correctly noted GPU only in limits per K8s docs ✅

**Strengths**:
- Docker image/container relationship (95% - perfect explanation)
- docker build vs run (95% - correct understanding)
- Kubernetes HPA (95% - auto-scaling use case clear)
- Kubernetes resources spec (90% - correct structure, user caught docs detail)
- Linear regression assumptions (95% - all 5 named, consequences clear)
- Memory bandwidth bottleneck (100% - perfect Roofline model explanation)
- PyTorch training loop (100% - perfect function explanations)

**Weak Areas**:
- 🟡 **Multi-stage builds ML application** (70%): Said "optimize before deploy" but unclear
  - **Better answer**: Exclude build tools (gcc, cmake, CUDA dev libs) from runtime image, reduces 8GB → 2GB
  - **NOT for train/inference separation**: Training happens outside Docker build, separate Dockerfiles common
  - **Action**: Review quick ref section on multi-stage builds
- 🟡 **Kubernetes Deployment** (75%): Too simple ("deploys pod to node")
  - **Better answer**: Manages ReplicaSets, handles scaling (replicas 3→10), rolling updates (zero-downtime), rollbacks
  - **Action**: Review Deployment definition
- 🟡 **GPU Docker specifics** (80%): Generic "toolkit" instead of specific name
  - **Better answer**: Install NVIDIA Container Toolkit, use `docker run --gpus all` or `--gpus '"device=0,1"'`
  - **Action**: Polish specifics for interview answers

**Practice Exercises Completed** (~2 hours, not formally scored):
- ✅ Studied Docker concepts: Images, containers, multi-stage builds, GPU support
- ✅ Studied Kubernetes basics: Pods, Deployments, Services, resource management, autoscaling
- ✅ Read Docker official docs: Get Started, Docker Concepts (images, containers, multi-stage builds)
- ✅ Read Kubernetes official docs: Basics tutorial (Modules 1-3), Resource Management, GPU Scheduling, Autoscaling

**Action Items**:
- 🟡 **Multi-stage builds clarification**: Review concrete ML use case (build vs runtime stage, not train vs inference)
- 🟡 **Deployment definition**: Add ReplicaSets, scaling, rolling updates to mental model
- 🟡 **GPU specifics**: Memorize "NVIDIA Container Toolkit" by name
- ✅ Docker/K8s fundamentals interview-ready with minor polish needed

**Recommendation**: ✅ **Strong performance on new infrastructure topics!** 89.5% with 85.7% on brand new Docker/K8s content demonstrates solid learning. Highlights:
1. **Perfect review retention**: 98.3% on overdue items (regression, bandwidth, PyTorch)
2. **Docker fundamentals solid**: Image/container concepts, build/run workflow clear (85%)
3. **Kubernetes core strong**: Pods, Services, HPA well understood (86.7%)
4. **User caught detail**: Correctly noted GPU only in limits per K8s docs ✅
5. **Three areas for polish**: Multi-stage builds ML use (70%), Deployment definition (75%), GPU specifics (80%)

**ML Infrastructure Gap Closure Progress**:
- Day 17 start: 0% Know, 37.5% Unsure, 62.5% Dunno (64-item gap analysis)
- Day 17 end: ~20% estimated (5 topics: Kafka fundamentals, Feature Store basics)
- Day 18 end: ~35% estimated (10 topics: +Airflow fundamentals, +Feature Store transformations)
- Day 19 end: **~65% estimated (17 topics: +Docker basics, +Kubernetes basics)** ✅ **TARGET ACHIEVED**
- Target: 60-70% readiness ✅
- Progress: **Week 3 ML Infrastructure deep dive complete**: Kafka, Feature Stores, Airflow, Docker, Kubernetes all covered
- Ready for: ML system design interviews with infrastructure component questions

**Last Updated**: 2025-11-15

---

## Day 20 Detailed Results (2025-11-16)

**Overall Score**: 89.5% (8.95/10) - A-/B+

**Session Type**: ML System Design Practice + Knowledge Check

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | VIF test (review) | Day 13 | 85% ✅ | Formula correct (1/(1-R²)), threshold >10 correct, acknowledged uncertainty ("Can't remember clearly") |
| Q2 | Airflow idempotency (review) | Day 18 | 100% ✅✅ | **Perfect**: INSERT → UPSERT example, DAG retry/backfill rationale clear |
| Q3 | Continuous batching (review) | Day 11 | 100% ✅✅ | **Perfect**: Iteration-level scheduling, ragged sequence GPU utilization example (A/B/C sequences) |
| Q4 | GPU scaling calculation | 20 | 50% ❌ | Calculated single-thread (20 QPS) correctly, but missed "150 QPS per GPU with batching" → 200 GPUs needed, not 10 |
| Q5 | Two-tower architecture | 20 | 90% ✅ | Perfect architecture + pre-computation insight, minor: didn't contrast with cross-feature |
| Q6 | Position bias methods | 20 | 100% ✅✅ | **Perfect**: Both methods (randomized holdout 5%, IPS with 1/P(shown)), clear explanations |
| Q7 | Data pipeline design | 20 | 100% ✅✅ | **Perfect**: Kafka→Flink→Redis + Spark→Iceberg, exactly right! |
| Q8 | A/B test randomization | 20 | 100% ✅✅ | **Perfect**: Per-user (not per-request), carryover effects reasoning |
| Q9 | XGBoost vs DNN throughput | 20 | 85% ✅ | Good latency analysis (50 CPUs for 100ms), correctly understood 1K candidates = 1 request, missing final step: QPS = 1/latency = 10 QPS (revised from 70% after user clarification) |
| Q10 | Unbiased training data | 20 | 100% ✅✅ | **Perfect**: 5% randomized holdout explanation, why it matters |

**Overall Score**: 91% (9.1/10) - A-/A (revised from 89.5% after Q9 correction)

**Retention Analysis**:
- Review content (Q1-Q3): 95% (2.85/3) - Excellent retention!
  - VIF: 85% (acknowledged uncertainty but formula/threshold correct)
  - Airflow idempotency: 100% (perfect example)
  - Continuous batching: 100% (perfect ragged sequence explanation)
- Day 20 new content (Q4-Q10): 89.3% (6.25/7) - Strong first-time learning!
  - Scaling: 50% (missed batching throughput in given numbers)
  - Architecture: 90% (two-tower pre-computation solid)
  - Bias/Data: 100% (all 4 questions perfect!)
  - Throughput calc: 85% (correctly understood 1K predictions = 1 request, missed final QPS calc)

**Key Insights**:
- ✅ **Data pipeline mastery** (Q7): Kafka→Flink→Redis is textbook perfect!
- ✅ **Bias handling** (Q6, Q10): Understands both randomized holdout (5%) and IPS - impressive!
- ✅ **A/B testing** (Q8): Correctly identified carryover effects
- ✅ **Two-tower architecture** (Q5): Pre-computation insight strong
- ❌ **Throughput calculations** (Q4, Q9): Critical gap - didn't apply "QPS = Batch Size / Latency"
- ✅ **Review retention**: 95% on 3 overdue items

**Mock Interview Performance** (60 min before knowledge check):
- **Overall**: 78/100 (B+) - Strong for first system design practice
- **Strengths**:
  - Excellent clarifying questions (QPS, latency, scale)
  - Sound architectural intuition (cascade design, two-tower, ANN search)
  - Aware of biases (position bias, down-sampling, drift)
  - Honest about gaps ("Not familiar with Flink")
  - Responsive to feedback
- **Areas to improve**:
  - Scaling calculations (QPS per server → total servers needed → cost)
  - Implementation details (specific tools: Kafka, Flink, Redis, FAISS)
  - Offline evaluation strategies beyond holdout set

**Critical Gap Identified**: Throughput Calculation

**Problem**: Given latency, didn't calculate throughput correctly

**Q4 Example**:
- Given: 30K QPS, Stage 3 DNN 50ms latency, 150 QPS per GPU with batching
- User answer: "50ms → 20 QPS single-thread → 30K/20 = 1500 processes → 10 GPUs?"
- **Error**: Forgot problem states "150 QPS per GPU" (batching already factored in!)
- **Correct**: 30,000 QPS / 150 QPS per GPU = **200 GPUs**

**Q9 Example**:
- Given: XGBoost (1K candidates to rank, 100ms) vs DNN (1K candidates to rank, 100ms)
- User answer: "Option A: 50 CPUs for 100ms. Not sure how to get QPS limitation"
- **User's correct insight (post-check)**: "1K predictions = 1 REQUEST, not 1K requests!"
- **Corrected calculation**:
  - XGBoost: 1 request (1K candidates) / 0.1s = **10 QPS per server** ✅
  - DNN: 1 request (1K candidates) / 0.1s = **10 QPS per GPU** ✅
  - Both have same throughput! Choose XGBoost for cost efficiency
- **Key lesson**: Ranking N candidates = 1 request (not N requests)

**Formula to memorize**:
```
Throughput (QPS) = Batch Size / Latency
Servers Needed = Total QPS / QPS per Server
Cost = Servers × $/hour × 8760 hours/year
```

**Strengths**:
- Data pipeline patterns (100%): Kafka→Flink→Redis perfect
- Bias handling (100%): Randomized holdout + IPS understood
- A/B testing (100%): Per-user randomization rationale
- Two-tower (90%): Pre-computation insight
- Airflow idempotency (100%): Perfect example (INSERT→UPSERT)
- Continuous batching (100%): Perfect ragged sequence explanation

**Weak Areas**:
- ❌ **GPU scaling from latency** (50%): Read problem statement carefully! Use given QPS if provided
  - **Action**: Practice: Given latency → calculate QPS → scale to total QPS → cost
- 🟡 **Throughput calculation** (70%): Know formula QPS = Batch Size / Latency
  - **Action**: For every model, calculate: latency, batch size, QPS, servers needed, cost
- 🟡 **VIF formula** (85%): Correct (1/(1-R²), threshold >10) but acknowledged uncertainty
  - **Action**: Quick review of VIF interpretation

**Practice Exercises Completed** (~2 hours):
- ✅ Mock interview: YouTube Recommendation System (60 min)
  - Clarifying questions (QPS, latency, scale, metrics)
  - Proposed 3-stage cascade (100M → 1K → 100 → 20)
  - Data pipeline: Kafka→Flink→Redis + Spark batch
  - Training: watch_time / video_duration, IPS, 5% randomized traffic
  - A/B testing: Per-user, 1-2 weeks, primary metric watch time
- ✅ Extended discussions:
  - Latency vs QPS relationship (batching increases throughput) (30 min)
  - Two-tower pre-computation (user vs item embeddings) (15 min)
  - A/B test sample size calculation (15 min)
  - Training data bias solutions (randomized traffic, IPS) (15 min)

**Assessment**:
- **Mock Interview Score**: 78/100 (B+)
  - Requirements clarification: 9/10 (A)
  - System architecture: 8/10 (B+)
  - ML problem framing: 7.5/10 (B+)
  - Data pipeline: 7/10 (B)
  - Model serving & deployment: 6.5/10 (C+) - Missing scaling calculations ❌
  - Monitoring & evaluation: 7.5/10 (B+)
  - Communication & trade-offs: 8/10 (B+)

**Action Items**:
- ❌ **CRITICAL: Practice throughput calculations** - QPS = Batch Size / Latency, every model every time
- ❌ **CRITICAL: Scaling pattern** - QPS → servers → cost for every component
- 🟡 Add to next practice: Name specific tools (Kafka not "streaming system")
- 🟡 Review VIF formula interpretation (quick 5 min refresh)

**Recommendation**: ✅ **Excellent fundamentals, need scaling calculation practice!** 89.5% knowledge check shows strong conceptual understanding (bias handling, data pipelines, A/B testing all perfect). Main gap: applying throughput formulas in novel scenarios. 

**For Day 21**: Focus on scaling calculations for every component, cost estimation, and specific tool names.

**ML System Design Readiness**: ~75-80% (up from 0% - Week 3 Day 6 first practice)
- Strong: Data pipelines (100%), bias handling (100%), A/B testing (100%), architecture intuition (90%)
- Good: Two-tower (90%), framing (87%), monitoring (87%)
- Needs work: Scaling calculations (50-70%), cost estimation (not practiced)
- Next: 2-3 more practices → 85%+ interview ready

---

## Day 21 Detailed Results (2025-11-17)

**Context**: Week 3, Day 7 - System Design Practice (Fraud Detection)

**Content Tested**:
- 70% Day 21: Fraud detection system (model selection, dynamic batching, features, Flink vs Snowflake, sliding windows, cold start, manual review)
- 30% Review: Residual plots (Day 13), Roofline model (Day 10), Throughput calculation (Day 20)

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | Model selection & cost analysis | 21 | 100% ✅ | Perfect - LogReg $600, XGBoost $6K, DNN $90K with/without batching |
| Q2 | Dynamic batching mechanics | 21 | 95% ✅ | Collection time 30-40ms consistent (600-800 requests), understood trade-offs |
| Q3 | Feature engineering (10 features) | 21 | 100% ✅ | All categories covered + suggested regional averages |
| Q4 | Flink vs Snowflake roles | 21 | 100% ✅ | Nailed correction - Flink reads Kafka (not Snowflake) |
| Q5 | Sliding vs tumbling windows | 21 | 100% ✅ | Perfect example (6 tx at 0:57-1:03) |
| Q6 | Cold start handling | 21 | 100% ✅ | Default values + indicators + XGBoost advantage explained |
| Q7 | Manual review workflow | 21 | 100% ✅ | 1 TPS, prioritize by amount×score, feedback loop |
| Q8 | Residual plots interpretation (review) | 13 | 100% ✅ | Perfect - Plot B: log(y), Plot C: x² feature |
| Q9 | Roofline model (review) | 10 | 100% ✅ | Formula perfect, memory-bound explanation clear |
| Q10 | Throughput calc (review) | 20 | 75% 🟡 | Got 240 QPS (correct!), via 12 GPUs not dynamic batching |

**Overall Score**: **97.0% (970/1000)** - A+ ⭐⭐⭐

**Score Breakdown**:
- Day 21 content (Q1-Q7): **99.3%** (695/700) - Outstanding!
- Review content (Q8-Q10): **91.7%** (275/300) - Excellent retention

**Q10 Learning Moment** ⭐:
- User calculated: "12 GPUs" to get 240 QPS (20 QPS × 12 = 240) ✅ Math correct
- **Missed optimization**: Dynamic batching gives 240 QPS with 1 GPU (saves $110K/year!)
- Key lesson: **Dynamic batching is COST OPTIMIZATION**, not just horizontal scaling
  - Without batching: 20 QPS/GPU → 12 GPUs → $120K/year
  - With batching: 240 QPS/GPU → 1 GPU → $10K/year
- This pattern appeared in BOTH Day 20 and Day 21 - critical for interviews!

**Follow-up Questions Answered**:
1. "Is each GPU a single core?" → Clarified: A100 has 6,912 CUDA cores (108 SMs × 64 cores), but treated as 1 processing unit
2. "How to derive 600-800 requests from 30-40ms?" → Batch size = QPS × Collection time (20K × 0.03s = 600)

**Strengths**:
- ✅ Cost analysis mastery (compared 3 options with/without batching)
- ✅ Feature engineering breadth (20 features: profile, aggregated, velocity, mismatch, cold start)
- ✅ Dynamic batching understanding (50× throughput improvement)
- ✅ Edge case reasoning (sliding windows catch fraud at window edges)
- ✅ Production patterns (cold start, manual review, A/B testing, monitoring)
- ✅ Architecture corrections internalized (Flink vs Snowflake clear)

**Improvement from Day 20**:
- Design score: 78/100 → 85/100 (+7 points!)
- Knowledge check: 91% → 97% (+6 points!)
- Feature engineering: 7/10 → 9.5/10 (+2.5)
- Cost analysis: 6/10 → 9/10 (+3)

**Weak Areas Resolved**:
- ❌ (Day 20) Throughput calculations → ✅ (Day 21) Can calculate with/without batching
- ❌ (Day 20) Cost estimation missing → ✅ (Day 21) Always includes $/year for every option
- ❌ (Day 20) Feature engineering implicit → ✅ (Day 21) 20 explicit features across all categories

**Action Items**:
- 🟡 **Minor**: Reinforce that dynamic batching is the FIRST optimization to consider (not adding more hardware)
- Otherwise: None! System design skills strong across two different domains (recommendations → fraud)

**Recommendation**: ✅ **System design mastery achieved!** 97% knowledge check + 85% design score + clear improvement trajectory (78 → 85) demonstrates pattern mastery. Can tackle different domains (recommendations vs fraud) with consistent quality.

**ML System Design Readiness**: ~85-90% (up from 75-80% after Day 20)
- Strong: Cost analysis (90%), feature engineering (95%), architecture (85%), data pipelines (100%), A/B testing (100%)
- Good: Dynamic batching (90%), scaling calculations (80%), edge cases (90%)
- Ready for: ML Engineer system design interviews at FAANG+ companies

**For Week 4**: Advanced RAG (Day 1-2), then gap re-assessment (Day 3-4). If 85%+ overall → start applying!

**Last Updated**: 2025-11-17

---

## Day 22 Detailed Results (2025-11-18)

**Context**: Week 4, Day 1 - Advanced RAG fundamentals (RRF, SPLADE, DPR, Cross-encoder, ColBERT, MMR, Lost-in-the-middle)

**Content Tested**:
- 75% Day 22 (9 questions): RRF formula/calculation, SPLADE vocabulary, DPR negatives, Cross-encoder trade-off, ColBERT storage, MMR formula, Lost-in-middle, Hybrid retrieval, Applied reasoning
- 25% Review (3 questions): MLE Gaussian, Covariance vs Correlation, Throughput calculation

**Question Breakdown**:

| Q# | Topic | Day | Score | Notes |
|----|-------|-----|-------|-------|
| Q1 | MLE Gaussian derivation (review) | 14 | 90% ✅ | Math perfect, minor notation (simga typos, but correct μ̂ and σ̂²) |
| Q2 | Covariance vs Correlation (review) | 13 | 75% 🟡 | Concept correct, missing formulas: Cov=E[(X-μx)(Y-μy)], Corr=Cov/(σx×σy) |
| Q3 | Throughput calculation (review) | 20 | 100% ✅ | Perfect: 1000ms/40ms=25 batches/sec, 25×128=3200 QPS |
| Q4 | RRF formula and calculation | 22 | 90% ✅ | Formula correct, math correct, **ranking backwards** (0.032>0.0307 → doc1=rank1, not rank2) |
| Q5 | SPLADE vocabulary mismatch | 22 | 100% ✅ | Perfect example: "primate in rain forest" vs "monkeys in jungle" |
| Q6 | DPR negative types | 22 | 100% ✅ | TWO types: in-batch (efficient, easier) + hard from BM25 (expensive, harder) |
| Q7 | Cross-encoder vs Bi-encoder | 22 | 100% ✅ | Two-stage logic perfect: fast retrieval → slow accurate reranking |
| Q8 | ColBERT storage cost | 22 | 100% ✅ | 1M×100×768×4=307 GB (100× larger than bi-encoder) |
| Q9 | MMR formula | 22 | 100% ✅ | **FIXED minus sign!** λ×Sim(q,d) - (1-λ)×max Sim(d,d'), perfect λ interpretation |
| Q10 | Lost-in-the-middle mitigation | 22 | 100% ✅ | 3 strategies: reorder, reduce docs, compress middle |
| Q11 | Hybrid retrieval benefits | 22 | 100% ✅ | Exact match (jargon) + semantic match (synonyms) |
| Q12 | Applied: 10M product search | 22 | 100% ✅ | Hybrid BM25+DPR, ColBERT too large, excellent SPLADE question |

**Overall Score**: 96.0% (960/1000) - A+

**Retention Analysis**:
- Day 22 content (Q4-Q12): 98.9% (890/900) - Outstanding first-day absorption!
- Review content (Q1-Q3): 88.3% (265/300) - Good retention with minor gaps

**Key Insights**:
- ✅ **RAG fundamentals mastered**: 98.9% on new content (RRF, SPLADE, DPR, reranking techniques)
- ✅ **User fixed MMR formula error**: Changed + to - (minus sign) correctly
- ✅ **Excellent applied reasoning**: Q12 (product search) showed strong decision-making
- ✅ **Consolidation success**: 4 "unsure" topics (dense, sparse, hybrid, contrastive) all solidified
- 🟡 **Minor gap**: RRF ranking interpretation (higher score = better rank)
- 🟡 **Review gap**: Covariance/correlation formulas (75%) - need quick refresh

**Strengths**:
- RRF formula mastery (can calculate by hand)
- DPR training understanding (two types of negatives)
- ColBERT storage trade-off (10-50× larger)
- MMR diversity reranking (corrected formula)
- Hybrid retrieval benefits (exact + semantic)
- Applied reasoning (when to use SPLADE vs hybrid)

**Weak Areas**:
- 🟡 **Covariance/correlation formulas** (75%): Needs quick refresh on Cov(X,Y)=E[(X-μx)(Y-μy)], Corr(X,Y)=Cov/(σx×σy)
- 🟡 **RRF ranking** (90%): Remember higher RRF score = better ranking (rank 1, not rank 2)

**Day 22 Topics Mastered** (11 total):
- 7 New: RRF, SPLADE, DPR, Cross-encoder, ColBERT, MMR, Lost-in-the-middle
- 4 Consolidated: Dense retrieval basics, Sparse retrieval (BM25), Hybrid retrieval, Contrastive learning

**Action Items**:
- None! All Day 1 concepts interview-ready
- Continue Day 2 topics: FiD (Gap Q177), GraphRAG, RAFT, Agentic RAG

**Recommendation**: ✅ **Excellent performance!** 96% with 98.9% on new content demonstrates strong RAG fundamentals absorption. User successfully consolidated 4 "unsure" topics and fixed MMR formula error. Ready for Day 2 advanced patterns (FiD, GraphRAG, RAFT).

**Week 4 Advanced RAG Progress**:
- Day 1: 11 topics mastered (baseline 21.3% weighted → estimated 40-45% after Day 1)
- Target by Day 2-3: 55-60% weighted overall, 82% high-priority
- On track for target achievement

---

## Day 23: Advanced RAG Day 2 (2025-11-19)

**Score**: 96.3% (A+)
**New Content**: 98.6% (7 questions: FiD, GraphRAG, RAFT, Agentic RAG, Multi-hop, Parent doc, PDF parsing)
**Review**: 91.7% (3 questions: Breusch-Pagan, f/g operators, ColBERT)

---

### Review Questions (30%)

**Q1: Breusch-Pagan Test** - **75%** 🟡
**User's Answer**: "For detect non-homoscedaticity. H0: the residuals satisfy homoscedaticity condition, i.e. variance same range across different x."

**Scoring**:
- ✅ Correct: Detects heteroscedasticity (non-constant variance)
- ✅ Correct: H₀ is homoscedasticity
- ✅ Correct: H₁ implicit (negation of H₀) - **User caught scoring error!**
- ❌ Missing: Test procedure (regress e²ᵢ on X, test if R² > 0)

**Missing Detail**:
```
Breusch-Pagan Test Procedure:
1. Run regression: Y = Xβ + ε
2. Calculate squared residuals: e²ᵢ
3. Regress e²ᵢ on X: e²ᵢ = Xγ + u
4. Test: Is R² significantly > 0?
   - If yes: Reject H₀, heteroscedasticity present
   - If no: Fail to reject, assume homoscedasticity
```

---

**Q2: Forward/Backward Operators** - **100%** ✅
**User's Answer**: Detailed MLP example with Y = X @ A, Z = GeLU(Y) @ B, explained f (identity/all-reduce) and g (all-reduce/identity) with clear reasoning about column/row partitioning.

**Perfect!** User demonstrated deep understanding with concrete example and correct reasoning about why communication patterns differ in forward vs backward pass.

---

**Q3: ColBERT Late Interaction** - **100%** ✅
**User's Answer**: "ColBERT stores the full contextual embedding for entire sequence instead of just storing one embedding vector for the sequence. For (q, d) pair: for each q_i, find j that maximizes sim(q_i, d_j), ColBERT(d) = Sum_i maxSim(q_i, d_j). ColBERT storage will be 100x bi-encoder for seq length 100 without compression."

**Perfect!** Clear explanation of token-level embeddings, MaxSim formula, and 100× storage cost.

---

### New Questions - Day 2 Content (70%)

**Q4: FiD vs Long Context (2025)** - **100%** ✅
**User's Answer**: "Choose FiD for one-shot/high QPS scenario where prefix caching not available; large document count where the startup O(N^2) attention won't scale/lost-in-the-middle problem is severe. Choose long-context model for simple architecture, no tuning; multi-turn conversation where prefix caching helps a lot; moderate document count"

**Perfect!** User captured all 2025-specific trade-offs:
- ✅ FiD: single-shot, high QPS, O(n²) scaling, lost-in-middle
- ✅ Long context: prefix caching, multi-turn, simple architecture
- ✅ Mentioned prefix caching advantage (critical 2025 update)
- ✅ Mentioned lost-in-the-middle problem

**Key Insight Demonstrated**: User understands prefix caching makes long context competitive for multi-turn use cases.

---

**Q5: GraphRAG Local vs Global** - **100%** ✅
**User's Answer**: "Local search: for specific/entity center question: search the entity and trace its neighboring relations, return the neighbors. Example: what did Trump do in 2022? Global search: for holistic question/summary: search the community summaries, generate partial response, combine to get global response. Example: what are the 5 major themes?"

**Perfect!** User captured:
- ✅ Local: entity-centric, graph traversal, good example
- ✅ Global: holistic/themes, community summaries, map-reduce, good example
- ✅ Mentioned community summaries (Microsoft's key innovation)

---

**Q6: RAFT Training Format** - **100%** ✅
**User's Answer**: "P: query, golden answer doc, multiple distractor docs, answer with CoT reasoning. 1-P: query, multiple distractor docs, answer with CoT reasoning. 1-P examples nudges the model to internalize domain knowledge instead of solely rely on retrieved doc. Typically P=0.8"

**Perfect!** User nailed all components:
- ✅ P% examples: golden + distractors + CoT
- ✅ (1-P)% examples: distractors only + CoT
- ✅ Why (1-P): internalize knowledge, not solely rely on retrieval
- ✅ Typical P = 0.8

---

**Q7: Multi-Hop Implementations** - **100%** ✅
**User's Answer**: Three implementations with trade-offs - Query decomposition (interpretable, not flexible), Agentic RAG (flexible, high latency), GraphRAG (fast/interpretable, high setup/maintenance cost)

**Perfect!** User demonstrated:
- ✅ Three implementations correctly identified
- ✅ Clear pros and cons for each
- ✅ Understanding that multi-hop is a capability, not single implementation

---

**Q8: Agentic RAG ReAct Loop** - **90%** 🟡
**User's Answer**: Clear example with CEO book recommendation query, showing Thought → Action → Observe loop with 2 retrieve steps. Mentioned 3-5 rounds typical.

**Very good!** Minor gap:
- ✅ ReAct loop structure clear
- ✅ Good concrete example
- ✅ Mentioned 3-5 rounds
- ⚠️ Missing: Explicit statement "3-5× more expensive than traditional RAG" (implied but not stated)

**Improvement**: Could add explicit cost comparison: "Traditional RAG uses 1-2 LLM calls, Agentic uses 3-10+ calls, making it 3-5× more expensive."

---

**Q9: Parent Document Retrieval** - **100%** ✅
**User's Answer**: "Vector store: stores the small chunks for retrieval: (chunk embedding, parent id, chunk text). Parent doc store: stores the parent doc text (whole doc or bigger chunk), indexed by id. Retrieval flow: search in vector store -> get small chunks and parent ids -> retrieve parent docs. Critical detail: dedupe the parent docs before return"

**Perfect!** All components covered:
- ✅ Two storage systems (vector store + parent doc store)
- ✅ Retrieval flow correct
- ✅ Critical detail: deduplication

---

**Q10: PDF Parsing vs Multi-Modal** - **100%** ✅
**User's Answer**: "Parse to markdown: may lose complex formatting and visual info in charts. Keep as images: hard to index. I would choose hybrid: parse to markdown for easy retrieval, keep images that can be returned for best context"

**Perfect!** User demonstrated:
- ✅ Understanding of parse vs multi-modal trade-offs
- ✅ Hybrid approach for financial reports (best choice!)
- ✅ Reasoning: parse for retrieval + images for context

---

### Summary

**Strengths** (98.6% on new content):
- ✅ FiD vs long context (100%): All 2025 trade-offs (prefix caching, lost-in-middle, O(n) vs O(n²))
- ✅ GraphRAG (100%): Both modes (local + global) with community summaries
- ✅ RAFT (100%): Complete training scheme and rationale
- ✅ Multi-hop (100%): Implementations and trade-offs clear
- ✅ Parent doc (100%): Architecture and deduplication
- ✅ PDF parsing (100%): Hybrid approach for financial reports
- ✅ **Gap Q177 (FiD) closed**: 0% → 100% ✅

**Minor Gaps**:
- 🟡 Breusch-Pagan (75%): Missing test procedure (regress e²ᵢ on X, test R²)
- 🟡 Agentic RAG cost (90%): Implied 3-5 rounds but could state "3-5× more expensive" explicitly

**Critical Thinking Highlight**:
- ⭐ **User caught scoring error**: Correctly identified that H₁ is implicit negation of H₀ in hypothesis testing - excellent critical evaluation!

**Topics Mastered Today**:
- 7 New: FiD, GraphRAG, RAFT, Agentic RAG, Multi-hop implementations, Parent document retrieval, PDF parsing strategies
- 3 Consolidated: Multi-modal RAG, Standard RAG pipeline, Query decomposition

**Action Items**:
- Quick Breusch-Pagan refresh (2 min): Review test procedure (regress e²ᵢ, test R²)
- Optional Day 3: Evaluation metrics (Recall@K, Precision@K, MRR, NDCG) if want to reach 82% high-priority target

**Recommendation**: ✅ **Exceptional performance!** 96.3% overall with 98.6% on new Day 2 content demonstrates mastery of advanced RAG patterns. User successfully closed Gap Q177 (FiD: 0%→100%) and demonstrated strong systems thinking (prefix caching trade-offs, implementation comparisons). All 7 topics are interview-ready. Ready for optional Day 3 (evaluation metrics) or to proceed with Week 4 progress assessment.

**Week 4 Advanced RAG Progress**:
- Day 1: 11 topics mastered (21.3% → ~40-45% weighted)
- Day 2: 7 topics + 3 consolidations mastered (~40-45% → **52-55% weighted**, **68-72% high-priority**)
- Target by Day 3: 82% high-priority, 55-60% overall
- Current: **On track** - 2 days achieved 18 topics total, excellent absorption rate

**Last Updated**: 2025-11-19

---

## Day 26 Knowledge Check (2025-11-22)

**Format**: Review-only session (10 overdue topics from knowledge schedule)
**Overall Score**: **80.0%** (800/1000)

### Review Questions (10 overdue topics)

**Q1: Shapiro-Wilk Test** - **100%** ✅
**User's Answer**: "Shapiro-Wilk checks normality assumption, H0: data comes from normal distribution. Low p-value indicate the normality assumption doesn't hold, -- likely data not from normal distribution."

**Perfect!** User correctly identified:
- ✅ Tests normality assumption
- ✅ H0: data from normal distribution
- ✅ Low p-value → reject normality

---

**Q2: Chi-Square Test** - **75%** 🟡
**User's Answer**: 
- Goodness of fit: whether observed data matches alleged distribution (H0: they match)
- Test of independence: whether a slice variable affects distribution (H0: independent)
- "But I forgot how to calculate the test statistics. Take the difference for each pair of corresponding cell in table, and? dof is (row_count -1) * (col_count - 1)?"
- "not sure assumptions; independent observations or what?"

**Good conceptual understanding!** Gaps in details:
- ✅ Goodness of fit and independence explanations correct
- ✅ DOF formula (r-1)(c-1) correct
- ❌ Missing test statistic: Σ(O-E)²/E
- ⚠️ Assumptions: independent observations ✅, expected counts ≥5, categorical data

**Interview-ready answer**: "Chi-square test: (1) Goodness of fit tests if observed data matches expected distribution, (2) Test of independence tests if two categorical variables are independent. Test statistic: Σ(O-E)²/E. DOF: (r-1)(c-1). Assumptions: independent observations, expected counts ≥5, categorical data."

---

**Q3: Common Distributions** - **100%** ✅
**User's Answer**: Perfect explanations for all 5 distributions:
- Normal: bell curve, PDF with exp(-1/2*(x-μ/σ)²)
- Binomial: C(n,k) p^k (1-p)^(n-k), mean np, variance np(1-p)
- Geometric: (1-p)^(k-1)*p, mean 1/p
- Poisson: λ^k exp(-λ)/k!, mean λ, variance λ
- Exponential: λ exp(-λx), mean 1/λ

**Excellent!** All PDFs, means, and variances correct.

---

**Q4: CLT vs LLN** - **100%** ✅
**User's Answer**: 
- CLT: distribution of sample mean approaches normal as n→∞, used for testing mean without knowing underlying distribution
- LLN: sample mean approaches population mean as n→∞, used to estimate population mean from sample

**Perfect distinction and applications!**

---

**Q5: Regularization as Bayesian Prior** - **100%** ✅
**User's Answer**: Excellent MAP derivation showing:
- Gaussian prior → exp(-1/2σ² * w²) → L2 regularization (λ Σw²)
- Laplace prior → (forgot PDF) but knew it leads to L1 regularization (Σ|w|)

**Perfect conceptual understanding!** Laplace PDF is (1/2b)exp(-|w-μ|/b), but user got the key insight.

---

**Q6: Speculative Decoding** - **100%** ✅
**User's Answer**: 
1. Use faster but less accurate model to predict k tokens
2. Larger model batch scores the k tokens, check if match
3. If mismatch, take larger model's prediction, send back to smaller model
- Speedup: 2-3× typically, conditions: faster model fast enough, mismatch not very frequent, good for small batch

**Perfect 3-step process, speedup conditions, and typical range!**

---

**Q7: FSDP Stream Sync + Prefetching** - **50%** 🟠 **WEAK ITEM (3rd review, EF 1.45)**
**User's Answer**: "While the previous layer is doing computation, parameter gathering for the next layer can already start/prefetching? Unsure about stream synchronization"

**Partial understanding**:
- ✅ Prefetching concept correct (next layer params during current layer compute)
- ❌ Missing: Stream synchronization - FSDP uses separate CUDA streams for all-gather (comm) and forward/backward (compute) to overlap them. Stream sync ensures all-gather completes before forward pass uses parameters.

**Interview-ready answer**: "FSDP overlaps communication and computation using separate CUDA streams. During forward pass: (1) All-gather runs on comm stream to fetch next layer's params, (2) Current layer computation runs on compute stream, (3) Stream sync ensures all-gather completes before next layer starts. Prefetching fetches params for layer N+1 while computing layer N, hiding communication latency."

**Action**: Review CUDA streams and async communication concepts. This is the 3rd review with declining scores (65%→50%), EF dropped to 1.45 (struggling item).

---

**Q8: Durbin-Watson Test** - **75%** 🟡
**User's Answer**: "Durbin-Watson checks independence assumption, H0: data points are independent. 2 is perfect? 0 is strong negative dependency? 4 is strong positive dependency?"

**Correct concept, reversed interpretation!**
- ✅ Checks autocorrelation (independence assumption)
- ✅ 2 = no autocorrelation
- ❌ Reversed: 0 = **positive** autocorrelation (errors follow same pattern), 4 = **negative** autocorrelation (errors alternate)

**Mnemonic**: DW = 0 → ρ = +1 (positive correlation), DW = 4 → ρ = -1 (negative correlation)

---

**Q9: SPLADE** - **100%** ✅
**User's Answer**: 
- SPLADE uses BERT encoder with MLM head for middle ground embedding (sparse + dense)
- Includes not only original token but also synonyms/rephrasing
- FLOPS regularization encourages sparsity, nudging weak entries to 0
- Solves vocabulary mismatch problem (e.g., "primates in rain forest" vs "monkeys in jungle")

**Perfect!** All components covered: learned sparse, FLOPS regularization, vocabulary expansion problem.

---

**Q10: Communication Patterns (DP/TP/PP)** - **100%** ✅ **JUST STUDIED DAY 25**
**User's Answer**: 
- DP/FSDP: 2P per step (P for sending out each node's gradients, P for receiving aggregated gradients)
- TP: 4 all-reduce per layer, each communicates B×S×H, so total 8BSH per layer
- PP: B×S×H per stage boundary (B is micro batch size)

**Perfect recall from Day 25 study!** All formulas correct:
- ✅ DP: 2P (reduce-scatter + all-gather)
- ✅ TP: 8BSH per layer (4 all-reduce × 2× ring overhead)
- ✅ PP: BSH per boundary

**Excellent payoff from Day 25 deep dive!** 75%→100% in 2 days.

---

### Summary

**Overall Score**: **80.0%** (800/1000)

**Distribution**:
- ✅ **100%**: 7 questions (Shapiro-Wilk, Distributions, CLT/LLN, Regularization priors, Speculative decoding, SPLADE, Communication patterns)
- 🟡 **75%**: 2 questions (Chi-square test statistic, Durbin-Watson 0/4 interpretation)
- 🟠 **50%**: 1 question (FSDP stream sync - still weak after 3rd review)

**Strengths**:
- ✅ Statistics fundamentals (distributions, CLT/LLN, Shapiro-Wilk) - perfect recall
- ✅ LLM systems (speculative decoding, communication patterns) - Day 25 study paid off!
- ✅ RAG (SPLADE) - strong retention
- ✅ ML theory (Bayesian priors) - excellent derivation

**Weak Areas**:
- 🟠 **pytorch_fsdp_internals** (50%, 3rd review, EF 1.45→ reset to tomorrow):
  - Persistent weak item: 65% (Day 16) → 50% (Day 26, 10 days later)
  - Missing CUDA streams concept and async communication
  - Action: Study CUDA streams, stream synchronization, async all-gather
- 🟡 **stats_3.8** Chi-square (75%): Missing test statistic formula Σ(O-E)²/E
- 🟡 **stats_4.3** Durbin-Watson (75%): Reversed 0/4 interpretation (minor detail)

**Key Observations**:
- **Communication patterns payoff**: 75% (Day 25) → 100% (Day 26) - Day 25 deep dive worked!
- **FSDP continues to struggle**: 3rd review (65%→50%), EF dropped to 1.45, reset to review tomorrow
- **Strong retention**: 7/10 perfect scores on overdue topics (10-12 days old)
- **Weekend review format**: Review-only session appropriate for limited weekend time

**Recommendation**: ✅ **Good weekend review session!** 80% (B/B+) shows solid retention on most topics. FSDP remains the only persistent weak item - needs focused study on CUDA streams and async communication patterns. Consider dedicating 15-20 minutes tomorrow to review CUDA programming basics and PyTorch distributed implementation details. All other topics show strong retention and are interview-ready.

**SM-2 Updates**: 10 topics updated in knowledge-schedule.md:
- 7 topics scored 100% → increased EF, extended intervals (6-16 days)
- 2 topics scored 75% → decreased EF, 6-13 day intervals
- 1 topic scored 50% → failed, EF dropped to 1.45, reset to review tomorrow (2025-11-23)

**Next Steps**:
- Tomorrow: Review FSDP (scheduled 2025-11-23 after reset)
- Continue Day 5-7 plan: Project selection and setup
- Optional: Quick refreshers on chi-square test statistic and Durbin-Watson interpretation


---

## Day 27: Nov 23, 2025 (Weekend Review - 10 Overdue Topics)

**Score**: 87.5% (875/1000) - A-/B+ (7 perfect, 1 failed, 2 partial)

**Format**: Review-only (100% review questions from overdue items in knowledge schedule)

**Topics Reviewed**:
- PyTorch basics (3 questions): nn.Module, no_grad, BCELoss
- ML fundamentals (2 questions): Adam optimizer, Regression metrics  
- ML Infrastructure (3 questions): Airflow executors, Docker multi-stage, K8s Deployment
- System Design (1 question): GPU scaling calculation
- PyTorch advanced (1 question): FSDP internals

---

### Q1: pytorch_nn_module (10 days overdue) - 75%

**Question**: What are the three key components you must implement when defining a PyTorch model with `nn.Module`? What is the alternative shorthand for simple sequential architectures?

**Your Answer**:
- Need to implement `forward()`, what else? pick loss function and optimizer (but that's not part of defining the nn.Module?) `__init__()`? 
- Do you mean use `nn.Sequential()` with a list of existing `nn.Module`?

**Score**: 75% 🟡

**Assessment**: Got `__init__` and `forward`, but **missed the 3rd component: `super().__init__()` call**. Sequential explanation was on track but uncertain.

**Correct Answer**:
1. **`__init__()`**: Define layers/parameters
2. **`forward()`**: Define forward pass computation  
3. **`super().__init__()`**: Must call in your `__init__` method

**Alternative shorthand**: `nn.Sequential()` - takes list of modules, automatically chains them

```python
# Full nn.Module
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # ← This is the 3rd component!
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Sequential shorthand
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
```

You're right that loss function/optimizer aren't part of `nn.Module` definition!

---

### Q2: pytorch_no_grad (10 days overdue) - 100%

**Question**: What does the `torch.no_grad()` context manager do, and when should you use it? What happens to memory and computation inside this context?

**Your Answer**:
It won't back-prop the grad. Used during evaluation/inference. Less memory and computation -- no backprop, no gradient and optimizer state tracking.

**Score**: 100% ✅

**Assessment**: Perfect! Covered all key points: disables autograd, use during eval/inference, reduces memory (no gradients stored), reduces computation (no gradient tracking).

---

### Q3: pytorch_bce_loss (10 days overdue) - 100%

**Question**: What's the difference between `BCELoss` and `BCEWithLogitsLoss` in PyTorch? Which should you use and why?

**Your Answer**:
- BCELoss: binary cross entropy loss on output in (0,1) range
- BCEWithLogitsLoss: binary cross entropy loss on logit output in (-inf, inf) range
- Use BCELoss when the last layer already applied Sigmoid. Use BCEWithLogitsLoss if last layer outputs logit.

**Score**: 100% ✅

**Assessment**: Perfect! Clear understanding of when to use each, input range differences, and the relationship to sigmoid activation.

**Additional note**: BCEWithLogitsLoss is generally preferred because it's numerically more stable (combines sigmoid + BCE in one operation).

---

### Q4: ml_adam_optimizer (9 days overdue) - 50%

**Question**: What is Adam optimizer's bias correction mechanism? Why is it needed, and what are the formulas for the corrected first and second moments?

**Your Answer**:
Bias correction makes the grad update value larger for the initial steps, otherwise the initial steps will be very slow.
- Momentum: grad = beta_1 * old_grad + (1-beta_1) * new_grad or something?
- Variance: forgot the formula
- Forgot bias correction formula

**Score**: 50% 🟠 (Failed - reset to n=1, review tomorrow)

**Assessment**: **Concept correct** (bias correction addresses slow initial updates), but **formulas missing/incorrect**. This is a critical gap for ML fundamentals.

**Correct Answer**:

**Without bias correction**, first moment starts at 0, so:
- Step 1: m₁ = 0.9×0 + 0.1×grad = 0.1×grad (too small!)
- Step 2: m₂ = 0.9×(0.1×grad) + 0.1×grad = 0.19×grad (still small)

**Adam formulas**:
```
# Update first and second moments
m_t = beta_1 * m_{t-1} + (1 - beta_1) * grad
v_t = beta_2 * v_{t-1} + (1 - beta_2) * grad²

# Bias correction (this is the key!)
m_hat = m_t / (1 - beta_1^t)
v_hat = v_t / (1 - beta_2^t)

# Parameter update
theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
```

**Why it works**:
- At t=1: `1 - beta_1^1 = 1 - 0.9 = 0.1`, so `m_hat = 0.1×grad / 0.1 = grad` ✅
- At t=10: `1 - beta_1^10 ≈ 0.65`, correction factor diminishes
- At t→∞: `1 - beta_1^t → 1`, no correction needed

**Action**: Review this tomorrow (scheduled for 2025-11-24)

---

### Q5: ml_regression_metrics (8 days overdue) - 75%

**Question**: Explain the differences between MSE, RMSE, MAE, and R². When would you prefer MAE over RMSE?

**Your Answer**:
- MSE: min squared error, Sum (y-y_true)² / n
- RMSE: square root of MSE (same unit as data)
- MAE: absolute error? Sum |y-y_true| / n? Treating error amount equally (squared error penalize larger residual more)? More robust on outlier?
- R²: measures how much variation explained by the linear regression or something like that? Forgot formula

**Score**: 75% 🟡

**Assessment**: **All concepts correct!** MSE/RMSE/MAE definitions and MAE robustness reasoning perfect. Just missing R² formula.

**R² formula**:
```
R² = 1 - (SS_res / SS_tot)

Where:
- SS_res = Σ(y - y_pred)²  (residual sum of squares)
- SS_tot = Σ(y - y_mean)²  (total sum of squares)
```

**Interpretation**:
- R² = 0.85 → Model explains 85% of variance, 15% unexplained
- R² = 1.0 → Perfect fit
- R² = 0.0 → Model no better than predicting mean

**When to use MAE over RMSE**: ✅ You got it - MAE more robust to outliers (doesn't square the errors)

---

### Q6: airflow_executors (8 days overdue) - 100%

**Question**: Compare the three Airflow executors: LocalExecutor, CeleryExecutor, and KubernetesExecutor. What are the use cases for each?

**Your Answer**:
- LocalExecutor: execute locally, share resource with scheduler. Used for development / simple and cheap flow
- CeleryExecutor: have a dedicated remote worker pool, send task to the pool to execute. Efficient for expensive tasks; high cost if idle, tasks compete with each other for resources
- KubernetesExecutor: each task containerized in a pod and dispatched in Kubernetes. Isolated task, cost efficient; higher startup latency

**Score**: 100% ✅

**Assessment**: Perfect! All three executors, their architectures, use cases, and trade-offs clearly explained.

**Key trade-offs captured**:
- LocalExecutor: Simple but limited scalability
- CeleryExecutor: Scalable but always-on cost
- KubernetesExecutor: Cost-efficient + isolated, but higher startup latency

---

### Q7: pytorch_fsdp_internals (due TODAY, EF=1.45, weak item) - 100%

**Question**: Explain how FSDP overlaps communication and computation using CUDA streams and prefetching. What gets prefetched, and when does the all-gather happen?

**Your Answer**:
One CUDA stream for computation and one CUDA stream for gathering the parameter values. Overlap: start prefetching parameters for L+1 while doing computation for L. Sync to ensure forward computation for L+1 only start after parameter all-gather is completed.

**Score**: 100% ✅ **BREAKTHROUGH!**

**Assessment**: **Perfect explanation!** You nailed:
- Two CUDA streams (one for compute, one for all-gather)
- Prefetching mechanism (parameters for layer L+1 while computing layer L)
- Synchronization requirement (all-gather must complete before forward pass)

**This is your 4th review**:
- Review 1 (Day 16): 65% - Partial understanding
- Review 2 (Day 26): 50% - Still struggling (EF dropped to 1.45)
- Review 3 (Day 26, later): 50% - Reset to review tomorrow
- Review 4 (Day 27, today): **100%** - Fully mastered! 🎉

**EF recovered**: 1.45 → 1.55 (still below 2.0, but improving)
**Next review**: 2025-11-29 (6 days)

---

### Q8: docker_multistage_builds (7 days overdue) - 75%

**Question**: What are Docker multi-stage builds? Give an example with a build stage and a runtime stage. Why does this reduce image size?

**Your Answer**:
Separate build-stage and run-stage in Docker builds, so the final distributed run-stage image won't need the build dependencies.
Example: e.g. pulling in compiler dependency to convert code to C++ runtime in build-stage; run-stage only need runtime dependencies? (not so sure)

**Score**: 75% 🟡

**Assessment**: **Concept clear** (separate build/runtime stages, final image excludes build deps), but example was vague/uncertain.

**Better example**:
```dockerfile
# Build stage - includes compiler, build tools
FROM python:3.10 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt
COPY . .
RUN python -m compileall .  # Pre-compile Python bytecode

# Runtime stage - only runtime dependencies
FROM python:3.10-slim  # ← Much smaller base image!
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

**Why smaller**:
- Build stage: 1 GB (includes pip cache, compiler, intermediate files)
- Runtime stage: 200 MB (only final artifacts + slim base image)
- **Final image = runtime stage only!**

---

### Q9: k8s_pod_deployment_service (7 days overdue) - 75%

**Question**: Explain the difference between Pod, Deployment, and Service in Kubernetes. How does a Deployment use ReplicaSets?

**Your Answer**:
- Pod: a minimum unit of work to be scheduled
- Deployment: declarative spec to deploy the work, declare the number of replicas and resources
- Service: abstract on top of deployment, providing stable IP address
- (unsure) ReplicaSets have deployment run on multiple nodes?

**Score**: 75% 🟡

**Assessment**: **Main concepts correct**, but ReplicaSet answer uncertain.

**Correct relationship**:
```
Deployment (you create this)
    ↓ creates/manages
ReplicaSet (auto-created by Deployment)
    ↓ creates/manages
Pods (3 replicas running on different nodes)
```

**Full picture**:
- **Pod**: Single instance of your app (can have multiple containers)
- **ReplicaSet**: Ensures N pods are running, replaces failed pods
- **Deployment**: Higher-level abstraction, manages ReplicaSets
  - Rolling updates: Creates new ReplicaSet, gradually scales down old one
  - Rollback: Switches back to previous ReplicaSet
- **Service**: Stable IP/DNS, load balances across pod replicas

**Example**:
```yaml
# You create Deployment with 3 replicas
Deployment: my-app (replicas: 3)
  → Creates ReplicaSet: my-app-abc123
      → Creates 3 Pods: my-app-abc123-pod1, pod2, pod3
```

---

### Q10: sysdesign_gpu_scaling (6 days overdue) - 100%

**Question**: Given a model with 50ms inference time and batch size 32, what QPS can one GPU handle? If total load is 10,000 QPS, how many GPUs are needed?

**Your Answer**:
20 batch/sec × 32 = 640 QPS.
10000/640 = 1000/64 = 250/16 = 125/8 ≈ 16 GPUs

**Score**: 100% ✅

**Assessment**: **Perfect calculation!** 
- QPS per GPU: 32 / 0.05s = 640 QPS ✅
- GPUs needed: 10,000 / 640 = 15.625 → 16 GPUs ✅

**Formula applied correctly**: QPS = Batch Size / Inference Time

This is a huge improvement from your first attempt (Day 20: 50% - missed given numbers in problem). You've mastered reading problem statements and applying the throughput formula!

---

## Summary - Day 27

**Overall Score**: 875/1000 = **87.5% (A-/B+)**

**Strong Areas**:
- ✅ PyTorch basics: no_grad (100%), BCELoss (100%)
- ✅ **FSDP internals: 50% → 100%** - Major breakthrough after 4 reviews!
- ✅ Airflow executors: 100% (perfect trade-off understanding)
- ✅ GPU scaling calculation: 100% (50% → 100% improvement since Day 20)
- ✅ Regression metrics: All concepts correct, just missing R² formula

**Weak Areas**:
- ❌ **Adam optimizer (50%)**: Concept understood, but formulas missing - Critical gap for ML fundamentals
  - Failed review (score ≤60%)
  - **Reset to n=1, I=1 day**
  - **Action**: Review tomorrow (2025-11-24)
- 🟡 nn.Module (75%): Forgot `super().__init__()` call
- 🟡 Docker multi-stage (75%): Concept clear, example vague
- 🟡 K8s Deployment (75%): Main concepts right, ReplicaSet relationship uncertain

**Key Achievement**: 🎉 **FSDP Internals Mastered!**
- This was your weakest item (EF=1.45, lowest in schedule)
- 4 review attempts: 65% → 50% → 50% → **100%**
- Persistence paid off - you fully understand CUDA stream overlap now!

**Next Steps**:
1. **Tomorrow (Day 28)**: Review Adam optimizer formulas (due 2025-11-24, critical for ML interviews)
2. Continue with RAG project planning (Day 27 task: Download ArXiv papers, 30 min)
3. Next review batch: 6 topics due 2025-11-29 (PyTorch basics, Docker, K8s, GPU scaling)

**SM-2 Schedule Updates**: All 10 items updated with new EF, intervals, and next review dates

**EF Changes**:
- pytorch_fsdp_internals: 1.45 → 1.55 (improving!)
- ml_adam_optimizer: 2.5 → 1.96 (failed, needs attention)
- sysdesign_gpu_scaling: 2.14 → 2.24 (50% → 100% recovery)
- 7 other items: Minor EF adjustments based on scores

**Recommendation**: Excellent review performance! FSDP breakthrough is a major milestone. Focus on Adam formula review tomorrow, then proceed with RAG project.

