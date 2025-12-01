# ML Engineer Interview Prep - 12 Week Study Plan

**Timeline**: 2-3 months
**Time Commitment**: 5-10 hours/week
**Total Hours**: 60-120 hours
**Target Role**: ML Engineer
**Background**: Senior SWE with production ML experience + ML certifications (knowledge faded, needs refresh)

---

## Overview

This plan is optimized for **passing ML Engineer interviews** at top tech companies. Focus is on:
- **Refreshing implementation skills** (you know theory, need practice coding)
- Building interview-ready portfolio projects
- Mastering ML system design
- Sharpening coding and theory questions

**Adjustment from generic plan**: Since you've completed certifications, you'll move faster through lectures and spend MORE time on hands-on coding and projects.

---

## Phase 1: Foundations Refresh (Weeks 1-4)

**Goal**: Refresh ML fundamentals and build confidence through hands-on projects
**Time**: 20-40 hours total

### Week 1: Assessment + Fast.AI Kickstart (CUSTOMIZED FOR YOU)

**NEW: Assessment Phase (Day 1-2, 2-3 hours)**:

Since you've completed certifications but feel rusty, start with a reality check:

**Day 1 - Coding Assessment (COMPLETED)** ✅:
- [X] Implemented Linear Regression from scratch (~1 hour)
  - Correct implementation with minor issues (forgot averaging in loss)
  - Good vectorization and code structure
- [X] Implemented Logistic Regression from scratch (~1 hour)
  - Correct implementation with gradient dimension bug
  - Had to look up cross-entropy formula
  - Struggled with derivative computation
- [X] Identified rust areas:
  - 🔴 **Gradient derivations** (HIGH priority) - took most time
  - 🟡 Matrix dimensions (medium) - small bugs with X.T @ error
  - 🟡 Loss function details (medium) - forgot averaging
  - ✅ Numpy operations (good)
  - ✅ Algorithm structure (good)

**Day 1 Key Learnings**:
- Implementation time: 1 hour each (target: 20-25 min for interviews)
- Need to memorize common gradient formulas
- Foundation is solid, just need to build speed through practice

---

**Day 2 - Speed Practice (REVISED for 1-2 hours)**:

**Goal**: Prove you can implement faster with muscle memory

**Session 1 - Re-implement Logistic Regression (45-60 min)** ✅ **COMPLETED**:
- [X] Implement Logistic Regression again from scratch
  - Target: 30-40 minutes → **Actual: 15 minutes!** 🎉
  - Didn't look at yesterday's code ✅
  - Got gradient right first time: `X.T @ (yP - y) / n` ✅
  - Remembered to average the loss: `loss / n` ✅
  - Added numerical stability: `np.clip(yP, epsilon, 1-epsilon)` ✅
- [X] Compare with yesterday's implementation
  - **4x speed improvement**: 60 min → 15 min ✅
  - Gradient correct this time ✅
  - Loss function has averaging ✅
  - **Beat sklearn accuracy**: 98.2% vs 97.4% ✅

**Session 2 - Quick Reference (30-45 min if time)**:
- [X] Create personal gradient cheat sheet (see Gradient-Formulas-Cheatsheet.md)
- [X] Quick skim of ML-Coding-Questions.md
- [X] List 3 algorithms to practice this week: K-NN, K-Means, (Optional: Decision Tree)

**Day 2 Success Criteria** ✅ **ALL ACHIEVED**:
- ✅ Implemented LogReg in **15 min** (exceeded 30-40 min target!)
- ✅ Got gradient formula right without looking it up
- ✅ Built muscle memory for the pattern
- ✅ **Bonus**: Achieved 98.2% accuracy, beat sklearn

**Day 2 Results Summary** (2025-10-29):
- ⏱️ **Time**: 15 minutes (beat target by 2x)
- 🎯 **Accuracy**: 98.2% (beat sklearn's 97.4%)
- 📈 **Speed improvement**: 4x faster than Day 1
- 💯 **All formulas correct** on first try
- ✅ **Interview-ready**: Well within 20-25 min target

**Confidence Level Updated**: **HIGH** - Ready to move forward
- Day 2 proved rust was superficial
- Muscle memory returns quickly with practice
- Can accelerate through remaining material

---

**Day 3-7**: Ready to proceed as planned

---

**Day 3 - Hybrid Approach (2 hours)** ✅ **Session 1 COMPLETE**:

**Session 1: Theory Refresh** (1 hour) ✅:
- [X] Reviewed ML-Theory-Questions.md - identified gaps
- [X] Watched optimization videos (Momentum, RMSprop, Adam) - 30 min
- [X] Watched AUC-ROC video (StatQuest) - 15 min
- [X] Watched Attention mechanism video (StatQuest) - 15 min
- [X] Created Day3-Quick-Reference.md with consolidated notes

**Key learnings**:
- Adam optimizer: Combines momentum + RMSprop, bias correction for early iterations
- AUC-ROC: TPR vs FPR, use for moderate imbalance, not extreme
- Attention/Transformers: Q-K-V mechanism, BERT vs GPT parallel processing
- Connected to production ML experience (transormers, LLMs)

**Theory Gaps Identified** 🟡:
- Refreshed: Optimizers (Adam, momentum), AUC-ROC, Attention/Transformers ✅
- Still need: Boosting, Elastic Net, kernel trick, batch norm details (lower priority)
- Can skip: Reinforcement learning (unless RL-specific role)

**Session 2: K-NN Implementation** (50 min) ✅ **COMPLETED**:
- [X] Implemented K-Nearest Neighbors from scratch
- [X] Tested on iris dataset
- [X] Algorithm correct, clean code structure
- [X] Used efficient NumPy operations (argpartition, unique)
- [X] Learned NumPy broadcasting for vectorization

**Day 3 Results Summary** (2025-10-30):
- ⏱️ **Theory**: ~1 hour (optimizers, metrics, attention/transformers)
- ⏱️ **Implementation**: ~50 min (K-NN with interruptions)
- 💡 **Key learnings**: Adam bias correction, AUC-ROC vs Precision-Recall, Q-K-V attention
- 📝 **Created**: Day3-Quick-Reference.md with interview-ready answers
- 🔧 **Implementation**: Correct K-NN, good NumPy skills, Python scope practice needed

---

**Day 4 - Hybrid: Theory + K-Means (2 hours)** ✅ **COMPLETED**:

**Session 1: Theory Refresh** (45 min) ✅:
- [X] Regularization (L1/L2, Elastic Net) - 30 min
  - Watched StatQuest videos (Ridge, Lasso, Elastic Net)
  - L1 (Lasso): Sparse weights (exact 0), feature selection
  - L2 (Ridge): Asymptotic to 0, keeps all features
  - Elastic Net: Combines L1 + L2, better for correlated variables
  - Understanding: Why Elastic Net groups correlated features
- [X] Regression metrics - 15 min (reading, not videos)
  - MAE: Robust to outliers, equal penalty
  - RMSE: Penalizes large errors, same units as target
  - R²: Variance explained, range (-∞, 1]
  - R² limitations: Always increases with features → use Adjusted R²
- [X] Batch normalization - Deferred to Week 2-3 (DL-specific)

**Session 2: K-Means Implementation** (~40 min) ✅:
- [X] Implemented K-Means clustering from scratch
  - Random centroid initialization with `np.random.choice`
  - Assignment step using `np.argmin(distances, axis=1)`
  - Update step with `X[labels == k].mean(axis=0)`
  - Convergence check with `np.allclose`
  - Empty cluster handling (reinitialize randomly)
- [X] Tested on synthetic blob data
- [X] Compared with sklearn.cluster.KMeans
  - **Inertia match**: 212.006 (exact match!)
  - Centroids match (different label order, expected)

**Day 4 Results Summary** (2025-10-31):
- ⏱️ **Theory**: ~45 min (regularization 30 min + regression metrics 15 min)
- ⏱️ **Implementation**: ~40 min (K-Means)
- 💡 **Key learnings**:
  - Regularization: L1 vs L2 vs Elastic Net tradeoffs
  - Elastic Net better for correlated variables (grouping effect)
  - R² always increases with features → need Adjusted R²
  - K-Means sensitive to initialization (local minimum problem)
  - Python array assignment gotcha: need `.copy()`
- 📝 **Created**: Day4-Quick-Reference.md with regularization, metrics, K-Means
- 🔧 **Implementation**: Correct K-Means, matches sklearn exactly, efficient broadcasting

**Knowledge Check Results** (88% - B+/A- level):
- Implementation (LogReg, K-NN): 95% ✅
- Algorithm complexity: 100% ✅
- Optimizers: 80% 🟡 (minor detail on adaptive LR)
- Bias correction: 95% ✅
- Evaluation metrics: 75% 🟡 (reasoning clarity)
- Transformers: 85% ✅ (minor terminology)
- Attention: 90% ✅
- NumPy: 100% ✅

**Overall Progress**: Excellent retention, interview-ready on implementation, needs minor polish on explaining "why"

**Success Criteria**: ✅ All met
- ✅ Can explain when to use L1 vs L2 regularization
- ✅ Can explain difference between MAE and RMSE and R² limitations
- ✅ K-Means converges to stable clusters
- ✅ Results match sklearn exactly (inertia 212.006)

---

**Day 5 - Theory Only: Boosting + Parametric/Non-parametric (<1 hour)** ✅ **COMPLETED**:

**Context**: Weekend day with kids, limited time (~45 min)

**Session: Theory Refresh** (45 min) ✅:
- [X] AdaBoost (StatQuest) - 20 min
  - Stumps with "amount of say"
  - Sample weight adjustment
  - Formula: `0.5 * log((1-error)/error)`
- [X] Gradient Boost Part 1 (StatQuest) - 15 min
  - Builds trees on residuals
  - Learning rate scaling
  - Small trees (8-32 leaves)
- [X] Parametric vs Non-parametric - 5 min (reading)
  - Fixed parameters vs grows with data
  - Examples: LogReg/NN (parametric), K-NN/Decision Tree (non-parametric)
- [X] Knowledge check - 10 min

**Day 5 Results Summary** (2025-11-01):
- ⏱️ **Total time**: ~45 min (theory only, efficient weekend session)
- 💡 **Key learnings**:
  - Boosting: Sequential weak learners, focuses on errors
  - AdaBoost: Reweights samples, amount of say
  - Gradient Boost: Predicts residuals, learning rate
  - Parametric: Fixed parameters (LogReg, NN)
  - Non-parametric: Grows with data (K-NN, Decision Tree)
- 📝 **Created**: Day5-Quick-Reference.md with boosting + parametric/non-parametric
- 📊 **Knowledge Check: 98.5% (A+)** - Outstanding retention despite limited time
  - Perfect boosting understanding (99.3%)
  - Inertia resolved: 70% → 95% in 1 day! 🎉
  - Overall trend: +10.5% over 3 days (88% → 98.5%)

**Success Criteria**: ✅ All met
- ✅ Can explain boosting vs bagging
- ✅ Can explain AdaBoost algorithm with formulas
- ✅ Can explain Gradient Boost concept
- ✅ Can distinguish parametric vs non-parametric with examples

---

**Day 6-7: Rapid Gap Analysis (2025-11-03)** ✅ **COMPLETED**:

**Strategic Pivot**: Instead of continuing algorithm implementations, conducted **comprehensive gap assessment** across all 189 ML interview questions.

**Results**:
- ✅ All 189 questions assessed in ~190 minutes
- ✅ Overall score: 66.5% (75% interview readiness)
- ✅ Identified critical gaps: LLM systems optimization (30%), statistical testing (0-30%), advanced RAG (0-30%)
- ✅ Validated strengths: ML fundamentals (95%), classical ML (90%), deep learning (85%), NLP/transformers (85%)
- ✅ Created detailed documentation: Day6-Gap-Analysis.md + Day6-Gap-Analysis-Detailed.md

**Key Findings**:
- **Performance Distribution**: 55% strong (80-100%), 39% rusty (40-79%), 6% major gaps (0-39%)
- **Critical Gap**: LLM systems optimization - limited experience with distributed training and GPU optimization
- **Strategic Insight**: Gap pattern aligns with career trajectory (model building → LLM integration)

**Week 2 Plan Revised**: Focus on closing critical gaps
- Day 1-2: LLM systems optimization (Megatron-LM, ZeRO papers, NVIDIA GTC talks)
- Day 3: Statistical testing (t-tests, regression assumptions)
- Day 4-5: Advanced RAG (FiD architecture, hybrid retrieval)
- Day 6-7: ML evaluation & problem reframing

See **Day6-Gap-Analysis.md** for complete details and study recommendations.

---

**Learning - Adjusted Approach** (Updated based on Day 3 decisions):

**Fast.AI Decision**: ⏭️ **Skipped for now**
- Reason: Strong foundation
- Better use of time: Targeted theory refresh + algorithm implementation
- May revisit: Week 4+ if time permits, or specific lessons at 2x speed

**Deployment Learning**: 📅 **Deferred to Week 2-3**
- Will learn Gradio/HuggingFace Spaces when building projects (~30 min)
- Modern tooling landscape (Docker/K8s/Cloud ML) - 2-3 hours in Week 2

**Hands-on - MORE TIME HERE (Day 5-7, 4-5 hours)**:

This is where you need the most practice:
- [ ] Set up development environment (Jupyter, Python, PyTorch/TensorFlow)
- [ ] Complete Fast.AI Chapter 1 exercises (don't skip!)
- [ ] Implement 2-3 algorithms from scratch:
  - K-Nearest Neighbors (easy warmup)
  - Decision Tree or K-Means (medium)
  - Neural Network with backprop (hard, optional)
- [ ] Start Project 1: Image classification
  - Use transfer learning (ResNet or EfficientNet)
  - Focus on clean code, not just accuracy

**Week 1 Goals**:
- ✅ Know exactly where you're rusty
- ✅ Refresh hands-on coding muscle memory
- ✅ Have working image classifier started
- ✅ Confidence boost from implementing algorithms

**Resources**:
- Fast.AI Course: https://course.fast.ai/
- ML-Coding-Questions.md (your reference for algorithms to implement)
- Stanford CS229: https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU (reference only)
- Kaggle Datasets: https://www.kaggle.com/datasets

**Adjustment Tip**: If Week 1 feels too easy (you remember more than expected), skip ahead. If it feels too hard, slow down and watch more lectures.

---

## 🔄 **REVISED PLAN BASED ON DAY 6-7 GAP ANALYSIS**

**Context**: Day 6-7 gap analysis (189 questions) revealed:
- ✅ Strong foundation: ML fundamentals (95%), classical ML (90%), deep learning (85%), NLP/transformers (85%)
- ❌ Critical gaps: LLM systems optimization (30%), statistical testing (65%), advanced RAG (30%)
- 🎯 Overall readiness: 75% (B+) - ready for mid-level roles, need gap closure for senior roles

**Strategic adjustment**: **Prioritize closing critical gaps** over building projects in Weeks 2-4, then reassess.

---

### Week 2: LLM Systems Deep Dive + Statistical Testing

**Goal**: Achieve interview-ready level (60-70%) in LLM Systems Optimization, close statistical testing gaps

**Context**: Day 8 topic coverage check revealed 82% dunno (62/76 topics) in LLM Systems → This is foundational learning, not gap closure. Adjusted timeline from 2 days to 5 days with focused 24 high-impact topics.

**NEW PROTOCOL**: **Topic Coverage Check** ✅ **COMPLETED (Day 8)**
- ✅ Comprehensive 76-subtopic inventory created (Week2-LLM-Systems-Topic-Check.md)
- ✅ Self-assessment completed: 0 Know, 14 Unsure (18%), 62 Dunno (82%)
- ✅ Priority topics identified: 24 high-impact topics selected for interview readiness

**Strategy**: Breadth over depth, focus on interview-frequent topics, 3 key papers

---

**Day 1-2: Training Systems Fundamentals (4 hours)** 🔄 **IN PROGRESS**

**Topics** (8 topics):
1. Data parallelism - gradient synchronization
2. Model parallelism - layer splitting
3. Tensor parallelism - attention head splitting
4. **Strong scaling** (Gap Q182)
5. Activation checkpointing
6. ZeRO optimization (Stages 1/2/3)
7. **Memory sharding** (Gap Q184)
8. Mixed precision training

---

**Day 1 Progress** ✅ **COMPLETED (2025-11-04)**:

**Study**:
- [X] Read: Megatron-LM paper (Sections 1-3, focus on tensor parallelism) - 1.5 hours
  - URL: https://arxiv.org/abs/1909.08053
  - ✅ Tensor parallelism for attention heads, FFN layers
  - ✅ Model vs data parallelism trade-offs
  - ✅ Cross-node communication challenges
- [X] Read: ZeRO paper (Sections 1-4, superficial) - 1 hour
  - URL: https://arxiv.org/abs/1910.02054 or local: references/ZeRO_Memory Optimizations Toward Training Trillion Parameter Model 1910.02054v3.pdf
  - ✅ ZeRO-DP three stages overview (Pos/Pos+g/Pos+g+p)
  - ✅ Memory reductions: 4×, 8×, Nd×
  - 🔄 Section 5 technical details - **DEFERRED TO DAY 2**

**Knowledge Check**:
- [X] Completed: 10.5/13 = **80.8%** (B+)
  - Megatron-LM: 77% - Interview-ready for basic questions
  - ZeRO: 47% - Needs Section 5 study
  - Review: 100% - Excellent Week 1 retention

**Topics Status**:
- ✅ Data parallelism, model parallelism, tensor parallelism - Understood
- 🟡 Strong scaling, activation checkpointing, memory sharding - Partial
- ✅ ZeRO optimization Stages 1/2/3 - Conceptual understanding
- 🟡 Mixed precision training - Mentioned, needs depth

---

**Day 2 Progress** ✅ **COMPLETED (2025-11-05)**:

**Study**:
- [X] Read: ZeRO paper Sections 1-5 and 7
  - Clarified: Reduce-scatter vs scatter-reduce terminology
  - Understood: All-gather in forward AND backward for Pos+g+p
  - Learned: Communication breakdown (1Ψ forward + 2Ψ backward = 3Ψ total)
- [X] Read: HuggingFace guide - Partial (PP+DP diagram, 3D parallelism benefits)
  - Key learning: PP staggers DP communication → reduces network congestion
- [X] Create: 1-page cheat sheet - Week2-Day1-2-Cheatsheet.md
  - Includes: 3 parallelism strategies, Megatron-LM patterns, ZeRO 3 stages, ZeRO-R, 3D parallelism, interview Q&A, formulas, pitfalls, real-world examples
  - **Added**: Complete backward pass section (f/g operators, 4 all-reduces per layer)

**Practice**:
- [X] Derive: Backprop gradients for column-parallel (W1) and row-parallel (W2) layers
  - Result: 95-100% understanding, identified all-reduce locations and reasoning
- [X] Trace: Complete MLP block gradient flow (forward + backward)
  - Result: 98%, identified all 4 all-reduces (2 forward + 2 backward)
- [X] Calculate: Strong scaling example (3D parallelism memory/communication)
  - Result: 95%, wrote equations correctly (no numerical calc due to time)

**Knowledge Check** (15 min):
- [X] 70% Day 2 content + 30% Review
- [X] Score: **98% (9.8/10)** - A+
  - Day 9 content: 97.9%
  - Review: 95% (Day 8 weak items resolved, Week 1 retention perfect)

**Day 1-2 Final Status**: ✅ **EXCEEDED TARGETS**
- ✅ 8/8 topics at interview-ready level (can explain all in 2-3 min)
- ✅ ZeRO understanding: 47% → 95%+ (far exceeded 75% target)
- ✅ Overall: 70% → **85%+** (exceeded 80% target)
- ✅ **Weak items**: 3/3 Day 8 gaps resolved in 1 day

---

**Day 3: Parallelism Strategies & Hardware** ✅ **COMPLETED (2025-11-06)**

**Topics** (6 topics):
9. FSDP (Fully Sharded Data Parallel) - how it works
10. **Communication costs** - FSDP vs model parallelism (Gap Q185)
11. Gradient accumulation (via microbatches in PP)
12. Memory hierarchy (HBM, L2, L1, registers)
13. **Arithmetic intensity** (Gap Q183)
14. **Interconnect bandwidth** (Gap Q185)

**Study Completed**:
- [X] Read: Tim Dettmers "GPU Memory Bandwidth" - 20 min
  - URL: https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/
  - Key insight: Memory bandwidth is single best GPU performance indicator
  - Example: A100 vs V100 speedup = 1.73× ≈ bandwidth ratio (1555/900)
- [X] Skim: Roofline model (Wikipedia) - 15 min
  - Formula: Performance = min(Peak Compute, Bandwidth × Arithmetic Intensity)
  - LLMs hit slanted roof (memory-bound, not compute-bound)
- [X] Review: FSDP concepts - 10 min
  - Recognized: FSDP = ZeRO Stage 3 (PyTorch implementation)
  - Skipped PyTorch syntax (not needed for concept understanding)

**Practice Completed**:
- [X] Sketched: Communication patterns for DP, FSDP, TP, PP - 25 min
  - All-reduce (DP), all-gather + reduce-scatter (FSDP)
  - All-reduce within TP group (TP), point-to-point (PP)
- [X] Analyzed: Big O scaling for each parallelism strategy - 45 min
  - DP: O(P), FSDP: O(P), TP: O(B×S×H×L), PP: O(B×S×H×N)
  - Compared communication volume, frequency, bandwidth requirements
  - Synthesized trade-offs for interview answers

**Knowledge Check** (15 min):
- [X] 70% Day 3 content + 30% Review
- [X] Score: **99.0% (9.9/10)** - A+
  - Q1-Q7 (Day 3): 99.3% - Excellent understanding of hardware bottlenecks
  - Q8-Q10 (Review): 98.3% - Perfect retention of ZeRO and TP concepts
  - **User correction**: Caught bubble time approximation, should be (N-1)/(M+N-1) not (N-1)/M ✅
  - **User correction**: Noted TP > FSDP ranking for GPT-3 (not FSDP > TP) ✅

**Quick Reference Created**:
- [X] Day10-Quick-Reference.md - Comprehensive 11-section cheat sheet
  - GPU performance fundamentals, Roofline model
  - Communication patterns (4 strategies)
  - Complexity comparison table with GPT-3 examples
  - Scaling trade-offs (pros/cons/when to use)
  - Interview Q&A (ready-to-use answers)
  - Communication volume calculation details ⭐ NEW

**Day 3 Final Status**: ✅ **EXCEEDED TARGETS**
- ⏱️ Total time: ~2.25 hours (slightly over 2-hour target, but high efficiency)
- 📊 Final readiness: **90%+** for hardware/communication interview questions
- 🎯 Topics mastered: 6/6 at interview-ready level
- 💡 Key insights:
  - Memory bandwidth > TFLOPS for LLM performance
  - TP limited to 8 GPUs per node (hardware constraint)
  - PP has lowest communication volume but bubble time overhead
  - FSDP trades 1.5× communication for N× memory savings

**Target**: Understand parallelism trade-offs and hardware bottlenecks at interview level ✅ **ACHIEVED**

---

**Day 4: Inference Optimization (2 hours)** ✅ **COMPLETED (2025-11-07)**

**Topics** (6 topics):
15. **KV-cache** - how it works, O(n²)→O(n) (Gap Q187)
16. **Quantization** - INT8, INT4 basics (Gap Q187)
17. **Continuous batching** (Gap Q187)
18. **Speculative decoding** (Gap Q187)
19. Multi-query attention (MQA)
20. Serving frameworks - vLLM overview

**Study**:
- [X] Read: vLLM paper (abstract + key sections) - 40 min
  - URL: https://arxiv.org/abs/2309.06180
  - Key concepts: PagedAttention, continuous batching (iteration-level scheduling)
- [X] Read: "KV-cache explained" blog - 15 min
  - URL: https://huggingface.co/blog (derived memory formula: 2×H×L×B)
- [X] Read: Speculative decoding - 15 min
  - URL: https://huggingface.co/blog/whisper-speculative-decoding
  - Key insight: Draft model + parallel verification, synchronous vs continuous batching
- [X] Read: MQA & GQA - 15 min
  - URL: https://medium.com/@maxshapp/grouped-query-attention-gqa-explained-with-code-e56ee2a1df5a
  - Key concepts: Shared K/V heads (MHA: 32, MQA: 1, GQA: 4 groups)
- [X] Read: Quantization Overview - 15 min
  - URL: https://medium.com/@kimdoil1211/speeding-up-large-language-models-a-deep-dive-into-gptq-and-awq-quantization-0bb001eaabd4
  - Key concepts: GPTQ (Hessian-based), AWQ (activation-aware)

**Practice**:
- [X] Create: Inference optimization cheat sheet - 20 min
  - 6 topics: KV-cache, Quantization, Continuous batching, Speculative decoding, MQA/GQA, PagedAttention
  - Format: What it is + What problem it solves + Key trade-off

**Knowledge Check** (10 min):
- [X] 70% Day 4 content + 30% Review
- [X] Score: **97.5% (9.75/10)** - A+
  - Q1-Q7 (Day 4): 97.9% - Strong understanding of all 6 techniques
  - Q8-Q10 (Review): 96.7% - Excellent retention of ZeRO, TP, Roofline model
  - **Critical correction**: Speculative decoding batch size reasoning (75%) - ragged tensor problem, not continuous batching issue (user caught error, cited arxiv.org/html/2510.22876v1)

**Quick Reference Created**:
- [X] Day11-Quick-Reference.md - Comprehensive inference optimization cheat sheet
  - KV-cache memory formula: 2×H×L×2 bytes (FP16)
  - All 6 techniques with trade-offs
  - Interview-ready format

**Day 4 Final Status**: ✅ **COMPLETED**
- ⏱️ Total time: ~2h 10min (10 min over target, but covered all 6 topics)
- 📊 Final readiness: **90%+** for inference optimization interview questions
- 🎯 Topics mastered: 6/6 at interview-ready level
- 💡 Key insights:
  - KV-cache: O(n²)→O(n), but memory cost = 2×H×L×2×S bytes
  - Continuous batching = iteration-level scheduling (from Orca)
  - Speculative decoding: Ragged tensor problem when sequences accept different numbers of tokens (solutions: Masking/Rollback/Dynamic Padding)
  - Small batch benefits: GPU utilization + lower acceptance variance + less ragged tensor overhead
  - PagedAttention: Block-level memory management, eliminates fragmentation
  - MQA/GQA: Memory savings via shared K/V heads
  - Quantization: GPTQ (Hessian) vs AWQ (activation-aware)

**Target**: Can explain all 4 Q187 methods confidently, understand vLLM architecture ✅ **ACHIEVED**

---

**Day 5: Calculations & Transformer Parameters (2 hours)** ✅ **COMPLETED (2025-11-08)**

**Topics** (4 topics):
21. FLOPs calculation - forward/backward pass
22. Memory calculation - weights, optimizer, activations
23. **QKV projections** - parameter calculation (Gap Q189)
24. Transformer parameter counting

**Study**:
- [X] Read: "Transformer Math 101" (EleutherAI) - 40 min
  - URL: https://blog.eleuther.ai/transformer-math/
  - Key concepts: Parameter counting formulas, Chinchilla scaling law, activation memory

**Practice**:
- [X] Calculate: Parameters for GPT-2 (124M), GPT-3 (175B)
  - Formula: V×H + (4H² + 2H×4H)×L
  - GPT-2: 123.5M params ✓
  - GPT-3: 175B params ✓
- [X] Calculate: Memory for training 7B model on A100 (80GB)
  - Model states (no parallelism): 112 GB (need parallelism!)
  - With ZeRO-3 (N=4): 28 GB per GPU
  - Activation memory (full checkpointing): 2×s×b×h×L
  - Max batch size: 59 samples/GPU, 236 global ✓

**Knowledge Check** (10 min):
- [X] Score: **99.6% (9.96/10)** - A+
  - Q1-Q7 (Day 5): 100% - Perfect calculations (parameters, memory, batch size)
  - Q8-Q10 (Review): 97.7% - Excellent retention (memory bandwidth, KV-cache, ZeRO)

**Progress Validation** (15 min):
- [X] Quick 24-topic check: **83% Know** (25/30 high-priority topics)
  - Area 1 (Distributed Training): 75% ✅
  - Area 2 (Memory Optimization): 83% ✅
  - Area 3 (Hardware): 67% 🟡
  - Area 4 (Parallelism): 100% ✅✅
  - Area 5 (Inference): 100% ✅✅

**Day 5 Final Status**: ✅ **COMPLETED**
- ⏱️ Total time: ~2h 15min (study + practice + validation)
- 📊 Final readiness: **99%+ for calculations & transformer parameters**
- 🎯 Topics mastered: 4/4 at interview-ready level
- 💡 Key insights:
  - Attention params: 4H² (not 4H²×n_heads - heads split H!)
  - Chinchilla law (D=20P): Optimizes training cost, not total cost
  - Activation memory: 2sbhL with full checkpointing
  - Memory states require parallelism for 7B+ models (112 GB > 80 GB)
  - ZeRO-3 enables training by sharding model/optimizer/gradients

**Week 2 Day 1-5 Status**: ✅ **LLM SYSTEMS GAP CLOSURE SUCCESSFUL**
- **Target**: 60-70% interview ready → **Achieved**: 83% ✅
- All Gap Q182-189 topics covered with strong understanding
- Knowledge check average: 92% across 5 days (85%, 98%, 99%, 97.5%, 99.6%)
- Ready to move to Day 6-7: Statistical Testing

**Target**: Can calculate transformer params and memory requirements in interview setting ✅ **ACHIEVED**

---

**Day 6-7: Statistics & Probability (Days 13-14) (4-5 hours)** ⚠️ **EXTENDED, CONTINUES TO WEEK 3 DAY 1**

**Rationale**: Comprehensive 43-topic assessment revealed 18.6% baseline (vs estimated 65%) with 8 Dunno + 25 Unsure topics. Extended to 3 days (Week 2 Day 6-7 + Week 3 Day 1) for realistic coverage.

**Assessment Completed** (Day 13 start):
- [X] Comprehensive topic coverage check (see `gap_analysis/Week2-Statistics-Topic-Check.md`)
- [X] Results: 8 Know, 25 Unsure, 8 Dunno → 18.6% baseline readiness
- [X] Decision: Extend from Day 6-7 (2-3 hours) to Day 6-8 (6-7.5 hours total)

---

**Day 6 (Day 13): Regression Diagnostics + Covariance (2-2.5 hours)** ✅ **COMPLETED (2025-11-09)**

**Study** (80-100 min):
- [X] **Section 4: Regression Diagnostics** (60 min) - 4 Dunno topics
  - Durbin-Watson test (autocorrelation in residuals)
  - Breusch-Pagan test (heteroscedasticity detection)
  - Shapiro-Wilk test (normality of residuals)
  - VIF (Variance Inflation Factor for multicollinearity)
  - Used Gemini deep research for comprehensive summary (no hands-on practice)
- [X] **Covariance vs Correlation** (20 min) - Gap Q3 (0%)
  - Watched both StatQuest videos at 2× speed (~20 min total)
  - Units, bounds [-1,1], interpretation, when to use each

**Knowledge Check** (15 min):
- [X] Score: **99.5% (995/1000)** - A+
  - Q1-Q7 (Day 6): 99.3% - Excellent diagnostics & covariance understanding
  - Q8-Q10 (Review): 100% - Perfect retention
  - Caught error in Q6 (impossible correlation value)

**Quick Reference** (20-30 min):
- [X] Created `references/Day13-Quick-Reference.md`: Regression Diagnostics & Covariance (comprehensive, 13 sections)

**Update Documents** (10 min):
- [X] Update `ML-Interview-Prep-Plan.md` (mark Day 6 complete)
- [X] Update `00-CONVERSATION-SUMMARY.md` (Day 13 section)
- [X] Update `Daily-Knowledge-Check-Protocol.md` (Day 13 entry)
- [X] Update `gap_analysis/Week2-Statistics-Topic-Check.md` (progress)

**Target**: Move 5 topics from Dunno/Unsure → Know (Section 4 + covariance) ✅ **ACHIEVED**

---

**Day 7 (Day 14): MLE + Hypothesis Testing (2-2.5 hours)** ✅ **COMPLETED (2025-11-10)**

**Study** (80-100 min):
- [X] **MLE Derivations** (40 min) - Gap Q29 (0%)
  - Exponential distribution: λ̂ = 1/mean ✅
  - Gaussian distribution: μ̂, σ̂² ✅
  - Watched StatQuest videos + practice derivations
- [X] **Chi-square test** (20 min) - Section 3.8
  - Goodness of fit, independence testing ✅
- [X] **Hypothesis Testing Core** (30 min) - Critical Unsure
  - T-test vs Z-test (n<30 vs n≥30, assumptions) ✅
  - Normal distribution calculations (P(X>2), Z-scores) ✅

**Knowledge Check** (15 min):
- [X] 10 questions: 7 Day 7 content, 3 review (Day 6)
  - Score: **86.5% (B+/A-)**
  - New content: 88.6% (7 questions)
  - Review content: 81.7% (3 questions)

**Quick Reference** (20-30 min):
- [X] Created `references/Day14-Quick-Reference.md`: MLE & Hypothesis Testing

**Update Documents** (10 min):
- [X] Updated all files + SM-2 schedule

**Day 14 Results Summary** (2025-11-10):
- ⏱️ **Total time**: ~2 hours (study + knowledge check + docs)
- 💡 **Key learnings**:
  - MLE exponential: λ̂ = 1/mean (log-likelihood trick)
  - MLE Gaussian: μ̂ = x̄, σ̂² = Σ(xᵢ-μ̂)²/n (n, not n-1!)
  - T-test: wider tails than z-test (accounts for σ estimation uncertainty)
  - Chi-square: Σ(O-E)²/E, df=(r-1)(c-1) for contingency table
- 📝 **Created**: Day14-Quick-Reference.md (comprehensive 15-section guide)
- 📊 **Knowledge Check: 86.5% (B+/A-)** - Strong understanding, minor gaps in assumptions
- 🔄 **SM-2 updates**: 3 review items updated, 3 new topics added (now tracking 40 items)

**Success Criteria**: ✅ All met
- ✅ Can derive MLE for exponential and Gaussian distributions
- ✅ Can explain when to use t-test vs z-test with assumptions
- ✅ Understand chi-square test for goodness of fit and independence
- ✅ Can calculate normal distribution probabilities using Z-scores

**Target**: Move 3 topics from Dunno/Unsure → Know (MLE, chi-square, t-test) ✅ **ACHIEVED**

---

**Week 2 Expected Outcomes**:
- ✅ **LLM Systems**: 83% interview ready (25/30 high-priority topics covered) - **EXCEEDED TARGET**
  - Can explain: Data/model/tensor parallelism, ZeRO, FSDP, KV-cache, quantization, batching
  - Can calculate: Transformer parameters, FLOPs, memory requirements
  - Can discuss: 3 key papers (Megatron-LM, ZeRO, vLLM)
  - Knowledge check average: 92% across 5 days
- ⏳ **Statistics & Probability**: In progress (18.6% → target 65-70%)
  - Day 6-7 complete (regression diagnostics, covariance, MLE, hypothesis testing)
  - Continues to Week 3 Day 1 for completion
- ⏸️ **Advanced RAG**: Deferred to Week 3 or later
- ⏸️ **ML Evaluation**: Deferred to Week 3 or later

---

### Week 3: Statistics Completion + System Design ⚠️ **ADJUSTED FROM WEEK 2 OVERFLOW**

**Goal**: Complete statistics gap closure (Day 1), then strengthen system design from 70% → 85%

**Context**: Week 2 statistics work extended into Week 3 Day 1. PyTorch and other topics shifted accordingly.

---

**Day 1 (Day 15): Statistics Completion - A/B Testing + Fundamentals (2-2.5 hours)** ✅ **COMPLETED (2025-11-11)**

**Rationale**: Completing 3-day statistics deep dive from Week 2 (Day 6-8 plan)

**Study** (80-100 min):
- [X] **Regularization as Bayesian Prior** (20 min) - Section 6.7
  - L2 ≈ Gaussian prior (N(0, τ²)), L1 ≈ Laplace prior ✅
  - MAP = maximize (log likelihood + log prior) ✅
  - Read: https://bjlkeng.io/posts/probabilistic-interpretation-of-regularization/
- [X] **A/B Testing Design & Pitfalls** (30 min) - Section 3.9 (MUST-KNOW)
  - Metric choice (north star vs tactical), segmentation ✅
  - Pitfalls: novelty effect, multiple comparisons, selection bias, Simpson's paradox ✅
  - Used Gemini-generated report (original resources too long)
- [X] **Fundamentals Refresh** (30 min) - Section 1 (all Unsure)
  - PDF/CDF/PMF definitions ✅
  - Distributions: Binomial (np(1-p) variance), Geometric (1/p mean), Poisson (λ), Exponential ✅
  - CLT vs LLN: applications and concrete examples ✅
  - Watched video series for each distribution

**Knowledge Check** (15 min):
- [X] 10 questions: 7 Day 1 content, 3 review (overdue items)
  - Score: **92.0% (A-)**
  - New content: 88.6% (7 questions)
  - Review content: 100% (3 questions)

**Quick Reference** (20-30 min):
- [X] Created `references/Day15-Quick-Reference.md`: A/B Testing & Statistics Fundamentals

**Update Documents** (10 min):
- [X] Update `ML-Interview-Prep-Plan.md` (mark Week 3 Day 1 complete)
- [X] Update `00-CONVERSATION-SUMMARY.md` (Day 15 section)
- [X] Update `Daily-Knowledge-Check-Protocol.md` (Day 15 entry)
- [ ] Update `gap_analysis/Week2-Statistics-Topic-Check.md` (deferred - need full reassessment)

**Day 15 Results Summary** (2025-11-11):
- ⏱️ **Total time**: ~2.5 hours (study + knowledge check + docs)
- 💡 **Key learnings**:
  - Regularization as prior: L2 = Gaussian N(0, τ²), L1 = Laplace, λ ~ 1/τ²
  - A/B testing: One primary metric, watch Simpson's paradox, avoid peeking
  - Binomial variance: np(1-p) derived from sum of independent Bernoulli trials
  - CLT enables inference: CI, hypothesis tests work even with non-normal data
- 📝 **Created**: Day15-Quick-Reference.md (comprehensive 9-section guide)
- 📊 **Knowledge Check: 92.0% (A-)** - Strong understanding, minor gaps in CLT examples
- 🔄 **SM-2 updates**: 3 review items (all 100%), 4 new topics added (now tracking 44 items)
- 🎯 **Review retention**: Perfect scores on all overdue items (Megatron 77%→100%, Precision-Recall 88-92%→100%)

**Target Achieved**: Statistics gap closure progressing - core fundamentals solid (88-100% on new topics)

---

**Day 2 (Day 16): PyTorch Basics (2-3 hours)** ✅ **COMPLETED (2025-11-12)** ⭐ **SHIFTED FROM WEEK 3 DAY 1**

**Rationale**: Basic PyTorch literacy needed for research-heavy ML roles. Goal is reading comprehension, not production coding.

**Study** (1 hour):
- [X] PyTorch Quickstart Tutorial (30 min) ✅
  - URL: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
  - Focus: Tensors, Dataset/DataLoader, nn.Module structure
- [X] Autograd Mechanics (30 min) ✅
  - URL: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
  - Understand: .backward(), .grad, computational graph

**Practice** (1.5 hours):
- [X] Implement logistic regression in PyTorch (60 min) ✅
  - Defined nn.Module class with nn.Sequential
  - Forward pass, BCELoss, Adam optimizer
  - Achieved 69.0% test accuracy (vs sklearn 67.5%) ✅
  - Implementation: `notebooks/day16-pytorch.ipynb`
- [X] Read PyTorch FSDP code (30 min) ✅
  - Read `_pre_forward_unshard` function
  - Recognized: all-gather (unshard), reduce-scatter (reshard), prefetching
  - Connected to Week 2 distributed training concepts ✅

**Knowledge Check** (15 min):
- [X] 10 questions: 7 PyTorch content, 3 review (overdue items)
  - Score: **92.5% (A-)**
  - PyTorch content: 87.9% (7 questions)
  - Review content: 98.3% (3 questions - KNN, Boosting, MLE)

**Day 16 Results Summary** (2025-11-12):
- ⏱️ **Total time**: ~2.5 hours (study + implementation + FSDP reading + knowledge check)
- 💡 **Key learnings**:
  - PyTorch training loop: backward() → step() → zero_grad() (both orders work!)
  - torch.no_grad() for evaluation (stops gradient calc, saves memory)
  - BCELoss (needs [0,1]) vs BCEWithLogitsLoss (takes logits)
  - FSDP patterns: all-gather before forward, reduce-scatter after (ZeRO Stage 3)
  - CUDA streams: parallel task queues enable overlap of communication + computation
- 📝 **Created**: `notebooks/day16-pytorch.ipynb` (logistic regression implementation)
- 📊 **Knowledge Check: 92.5% (A-)** - Strong fundamentals, FSDP concepts directional
- 🔄 **SM-2 updates**: 3 review items (all 95-100%), 5 new PyTorch topics added (reorganized from 7 for efficient review)
- 🎯 **Review retention**: 98.3% on overdue items (KNN 7d overdue, Boosting 5d overdue, MLE 1d overdue)
- 🗂️ **Cleanup**: Removed duplicate topic `llm_tensor_parallel_comm` (merged into `llm_megatron`)
- 📋 **Topic organization**: Merged tightly-coupled concepts (training mechanics, FSDP internals) for more efficient review

**Target Achieved**: Basic PyTorch literacy - can read code in interviews, understand distributed patterns ✅

---

**Day 3-5 (Day 17-19): ML Technology Deep Dives (6 hours over 3 days, 2h/day)** 🔄 **IN PROGRESS**

**Pre-study**: Topic Coverage Check ✅ **COMPLETED (2025-11-13)**
- ✅ Created comprehensive 64-item checklist (9 categories)
- ✅ Self-assessed: 0% Know, 37.5% Unsure, 62.5% Dunno
- ✅ Identified top 5 priorities: Kafka, Feature Stores, Airflow, Docker, K8s

**Day 3 (Day 17, Thursday 2025-11-13) - 2 hours:** ✅ **COMPLETED**
- [x] Kafka fundamentals (1 hour)
  - Topics, partitions, consumer groups, offset management
  - Replication & durability (ISR, acks configuration)
  - Use cases: Event streaming, real-time pipelines
- [x] Feature Stores basics (30 min)
  - Online vs offline architecture
  - Point-in-time correctness
  - When to use vs database
- [x] Knowledge check (30 min): 95.1% (A) - 7 new, 3 review

**Day 4 (Day 18, Friday 2025-11-14) - 2 hours:** ✅ **COMPLETED**
- [x] Airflow fundamentals (1 hour)
  - DAGs, operators, tasks, scheduling
  - Idempotency & backfills (critical for interviews)
  - Executors: Local, Celery, Kubernetes
- [x] Feature Stores deep dive (30 min)
  - Feature transformation patterns (streaming vs batch)
  - Feast architecture & materialization
- [x] Knowledge check (15 min): 95.5% (A+) - 7 new, 3 review

**Completion Summary**:
- Study time: ~2 hours (Airflow + Feature Store transformations)
- Knowledge check score: 95.5% (A+)
  - Airflow: 97.5% - Executors 100%, catchup 100%, idempotency 95%, DAGs 95%
  - Feature Store transformations: 98.3% - Perfect batch vs streaming decision framework
  - Review retention: 90% - Kafka 100%, Feature Store 100%, backward pass 70% (gap identified)
- Quick reference created: references/day18-airflow-featureStoreTransform.md

**Day 5 (Day 19, Saturday 2025-11-15) - 2 hours:** ✅ **COMPLETED**
- [x] Docker basics (30 min)
  - Dockerfile, images, containers, registries
  - Best practices: Multi-stage builds, GPU support
- [x] Kubernetes basics (1 hour)
  - Pods, deployments, services, ingress
  - Resource management (CPU/GPU limits)
  - Autoscaling (HPA, VPA)
- [x] Knowledge check (15 min): 89.5% (B+/A-) - 7 new, 3 review

**Target**: Bring ML infra from 0-37.5% → 60-70% interview readiness ✅ **ACHIEVED: 65%**

---

**Day 6 (Day 20, Saturday 2025-11-16) - 2 hours:** ✅ **COMPLETED**

**Mock Interview**: YouTube Recommendation System (60 min)
- [x] Problem 1: Design YouTube recommendation system
  - Focus: Candidate generation, ranking, serving architecture
  - Score: 78/100 (B+) - Strong for first practice
  - Solution: 3-stage cascade (100M → 1K → 100 → 20)
  - Data pipeline: Kafka→Flink→Redis + Spark batch
  - Training: watch_time / duration, 5% randomized traffic, IPS
  - A/B testing: Per-user, 1-2 weeks

**Extended Discussions** (60 min):
- [x] Latency vs QPS relationship (batching increases throughput)
- [x] Two-tower pre-computation (user vs item embeddings)
- [x] A/B test sample size calculation
- [x] Training data bias solutions

**Knowledge Check** (15 min): 91% (A-/A) - REVISED from 89.5%
- Review: 95% (VIF, Airflow idempotency, continuous batching)
- New: 87.1% (system design concepts)
- Critical gap: Throughput calculations - REVISED to 85% after clarification
- **User insight**: "1K predictions = 1 REQUEST, not 1K requests!" ✅

**Follow-up Session (2025-11-17, 30 min)**: ⭐ **TYPICAL NUMBERS CHEATSHEET ADDED**
- [x] Identified real gap: Not formula knowledge, but realistic numbers to plug in
- [x] Created comprehensive cheatsheet with typical values:
  - Server configs (cores, memory, GPU types, costs)
  - Batch sizes (ANN=1, XGBoost=1K candidates, DNN=32-256)
  - Latencies (Redis=0.1-1ms, XGBoost=5-10ms, ANN=10-20ms, DNN=50ms)
  - Parallelism factors (CPU-bound=10-16×, memory-bound=4-8×)
  - Decision tree for "I don't know the numbers"
  - Quick estimation examples (XGBoost, DNN, ANN)
  - Common pitfalls & interview-ready template
- [x] Added 227-line section to `references/day20-system-design-youtube-rec.md`
- **Key meta-learning**: User demonstrated strong self-awareness of learning gaps

**Completion Summary**:
- Mock interview score: 78/100 (B+)
  - Strengths: Requirements (9/10), architecture (8/10), communication (8/10)
  - Needs work: Scaling calculations (6.5/10) ❌ → **GAP CLOSED with cheatsheet** ✅
- Knowledge check: 91% (A-/A) - revised after throughput correction
  - Perfect: Data pipelines (100%), bias handling (100%), A/B testing (100%)
  - Improved: Throughput calc (70% → 85% after 1K predictions clarification)
  - Weak: GPU scaling (50%) - can improve with practice
- Key learning: **Throughput = 1 REQUEST (scoring N candidates) / Latency**, not N/Latency!
- **Critical addition**: Typical Numbers Cheatsheet now available for future system design interviews

**Day 7 (Day 21, Sunday 2025-11-17) - 2 hours:** ✅ **COMPLETED**

**Practice Problems** (2-3 problems × 45-60 min each):
- [x] Problem 2: Design fraud detection system
  - Focus: Real-time inference, feature engineering, model monitoring, XGBoost vs DNN
  - **Score**: 85/100 (A-) - Strong design across all areas
  - **Architecture**: Kafka→Flink→Redis + Snowflake, XGBoost with dynamic batching ($6K/year)
- [ ] Problem 3: Design search ranking system (skipped - optional)
- [ ] Problem 4 (optional): Design ad click prediction system (skipped)

**Focus Areas for Day 21**: ✅ **ALL ACHIEVED**
- [x] **CRITICAL**: Scaling calculations (QPS → servers → cost) for every component - Calculated LogReg, XGBoost, DNN with/without batching
- [x] **CRITICAL**: Throughput formula (Batch Size / Latency) - Applied correctly, understood dynamic batching
- [x] Specific tool names (Kafka, Flink, Redis, FAISS) - All used correctly
- [x] Cost estimation ($/year for all components) - $600, $6K, $90K compared

**Knowledge Check**: **97.0% (A+)**
- Day 21 content: 99.3% (7 questions on fraud detection)
- Review content: 91.7% (residual plots, roofline, throughput calc)

**Completion Summary**:
- Mock interview score: 85/100 (A-) - **+7 points from Day 20!**
  - Feature engineering: 9.5/10 (20 comprehensive features)
  - Cost analysis: 9/10 (LogReg $600, XGBoost $6K, DNN $90K)
  - Architecture: 8.5/10 (Kafka→Flink→Redis, cold start handling, sliding windows)
  - Dynamic batching: 9/10 (50× throughput improvement)
- Knowledge check: 97% - **+6 points from Day 20 (91%)!**
  - Perfect scores on model selection, features, Flink vs Snowflake, sliding windows, cold start, manual review
- Key improvements:
  - Feature engineering: 7/10 → 9.5/10
  - Cost analysis: 6/10 → 9/10
  - Throughput calculations: Now calculates with/without batching
- Quick reference created: `references/day21-fraud-detection-system.md`

**Memorize** ✅:
- [x] Common architectures: Two-tower, cascade, lambda architecture
- [x] Scale numbers: QPS targets (1K, 10K, 100K+), latency (<10ms, 10-100ms, 100-500ms)
- [x] **NEW**: Dynamic batching is COST OPTIMIZATION (12-50× savings), not just scaling!

---

**Week 3 Note**: Schedule adjusted to accommodate ML Tech deep dive (3 days instead of 2). Advanced RAG moved to Week 4 Day 1-2 to enable comprehensive gap reassessment after all study complete.

---

### Week 4: Progress Check & Adaptive Planning ⚠️ **SUBJECT TO ADJUSTMENT**

**Goal**: Close Advanced RAG gap, then assess overall progress and decide next steps

**Context**: Week 4 serves as final gap closure + checkpoint. Advanced RAG must be studied BEFORE gap reassessment to enable meaningful evaluation of Q177-179.

---

**Day 1-2: Advanced RAG Architectures (2-3 hours)** ⭐ **MOVED FROM WEEK 3 DAY 6-7**

**Pre-study**: Topic Coverage Check (20 min) - if needed
- [x] ✅ **COMPLETED Day 22**: List RAG subtopics: Retrieval methods (sparse, dense, hybrid), Reranking, FiD, ColBERT, DPR, etc.
- [x] ✅ **COMPLETED Day 22**: Self-assess: know/unsure/dunno for each
  - **Result**: 86 topics assessed, 21.3% weighted baseline → Day 1 studied 11 topics (7 new + 4 consolidated)

**Study**:

**Day 1 Topics** (7 new + 4 consolidated) - ✅ **COMPLETED Day 22**:
- [x] ✅ RRF (Reciprocal Rank Fusion) - Formula: Score = Σ 1/(k + rank_r(d)), k=60
- [x] ✅ SPLADE (Learned Sparse) - FLOPS regularization, vocabulary expansion
- [x] ✅ DPR (Dense Passage Retrieval) - TWO types of negatives (in-batch + hard from BM25)
- [x] ✅ Cross-encoder reranking - Two-stage: bi-encoder retrieve → cross-encoder rerank
- [x] ✅ ColBERT (Late Interaction) - Token-level embeddings, MaxSim, 100× storage cost
- [x] ✅ MMR (Maximal Marginal Relevance) - Diversity reranking with MINUS sign
- [x] ✅ Lost-in-the-middle problem - LLMs ignore middle docs, mitigation strategies
- [x] ✅ Consolidated 4 "unsure" topics: Dense/Sparse/Hybrid retrieval, Contrastive learning

**Day 1 Achievement (Day 22)**: 96% knowledge check (98.9% on new content)
- ✅ Studied 11 topics: RRF, SPLADE, DPR, Cross-encoder, ColBERT, MMR, Lost-in-the-middle + 4 consolidations
- ✅ Created `references/Day22-Advanced-RAG-Day1.md` quick reference (8 pages)
- ✅ All 6 topics added to SM-2 schedule (fixed: ColBERT separate from reranking)
- 📊 Estimated progress: 21.3% → 40-45% weighted overall after Day 1

**Day 2 Topics** (7 new patterns + 3 consolidations) - ✅ **COMPLETED Day 23**:
- [x] ✅ **FiD (Fusion-in-Decoder)** - Gap Q177 closed (0%→100%), encode independently/decode jointly, FiD vs Long Context with prefix caching
- [x] ✅ **GraphRAG** - Two modes: Local (entity) + Global (themes via community summaries)
- [x] ✅ **RAFT** - P=0.8 golden+distractors, (1-P)=0.2 distractors-only, teach model to ignore distractors
- [x] ✅ **Agentic RAG** - ReAct loop (Thought→Action→Observation), 3-5× cost, multi-tool
- [x] ✅ **Multi-hop retrieval** - Three implementations (query decomp, agentic, GraphRAG) with trade-offs
- [x] ✅ **Parent document retrieval** - Two storage systems (vector+parent), deduplication critical
- [x] ✅ **Complex PDF parsing** - Parse vs multi-modal vs hybrid, table problem
- [x] ✅ Review 3 "unsure": Multi-modal RAG, Query decomposition, Standard RAG pipeline

**Day 2 Achievement (Day 23)**: 96.3% knowledge check (A+)
- ✅ Studied 7 new topics: FiD, GraphRAG, RAFT, Agentic RAG, Multi-hop, Parent doc, PDF parsing
- ✅ Consolidated 3 "unsure" topics: Multi-modal RAG, Standard RAG pipeline, Query decomposition
- ✅ Knowledge check: 98.6% on new content (exceptional mastery)
- ✅ Created `references/Day23-Advanced-RAG-Day2.md` quick reference (comprehensive)
- ⭐ **Gap Q177 (FiD) closed: 0% → 100%**
- 📊 Estimated progress: 40-45% → **52-55% weighted overall**, **68-72% high-priority** after Day 2

**Day 3 Topics (Optional)** - Evaluation + Final Consolidation:
- [ ] **Retrieval metrics** - Recall@K, Precision@K, MRR, NDCG (formulas + hand calculation)
- [ ] **End-to-end RAG metrics** - Faithfulness, Answer Relevance, Context Quality (RAGAs)
- [ ] **Context window optimization** - Truncation, compression, summarization strategies
- [ ] Review remaining "unsure": FAISS, HyDE, Query rewriting/expansion, Chunking strategies

**Practice** (Optional):
- [ ] Implement: Simple hybrid retrieval (BM25 + Sentence-Transformers + RRF)
- [ ] Calculate: Retrieval metrics by hand for sample scenario

**Progress Targets**:
- **Baseline**: 21.3% weighted (23.2% high-priority)
- **After Day 1**: ~40-45% weighted ✅
- **After Day 2**: **52-55% weighted**, **68-72% high-priority** ✅ **ACHIEVED**
- **After Day 3 (optional)**: Target 55-60% weighted overall, **82% high-priority**
- **Success Criteria**: ≥82% high-priority (23/28 topics), ≥55% weighted overall
- **Current Status**: 68-72% high-priority is functional for interviews, Day 3 optional for polish

---

**Day 3-4 (Day 24-25): Gap Re-assessment (2-3 hours)** ✅ **COMPLETED (2025-11-20 to 2025-11-21)**

**Activity**:
- [x] Re-test on ALL weak areas (45 questions over 2 days)
  - LLM systems (Q182-189): **89.4%** ✅ (exceeded 85% target)
  - Statistical testing (Q181-182): **81.7%** ✅ (strong understanding)
  - **Advanced RAG (Q177-179)**: **99.2%** ✅✅ (exceptional! exceeded 75% target)
  - System design (Q145-165): **86.0%** ✅ (exceeded 85% target)
  - ML infra (Q145-160 subset): **90.0%** ✅ (far exceeded 75% target)
  - Unstudied weak areas: **68.8%** 🟡 (better than expected 30-50%)
- [x] Calculate improvement: **81.7% overall** (up from ~50% Day 6 baseline, **+31.7%**)
- [x] Identified remaining gaps: Bonus discovery areas (dropout, few-shot learning)

**Extended Session - Communication Patterns Clarification** (60+ min):
- [x] Clarified QPS calculation: Use GPU processing time only, not batch waiting time
- [x] Clarified batch collection time: Use per-GPU QPS, not total QPS
- [x] Clarified communication volume formulas: Distinguish logical vs physical network traffic
  - DP/FSDP: 2P per step (ring all-reduce overhead)
  - TP: 8×B×S×H per layer (4 all-reduces × 2× ring overhead)
  - PP: B×S×H per stage boundary (point-to-point, no ring overhead)
- [x] Created comprehensive reference: `references/Day25-Communication-DynamicBatching.md`

**Assessment Results Summary**:

| Category | Score | Status | Notes |
|----------|-------|--------|-------|
| **LLM Systems** | 89.4% | ✅ | Strong scaling, ZeRO, KV-cache, communication patterns |
| **Statistics** | 81.7% | ✅ | MLE, t-tests, regression diagnostics |
| **Advanced RAG** | 99.2% | ✅✅ | **EXCEPTIONAL** - FiD, GraphRAG, RAFT, hybrid retrieval |
| **ML Infrastructure** | 90.0% | ✅ | Kafka, Airflow, Feature Stores |
| **System Design** | 86.0% | ✅ | YouTube + Instagram recommendations |
| **Unstudied Weak** | 68.8% | 🟡 | Transfer learning, spaced repetition working |
| **Overall** | **81.7%** | ✅ | **Up from ~50% Day 6 baseline** |

**Decision**: ✅ **Option A - Start Projects** (all critical gaps closed, targets exceeded!)

**Readiness Update**:
- Overall: 75% (Week 1) → **85%** (Week 4) ✅
- LLM Systems: 30% → **89.4%** (+59.4%) ⭐
- Statistics: 65% → **81.7%** (+16.7%)
- Advanced RAG: 21.3% → **99.2%** (+77.9%) ⭐⭐
- ML Infrastructure: 0-37.5% → **90.0%** (+52.5%+) ⭐
- System Design: 0% → **86%** (+86%) ⭐

**Key Achievements**:
- ✅ All studied areas ≥80% (target met)
- ✅ Gap Q177 (FiD): 0% → 100%
- ✅ Communication patterns confusion resolved with Day25 reference
- ✅ User caught 3 scoring errors (exceptional attention to detail)
- ✅ Ready for senior-level ML Engineer interviews

---

**Phase 1 Revised Deliverables**: ✅ **ALL ACHIEVED**
- ✅ **Critical gaps closed**: 81.7% overall, all studied areas ≥80% (target: 80%+) **EXCEEDED**
- ✅ **Interview readiness**: 85% overall (up from 75% Week 1) **EXCEEDED**
- ✅ **Strong system design skills**: Can design 3-5 ML systems (YouTube, fraud detection, Instagram)
- 🎯 **Projects**: 0 projects (Week 4 Day 5-7 to begin) - **Acceptable**, gap closure prioritized

**Key Success Metric**: ✅ **Confidence to start applying to senior-level ML roles by Week 5** (target exceeded!)

**Key Insight**: **Gap closure > Projects** for senior roles. Projects demonstrate skills but don't close knowledge gaps.

**Week 4 Day 3-4 Decision**: Option A - Start Projects (all critical gaps closed, all targets met or exceeded)

---

## Phase 2: Portfolio Project 1 - RAG System (remainder of Week 4 + Week 5)

✅ **DECISION MADE** after Week 4 checkpoint (85% readiness achieved)
- All critical gaps closed: LLM Systems 89%, Statistics 82%, RAG 99%, ML Infrastructure 90%
- Ready to start portfolio projects
- Skip redundant coursework, focus on implementation and interview prep

**Goal**: Build production-quality RAG Q&A system demonstrating modern ML engineering skills
**Time**: ~13 hours total (Week 4 Day 7: 2 hours + Week 5: 11 hours)

**Project**: RAG Q&A System over ArXiv Papers
- 32 ArXiv papers (2020-2025) on RAG and LLM techniques
- Hybrid retrieval (Dense FAISS + Sparse BM25 + RRF fusion)
- Tech stack: sentence-transformers, FAISS, rank-bm25, OpenAI API, Ragas
- Generation with GPT-3.5-turbo
- Evaluation: Ragas framework (context precision, faithfulness, answer relevance) + retrieval metrics (Recall@K, MRR, NDCG)
- Docker deployment to Streamlit Cloud

**Project Plan**: See `projects/rag-qa-system/project-plan.md` for complete specification

---

**Day 5 (Sat, Nov 22 - 30 min)** ✅ **COMPLETED**

**Knowledge Check (15-20 min)**:
- [X] 10 review questions from knowledge schedule (prioritize items due/overdue)
- Score: **80.0% (800/1000)** - B/B+
- Perfect scores (100%): 7 topics - Shapiro-Wilk, Distributions, CLT/LLN, Regularization, Speculative decoding, SPLADE, Communication patterns
- Good scores (75%): Chi-square, Durbin-Watson
- Weak item: FSDP stream sync (50%) - 3rd review, still struggling with CUDA streams

**Project Selection (15-30 min)**:
- [X] Choose: **Full Project 4 (Option B, ~12 hours over 2 weeks)** - Changed from Mini RAG
- [X] Decision rationale: 1 day insufficient for quality portfolio project, extended timeline realistic
- [X] Plan: Data source (ArXiv papers on RAG/LLMs), architecture (hybrid retrieval + RRF fusion + Ragas evaluation), tech stack (sentence-transformers, FAISS, rank-bm25, OpenAI API, Ragas)
- [X] Created subfolder: `projects/rag-qa-system/`
- [X] Created comprehensive `project-plan.md` (450+ lines)
  - Problem statement, architecture overview, tech stack decisions
  - Complete timeline (Weekend + Week 5, 12 hours total)
  - ArXiv papers list (20-30 URLs)
  - Full code structure (~800 lines)
  - Requirements list, evaluation plan, interview talking points

**Day 5 Summary**:
- ⏱️ Total time: ~30 min (knowledge check + project planning)
- 📊 Knowledge check: 80% (7 perfect scores, communication patterns improved 75%→100%)
- 📝 Project plan: Comprehensive specification document created
- 🎯 Progress: Ready for Day 6 light prep

---

**Day 6 (Sun, Nov 23 - 30 min)** ✅ **COMPLETED**

**Knowledge Check (15-20 min)**:
- [X] 10 review questions from knowledge schedule (all overdue items)
- Score: **87.5% (875/1000)** - A-/B+
- Perfect scores (100%): 7 topics - PyTorch (no_grad, BCELoss), FSDP internals (breakthrough!), Airflow executors, GPU scaling, All concepts for regression metrics
- Good scores (75%): nn.Module (forgot super().__init__), Docker multi-stage, K8s Deployment, Regression metrics (missing R² formula)
- Failed: Adam optimizer (50%) - Formulas missing, reset to review tomorrow

**RAG Project Setup (30 min)**:
- [X] Downloaded **32 ArXiv papers** (73.1 MB) to `data/raw/` - Added 4 papers beyond list, including 1 from 2025!
  - Core RAG: 7 papers (Lewis, FiD, Lost in Middle, Self-RAG, RAPTOR, RAFT, GraphRAG)
  - Retrieval: 4 papers (ColBERT, DPR, SPLADE, Hybrid)
  - Evaluation: 3 papers (RAGAS, ARES, RGB)
  - Advanced: 11 papers (Query rewriting, HyDE, Step-back, Active RAG, Vision RAG, Deliberative RAG, etc.)
  - Surveys: 3 papers (Gao, Li, Asai)
  - Multi-hop: 2 papers (ReAct, Multi-hop QA)
  - 2025 papers: Embedding limits (Weller)
- [X] Created folder structure: data/raw, data/processed, data/eval, src/, evaluation/, tests/, outputs/eval_results, outputs/logs
- [X] Created `requirements.txt` with 28 dependencies (sentence-transformers, FAISS, BM25, OpenAI, Ragas, etc.)
- [X] Created `download_papers.py` script for automated ArXiv downloads
- [X] Created `papers_to_download.md` reference list with 30 papers

**Day 6 Summary**:
- ⏱️ Total time: ~30 min (knowledge check + RAG project setup)
- 📊 Knowledge check: 87.5% (FSDP breakthrough 50%→100%, Adam optimizer gap identified)
- 📚 Papers: 32 ArXiv papers downloaded (600-800 pages corpus)
- 📁 Project: Complete folder structure + requirements ready
- 🎯 Progress: Ready for implementation

--

### Week 4 Day 7 (Mon, Nov 24) - ✅ **COMPLETED**



**Day 7 (Mon, Nov 24 - 2-3 hours)** ✅ **COMPLETED** - Main implementation day

**Knowledge Check (15 min)**:
- [X] 10 review questions (due/overdue items)
- Score: **97.0% (A+)** - 8 perfect, 2 partial
- **Adam optimizer mastered**: 50%→100% (all 5 formulas perfect)
- Perfect scores: Memory calc, batch size, Chinchilla, activation mem, residual plots, PyTorch training, Kafka
- User caught 3 scoring errors (Q2, Q5, Q9 - all valid corrections)

**Data Loading + Embeddings + FAISS Index (2 hours)**:
- [x] Parse 32 PDFs with PyMuPDF4LLM (layout-aware parsing for two-column papers)
- [x] Token-based chunking (500 tokens, 50 overlap using tiktoken)
- [x] Generate embeddings with sentence-transformers (all-MiniLM-L6-v2)
- [x] Build FAISS IndexFlatIP with normalized embeddings
- [x] Validate search quality (top-5 results all relevant, 0.75 cosine similarity)
- [x] Create reference documentation: `references/day28-rag-implementation.md`

**Code Structure**:
- [X] Clean OOP design: PDFDocument, CorpusLoader, VectorStore classes
- [X] Separate modules: utils.py, data_loader.py, vector_store.py
- [X] Main scripts: build_index.py, test_search.py

**Result**: 1541 chunks indexed, high-quality semantic search validated ✅

**Day 7 Summary**:
- ⏱️ Total time: ~2 hours (implementation + knowledge check + docs)
- 📊 Knowledge check: 97% (A+) - Adam breakthrough, 8/10 perfect
- 🔧 RAG implementation: Production-quality pipeline complete
  - Stats: 1541 chunks, 384-dim embeddings, <10ms search
  - Quality: High-quality retrieval verified
- 📝 Reference sheet: Comprehensive technical documentation
- 🎯 Progress: Week 5 Day 1 complete, ready for Day 2 (BM25 + RRF fusion)

---

### Week 5 (Nov 25-Dec 1)

**Day 1 (Tue, Nov 25) - Hybrid Retrieval (2.5 hours)** ✅ **COMPLETE**:
- [X] Implement BM25 sparse retrieval (rank-bm25) with NLTK tokenization
- [X] Implement RRF fusion: score = Σ 1/(k + rank_i), k=60, 1-based ranking
- [X] Compare retrieval quality: Dense-only vs Sparse-only vs Hybrid (2 query sets)
  - General NLP queries: Hybrid 40% < Dense 60% (query-corpus mismatch)
  - RAG-focused queries: Hybrid 80% > Dense 67% ✅
- [X] Key finding: Query-corpus alignment critical (BM25: 13%→67% with aligned queries)
- [X] Decision: Use Hybrid for production (80% > 67% for RAG queries)
- [X] **Daily knowledge check**: 96% (A+) - Exceptional hybrid retrieval mastery

**Day 2 (Wed, Nov 26) - Generation Pipeline (2 hours)**: ✅ **COMPLETE**
- [X] Design prompt template for RAG Q&A with citations (with/without context prompts)
- [X] Implement generation pipeline with gpt-4o-mini (cheaper + better than gpt-3.5)
- [X] Create 10 test questions in `data/eval/test_questions.json`
- [X] Smoke test: 5 questions × 4 modes (hybrid/dense/sparse/none)
- [X] Token usage validated: 2700 vs 50 tokens proves retrieval working
- [X] Issue discovered: Negative question handling (retrieval contamination)
- [X] **Daily knowledge check**: 97% (A+) - Outstanding log investigation, root cause analysis

**Day 3 (Thu, Nov 27) - Planning & Cost Analysis (1 hour)**: ✅ **COMPLETE**
- [X] Add ArXiv metadata to chunks (title, authors, year, URL)
- [X] Research Ragas 0.3.9 API (generate_with_langchain_docs, Ollama investigation)
- [X] **Critical discovery**: Ragas cost underestimate
- [X] Design sampling strategy (250 chunks, 7-8 per paper)
- [X] Analyze manual vs Ragas test format differences
- [X] Clarify ground truth requirements for metrics (Context Recall needs it, Context Precision better with it)
- [X] **Daily knowledge check**: 94% (A) - Excellent overdue item retention, caught error on Context Precision

**Day 4 (Fri, Nov 28) - Retrieval Evaluation (2 hours)**: ✅ **COMPLETE**
- [X] Add reference filtering to corpus (Ollama-based, 9.5% filtered out)
- [X] Rebuild index with filtered chunks (1395 remaining)
- [X] Implement sampling strategy (~250 chunks from 32 papers)
- [X] Generate 42 Ragas questions with Ollama (free, exceeded target)
- [X] Create evaluate_retrieval.py: Recall@5, MRR, NDCG
- [X] Run retrieval evaluation on 41 questions (3 modes: sparse, dense, hybrid)
- [X] **Critical insight**: incomplete ground truth (metrics are lower bounds)
- RAG evaluation deferred to Day 5 (use LLM-based context_recall)

**Day 5 (Sat, Nov 29) - RAG Evaluation & Error Analysis (5 hours)**: ✅ **COMPLETE**
- [X] Run RAG evaluation on 10 manual + 41 Ragas questions (4 modes)
- [X] **Question quality analysis**: Discovered 46% low-quality questions from bibliography sections
- [X] **Root cause identified**: Generated from 500-token chunks instead of whole documents
- [X] Manual review and filtering: 13 questions removed → 28 clean questions
- [X] Metrics recalculation: ~13% improvement after filtering
- [X] **Error analysis**: Categorized failure modes
  - Key finding: Dense 29.6% retrieval failures vs Sparse 10.7% (3× worse!)
  - SPARSE success rate: 57.1% (best), HYBRID: 46.4%, DENSE: 25.9%
- [X] **Decision**: Default to SPARSE (best performance), keep HYBRID as option

**Day 6 (Sun, Nov 30) - Deploy + Polish (6 hours)**: ✅ **COMPLETE**
- [X] **Testset regeneration**: Fixed root cause (whole documents vs chunks)
- [X] **RAG evaluation v2**: 42 questions (10 manual + 32 Ragas)
- [X] **Error analysis v2**: Validated Dense 3.7× worse retrieval failures
- [X] **Streamlit UI**: Complete with mode selection, top-K config, example questions
- [X] **Docker containerization**: Dockerfile + docker-compose.yml
- [X] **Comprehensive README.md**: Architecture, results, technical decisions
- [X] **File structure**: Created outputs/eval_results/ for clean separation
- [ ] Push to GitHub (deferred - will create separate portfolio repo)
- [ ] Streamlit Cloud deployment (optional)

**Phase 2 Deliverable**:
- ✅ Production-quality RAG system on GitHub
- ✅ Demonstrates: Hybrid retrieval, RRF fusion, automated evaluation, Docker deployment
- ✅ Interview-ready talking points (architecture, evaluation rigor, results)
- ✅ Real metrics: Recall@K, MRR, Ragas scores

**Resources**:
- Project folder: `projects/rag-qa-system/`
- Papers: 32 ArXiv PDFs in `projects/rag-qa-system/data/raw/`

---

## Phase 3: Portfolio Project 2 + Interview Prep (Weeks 6-8)

**Goal**: Build 2nd portfolio project + master interview formats
**Time**: 24.5-30 hours total (8-10 hours/week, matches Week 1-4 pace)
**Rationale**: Address hands-on freshness across ML areas, prepare for Big Tech + AI-first interviews

### Week 6: LeetCode Assessment + Neural Network Start (8-10 hours)

**Day 1-2: LeetCode Assessment (2.5-3 hours)**:
- [ ] **Day 1** (1.5 hrs): 3 Medium problems (arrays, hashmaps, two pointers)
- [ ] **Day 2** (1-1.5 hrs): 2 Medium problems (trees/graphs, DP or recursion)
- [ ] **Goal**: Assess current proficiency - if 7+/10 solved in <30 min, skills retained
- [ ] **Outcome**: Determine if intensive LeetCode prep needed or light ongoing practice sufficient

**Day 3-5: Neural Network Implementation Part 1 (5.5-7 hours)**:
- [ ] **Day 3** (2 hrs): Forward pass
  - Matrix multiplication for fully-connected layers
  - Activation functions (ReLU, sigmoid, softmax)
  - Loss calculation (cross-entropy)
  - Test on toy data (XOR or synthetic)
- [ ] **Day 4** (2.5 hrs): Backpropagation
  - Gradient calculations for each layer (∂L/∂W, ∂L/∂b)
  - Chain rule implementation
  - Numerical gradient check for correctness
  - Validate gradients match numerical approximation
- [ ] **Day 5** (1.5-2 hrs): Basic training loop
  - SGD optimizer implementation
  - Batch processing
  - Train on simple dataset (XOR or small synthetic)
  - Verify loss decreases

**Week 6 Checkpoint**: LeetCode proficiency assessed, Neural network 60% complete

---

### Week 7: Neural Network Completion + ML Coding Drills (8.5-10 hours)

**Mon-Tue: Neural Network Part 2 (4-5 hours)**:
- [ ] **Monday** (2-3 hrs): MNIST training + visualization
  - Load MNIST dataset
  - Train full network (2-3 hidden layers)
  - Generate loss curves and accuracy plots
  - Decision boundary visualization (if 2D projection)
  - Achieve >95% accuracy on MNIST
- [ ] **Tuesday** (2 hrs): Documentation + polish
  - Write comprehensive README:
    - Architecture diagram
    - Math formulas (forward pass, backprop equations)
    - Training results and convergence analysis
    - Comparison with theoretical backprop derivations
  - Add code comments and docstrings
  - GitHub upload with clear usage examples

**Deliverable**: ✅ 2nd portfolio project complete (Neural Network from scratch)

**Wed-Fri: ML Coding Speed Drills (4-5 hours)**:
- [ ] **Wednesday** (1.5 hrs): Logistic regression from scratch
  - Implement gradient descent for logistic regression
  - Practice until can complete in <15 min
  - Compare results with sklearn
- [ ] **Thursday** (1.5 hrs): k-NN and k-means
  - k-NN: 30 min implementation target
  - k-means: 45 min implementation target
  - Validate correctness against sklearn
- [ ] **Friday** (1.5 hrs): Neural network from memory + LeetCode
  - Reimplement simple neural network without looking at code (1 hr)
  - 2 LeetCode Medium problems (30 min)

**Week 7 Checkpoint**: 2 portfolio projects complete, speed validated on core algorithms

---

### Week 8: System Design + Mock Interviews (8-10 hours)

**Monday: ML System Design Framework Study (1.5 hours)**:
- [ ] Review problem-solving framework:
  - Problem definition and requirements
  - Data collection and labeling
  - Feature engineering pipeline
  - Model selection and training
  - Evaluation metrics and A/B testing
  - Deployment and serving architecture
  - Monitoring and retraining
- [ ] Watch 1-2 example ML system design interviews on YouTube
- [ ] Study solutions from `ML-System-Design-Questions.md`

**Tuesday: Search Ranking System Design (1.5 hours)**:
- [ ] Design full search ranking system (Google/Bing style):
  - Query understanding (spelling correction, intent classification)
  - Candidate generation (inverted index, ~1000 documents)
  - Ranking model (learning-to-rank, BM25 + neural features)
  - Real-time constraints (<100ms latency)
  - Personalization and diversity
  - Evaluation (NDCG, MAP, user engagement metrics)
- [ ] Draw architecture diagram
- [ ] Record yourself explaining (simulate interview, 45 min)

**Wednesday: ML Model Serving at Scale (1.5 hours)**:
- [ ] Design system to serve recommendation model at 100K QPS:
  - Load balancing strategy (regional load balancers + model replicas)
  - Caching layer (Redis for user/item features, model predictions)
  - A/B testing infrastructure (experiment framework, traffic splitting)
  - Model versioning and rollback (canary deployment, shadow mode)
  - Monitoring and alerting (latency, throughput, model quality drift)
  - Auto-scaling and resource management
- [ ] Infrastructure focus (leverages your 90% ML Infrastructure strength)
- [ ] Write full solution with diagrams

**Thursday: Mock Interviews (2-3 hours)**:
- [ ] Schedule 2 mock interviews on Pramp or interviewing.io:
  - 1× ML system design (45 min)
  - 1× ML theory + coding (45 min)
- [ ] Record detailed feedback:
  - Communication clarity
  - Technical depth
  - Areas for improvement
- [ ] Buffer time for scheduling and setup

**Friday: Review + Behavioral Prep (2 hours)**:
- [ ] Review mock interview feedback thoroughly
- [ ] Identify gaps and create action items
- [ ] Polish 8-10 STAR stories from `Behavioral-Questions.md`:
  - Google ML projects (BERT, neural networks in production)
  - Netflix LLM/RAG integration experience
  - Cross-team collaboration examples
  - Technical leadership and mentorship
- [ ] Prepare thoughtful questions for interviewers (based on company research)
- [ ] 2 LeetCode Medium problems (30 min) for maintenance

**Phase 3 Deliverables**:
- ✅ 2 portfolio projects (RAG + Neural Network) on GitHub
- ✅ LeetCode proficiency assessed (ongoing practice if needed)
- ✅ 2 fresh system design solutions (Search Ranking + Model Serving)
- ✅ Mock interview feedback captured and action items identified
- ✅ Behavioral stories polished and ready for interviews

**Resources**:
- ML System Design: `ML-System-Design-Questions.md`
- ML Coding: `ML-Coding-Questions.md`
- Behavioral: `Behavioral-Questions.md`
- Mock interviews: Pramp (https://www.pramp.com/), interviewing.io (https://interviewing.io/)

---

## Phase 4: Job Search & Iteration (Week 9-10+)

✅ **READY TO START** - 85% readiness achieved, 2 portfolio projects complete
- Target companies: Big Tech (Google, Meta, Amazon) + AI-first (OpenAI, Anthropic, Cohere)
- Strategy: Tailored applications + network referrals + ongoing interview prep

**Goal**: Apply to target companies, interview, and iterate based on feedback
**Time**: Ongoing (varies based on interview pipeline)

### Week 9: Applications (Variable time)

**Company Identification (1-2 hours)**:
- [ ] Identify 10-15 target companies:
  - **Big Tech**: Google (ML Engineer), Meta (ML Engineer), Amazon (Applied Scientist)
  - **AI-first**: OpenAI (Applied Scientist), Anthropic (ML Engineer), Cohere (ML Engineer)
  - **Top ML startups**: Scale AI, Hugging Face, Weights & Biases, etc.
- [ ] Research each company: products, ML use cases, team structure, culture
- [ ] Prioritize by fit: role level (senior), team (ML systems/infrastructure), location

**Resume & Materials (2-3 hours)**:
- [ ] Tailor resume for each application:
  - Highlight 2 portfolio projects (RAG + Neural Network)
  - Emphasize 9.5 years experience (7 years Google production ML + 2.5 years Netflix LLM/RAG)
  - Quantify impact where possible
  - Adjust keywords based on job description
- [ ] Write company-specific cover letters (optional for some)
- [ ] Prepare LinkedIn profile with projects and updated skills

**Network Outreach (2-3 hours)**:
- [ ] Reach out to Google network for referrals (7 years = strong network)
- [ ] Reach out to Netflix contacts for referrals or intros
- [ ] Use LinkedIn to find connections at target companies
- [ ] Send personalized messages (not generic templates)

**Submit Applications (1-2 hours)**:
- [ ] Submit 5-10 applications in Week 9
- [ ] Track applications in spreadsheet:
  - Company, role, date applied, status, referral (Y/N)
  - Recruiter contact, interview stages, feedback notes
- [ ] Follow up on referrals after 1 week if no response

---

### Week 10+: Interview Loop & Ongoing Maintenance

**Interview Preparation (Variable based on pipeline)**:
- [ ] **If Big Tech interviews scheduled**: Ramp up LeetCode practice
  - Daily practice: 1 Medium problem per day (30-45 min)
  - Focus on company-specific patterns:
    - Google: Graphs, trees, dynamic programming
    - Meta: Arrays, hashmaps, system design
    - Amazon: Leadership principles + coding (2 easy-medium)
  - Review common patterns: sliding window, two pointers, DFS/BFS, DP
- [ ] **If AI-first interviews scheduled**: Focus on ML implementation + theory
  - Review transformer architecture, attention mechanisms
  - Practice implementing backprop, loss functions, optimizers
  - Prepare to discuss recent papers and LLM trends

**Knowledge Retention (15 min/day)**:
- [ ] Daily knowledge checks with SM-2 spaced repetition
- [ ] Review items due from `data/knowledge-schedule.md`
- [ ] Maintain 85%+ readiness across all topics
- [ ] Focus on weak areas (<80%) with more frequent reviews

**Interview Response & Iteration**:
- [ ] Respond quickly to interview requests (within 24 hours)
- [ ] Schedule interviews strategically:
  - Less preferred companies first (practice interviews)
  - Top choice companies after 2-3 practice rounds
- [ ] After each interview:
  - Document all questions asked (coding, theory, system design, behavioral)
  - Identify knowledge gaps or weak responses
  - Study gaps immediately (within 24-48 hours)
  - Update interview prep materials with new questions
- [ ] Add targeted study based on feedback:
  - If LeetCode struggles: Add 1-2 weeks intensive practice
  - If system design gaps: Practice 2-3 more problems
  - If ML theory weak: Deep dive on specific topics

**Ongoing Portfolio Updates**:
- [ ] Add second project if specific gap identified in interviews
- [ ] Update project READMEs based on interviewer questions
- [ ] Consider adding blog post explaining RAG system architecture

**Expected Timeline**:
- **Week 9**: Applications submitted (5-10 companies)
- **Week 10**: First recruiter screens + phone interviews
- **Week 11-12**: On-site interview loops
- **Week 13+**: Offer negotiations + decision

**Flexible Approach**:
- Start with 5-10 applications, adjust volume based on response rate
- If high interview load: Reduce new applications, focus on current pipeline
- If low response rate: Expand target list, improve resume/materials, seek more referrals
- Use early interviews as calibration, iterate on weak areas before top choice companies

---

## 📊 Revised Timeline Summary (Updated Day 8 - Week 2 Start)

**Current Status** (End of Week 1 + Day 8):
- ✅ **Interview Readiness**: 75% (B+)
- ✅ **Strong Areas**: ML fundamentals (95%), classical ML (90%), deep learning (85%), NLP/transformers (85%)
- ❌ **Critical Gaps**: LLM systems (0% know, 82% dunno), statistical testing (65%), advanced RAG (30%)
- 🎯 **Target**: 80-85% readiness for ML Engineer roles by Week 4

**Revised Timeline** (Updated after Week 4 Day 7 completion, Nov 24):

| Week | Focus | Goal | Readiness Target | Status |
|------|-------|------|------------------|--------|
| **Week 1** | Algorithm implementations + gap analysis | Validate skills, identify gaps | 75% → Baseline | ✅ Completed |
| **Week 2** | **5 days LLM systems + 2 days statistics** | Interview-ready in LLM systems (60-70%) | 75% → 78% | ✅ Completed (83% achieved) |
| **Week 3** | System design + ML infrastructure + statistics | Strengthen system design to 85% | 78% → 82% | ✅ Completed |
| **Week 4 Day 1-2** | Advanced RAG | Close RAG gap (21% → 55%+) | 82% → 83% | ✅ Completed (99.2% RAG!) |
| **Week 4 Day 3-4** | Gap reassessment | Measure progress, decide next steps | 83% → **85%** | ✅ **COMPLETED** ⭐ |
| **Week 4 Day 5-7** | RAG project planning + setup + Day 1 implementation | Start portfolio project 1 | Maintain 85%+ | ✅ **COMPLETED** |
| **Week 5** | Complete RAG project (hybrid retrieval, generation, evaluation, deploy) | Finish portfolio project 1 | Maintain 85%+ | 🎯 **IN PROGRESS (Day 1 today)** |
| **Week 6** | LeetCode assessment + Neural Network start | Start portfolio project 2 | Maintain 85%+ | 📅 Planned |
| **Week 7** | Neural Network completion + ML coding drills | Complete portfolio project 2 | Maintain 85%+ | 📅 Planned |
| **Week 8** | System design practice + mock interviews | Interview format mastery | 85%+ → 90% | 📅 Planned |
| **Week 9+** | **Job search + interview loops** | Apply and iterate | **90%+ target** | 🎯 **On track** |

**Key Decisions Made**:
1. ✅ **Day 8 Topic Coverage Check** revealed 82% dunno in LLM Systems → foundational learning needed
2. ✅ **Week 2 extended to 5 days** for LLM Systems (was 2 days) - focus on 24 high-impact topics
3. ✅ **Advanced RAG moved to Week 4** (was Week 2 Day 4-5)
4. ✅ **Week 3-4 marked subject to adjustment** based on Week 2 progress
5. ✅ **Gap closure > Projects** - reassessed at Week 4 checkpoint
6. ✅ **Week 4 Day 3-4 Decision**: Start Projects (Option A) - all critical gaps closed ⭐
7. ✅ **Week 4 Day 7-Week 5 (Day 28-Nov 24)**: RAG project started - data loading/FAISS complete
8. ✅ **Phase 2-4 Revamp (Day 29-Nov 25)**: 2 projects + LeetCode + fresh system design problems
9. ✅ **Prompt engineering integrated**: Zero-shot/few-shot testing in Week 5 Day 2-3 (not separate)
10. ✅ **Timeline accelerated**: 12 weeks → 8-9 weeks (job search starts Week 9 instead of Week 11)

**Success Metrics**: 
- **Week 2 (Day 14)**: LLM systems 83% ✅ (exceeded 60-70% target), statistics 81.7% ✅
- **Week 4 (Day 25)**: **85% readiness** ✅ (exceeded 80-85% target)
  - LLM Systems: 89.4% ✅
  - Statistics: 81.7% ✅
  - Advanced RAG: 99.2% ✅✅
  - ML Infrastructure: 90.0% ✅
  - System Design: 86% ✅
- **Week 4 Day 7 (Day 28)**: RAG Day 1 complete - 1541 chunks indexed, search validated ✅
- **Week 5 (ongoing)**: Complete RAG project (Days 2-5)
- **Week 6-8**: 2nd project + interview prep
- **Week 9+**: **Ready to apply to senior-level ML roles** (2 projects complete) 

---

## Key Milestones & Checkpoints (Revised)

**End of Week 4** (✅ ACHIEVED):
- ✅ 85% overall readiness (exceeded 80-85% target)
- ✅ All critical gaps closed (LLM Systems 89%, Statistics 82%, RAG 99%, ML Infrastructure 90%)
- ✅ System design practice (86% readiness)
- ✅ ML fundamentals validated (95%)
- ✅ Decision made: Start portfolio projects

**End of Week 5** (Target: Dec 1):
- [ ] Portfolio project 1 complete (RAG Q&A system)
- [ ] Hybrid retrieval (FAISS + BM25 + RRF)
- [ ] Automated evaluation (Ragas metrics)
- [ ] Docker deployment
- [ ] Comprehensive README and interview talking points

**End of Week 6** (Target: Dec 8):
- [ ] LeetCode proficiency assessed (5 Medium problems)
- [ ] Neural network 60% complete (forward, backward, basic training)
- [ ] Know if intensive LeetCode prep needed

**End of Week 7** (Target: Dec 15):
- [ ] Portfolio project 2 complete (Neural Network from scratch)
- [ ] ML coding speed validated (<15 min logistic regression, <45 min k-means)
- [ ] Can implement neural network from memory in <1 hour
- [ ] 2 portfolio projects ready to showcase

**End of Week 8** (Target: Dec 22):
- [ ] 2 fresh system design solutions (Search Ranking + Model Serving at Scale)
- [ ] 2 mock interviews completed with feedback
- [ ] Behavioral stories polished (8-10 STAR examples)
- [ ] Ready for technical interviews (all formats)

**Week 9** (Target: Dec 29):
- [ ] 5-10 applications submitted
- [ ] Target companies identified
- [ ] Network referrals requested
- [ ] Resume tailored for each company

**Week 10+** (Jan 2025+):
- [ ] Actively interviewing
- [ ] Daily LeetCode if Big Tech interviews (30-45 min)
- [ ] Daily knowledge checks (15 min)
- [ ] Iterating based on feedback

---

## Resources Summary (Revised)

### Portfolio Projects:
1. **RAG Q&A System** - `projects/rag-qa-system/`
   - 32 ArXiv papers corpus
   - Hybrid retrieval (FAISS + BM25 + RRF)
   - Ragas evaluation framework
   - Reference: `references/day28-rag-implementation.md`
2. **Neural Network from Scratch** (Week 6-7)
   - Forward/backward pass implementation
   - MNIST training and visualization
   - Demonstrates ML fundamentals depth

### LeetCode Practice:
- **LeetCode Top Interview Questions**: https://leetcode.com/problem-list/top-interview-questions/
- **Focus patterns**: Arrays, hashmaps, two pointers, trees/graphs, DFS/BFS, DP
- **Company-specific**: Use company tags for Google, Meta, Amazon

### Books & Papers:
- "Machine Learning System Design Interview" by Ali Aminian & Alex Xu
- "Designing Machine Learning Systems" by Chip Huyen
- Key papers already studied (Megatron-LM, ZeRO, FiD, RAPTOR, GraphRAG, Self-RAG, etc.)

### Knowledge Retention:
- **SM-2 Spaced Repetition**: `data/knowledge-schedule.md`
- **Daily Protocol**: `Daily-Knowledge-Check-Protocol.md`
- **Progress Tracking**: `00-CONVERSATION-SUMMARY.md`

### Practice Platforms:
- **LeetCode** - For coding practice
- **Kaggle** - For datasets and competitions
- **Pramp / interviewing.io** - For mock interviews

### Communities:
- r/MachineLearning, r/learnmachinelearning (Reddit)
- ML Discord servers
- HuggingFace community

---

## Interview Preparation Materials

Reference these files throughout your prep:

1. **`ML-System-Design-Questions.md`** - Practice system design problems
2. **`ML-Coding-Questions.md`** - Practice coding problems
3. **`ML-Theory-Questions.md`** - Review theory concepts
4. **`Behavioral-Questions.md`** - Prepare behavioral responses
5. **`Project-Ideas.md`** - Detailed project specifications

---

## Tips for Success

### Consistency > Intensity
- Block time on your calendar
- 1 hour/day is better than 7 hours on Sunday
- Build a habit, don't rely on motivation

### Projects > Passive Learning
- Interviewers care about what you've built
- "Learning by doing" is faster and sticks better
- Each project teaches you 10x more than watching lectures

### Leverage Your Background
- Production ML experience is valuable credibility
- **Continuous ML engagement**: Traditional ML (BERT, neural networks) → Modern AI (LLM/RAG)
- Emphasize: production ML experience + modern AI stack (LLMs, RAG, evaluation frameworks)
- Position as: "From ML model development to ML integration, now returning to model development"
- Hands-on experience across the full ML spectrum

### Interview Strategy
- Start applying in Week 6-7 (don't wait until "ready")
- Apply broadly, practice with companies you care less about first
- Each interview is learning - document and improve
- Don't get discouraged by rejections - it's a numbers game

### Network
- Reach out to former colleagues who moved to ML roles
- LinkedIn outreach to recruiters and hiring managers
- Attend ML meetups (virtual or in-person)
- Contribute to open-source ML projects

---

## Adjustment Guidelines

**If you're moving faster than planned**:
- Add more projects (aim for 5-6 total)
- Go deeper on topics (read papers, implement more algorithms)
- Start applying earlier (Week 4-5)

**If you're moving slower**:
- Focus on breadth over depth
- Prioritize: 3 projects minimum + system design + coding
- Skip optional items (implementing from scratch, etc.)
- Extend timeline to 14-16 weeks

**If you get interviews early**:
- Pause coursework
- Focus 100% on interview prep for that company
- Do targeted practice on their interview format
- Don't decline interviews - treat early ones as practice

---

## Progress Tracking

Use this checklist to track your progress:

### Projects:
- [ ] Project 1: Image Classification
- [ ] Project 2: Text Classification
- [ ] Project 3: Structured Data
- [ ] Project 4: LLM Application

### Interview Skills:
- [ ] Can design ML systems (5+ practice problems done)
- [ ] Can implement ML algorithms from scratch (5+ done)
- [ ] Can answer ML theory questions (reviewed 30+ questions)
- [ ] Prepared behavioral stories (5+ stories ready)

### Applications:
- [ ] Resume updated
- [ ] LinkedIn updated
- [ ] 20+ applications submitted
- [ ] 5+ phone screens completed
- [ ] 3+ onsite/virtual onsites completed

---

**Last Updated**: 2025-11-22 (Week 4 Day 3-4 gap reassessment complete)
**Created for**: Senior SWE → ML Engineer transition
**Timeline**: 12 weeks (adjustable)
**Current Status**: 85% interview readiness achieved, ready for senior-level ML Engineer roles ✅

---

## Daily Knowledge Check Protocol

**NEW: Starting Day 5** (2025-10-31)

At the end of each study day:
1. Complete theory/implementation work
2. **Knowledge Check** (10-15 min):
   - 70% today's content
   - 30% previous content (spaced repetition based on forgetting curve)
3. Create quick reference sheet
4. Update progress files

**Purpose**: Spaced repetition to maximize retention and identify gaps early

**Details**: See `Daily-Knowledge-Check-Protocol.md`
