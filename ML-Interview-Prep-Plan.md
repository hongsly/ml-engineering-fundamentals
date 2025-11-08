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
  - **Clarification needed**: Speculative decoding batch size reasoning (75%) - synchronous vs continuous batching implementations

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
  - Speculative decoding: Better for small batches (GPU utilization), synchronous vs continuous implementations
  - PagedAttention: Block-level memory management, eliminates fragmentation
  - MQA/GQA: Memory savings via shared K/V heads
  - Quantization: GPTQ (Hessian) vs AWQ (activation-aware)

**Target**: Can explain all 4 Q187 methods confidently, understand vLLM architecture ✅ **ACHIEVED**

---

**Day 5: Calculations & Transformer Parameters (2 hours)**

**Topics** (4 topics):
21. FLOPs calculation - forward/backward pass
22. Memory calculation - weights, optimizer, activations
23. **QKV projections** - parameter calculation (Gap Q189)
24. Transformer parameter counting

**Study**:
- [ ] Read: "Transformer Math 101" (EleutherAI or similar) - 40 min
  - URL: https://blog.eleuther.ai/transformer-math/
  - Key concepts: Parameter counting formulas, FLOPs per token

**Practice**:
- [ ] Calculate: Parameters for GPT-2 (124M), GPT-3 (175B) - 40 min
  - Verify: d_model, n_layers, n_heads for both
  - Formula: Attention params, FFN params, embeddings
  - Check against known values
- [ ] Calculate: Memory for training 7B model on A100 (80GB) - 40 min
  - Include: Model weights (FP16), optimizer states (Adam), gradients, activations
  - Determine: Max batch size, whether need model parallelism
  - Practice: Explain reasoning out loud

**Target**: Can calculate transformer params and memory requirements in interview setting

---

**Day 6-7: Statistical Testing (2-3 hours)**

**Pre-study**: Topic Coverage Check (15 min) - if needed for statistics
- [ ] List statistical tests: t-test, z-test, chi-square, ANOVA, regression diagnostics, etc.
- [ ] Self-assess: know/unsure/dunno for each

**Study**:
- [ ] StatQuest: T-test vs Z-test (20 min)
  - When to use: n < 30 vs n ≥ 30, known vs unknown variance
  - URL: https://www.youtube.com/watch?v=0Pd3dc1GcHc (or similar)
- [ ] StatQuest or read: Linear Regression Assumptions (30 min)
  - 5 assumptions: Linearity, Independence, Homoscedasticity, Normality, No Multicollinearity
  - Tests: Residual plots, Durbin-Watson, Breusch-Pagan, Shapiro-Wilk/Q-Q plot, VIF

**Practice**:
- [ ] Run all 5 regression assumption tests on sample data (45 min)
  - Use `statsmodels` library
  - Interpret p-values, understand when assumptions violated
- [ ] Create cheat sheet: Test names, when to use, interpretation (30 min)

**Target**: Bring statistical testing from 65% → 90%, can explain t-test vs z-test, run all 5 regression tests

---

**Week 2 Expected Outcomes**:
- ✅ **LLM Systems**: 60-70% interview ready (24/76 topics covered, highest-impact)
  - Can explain: Data/model/tensor parallelism, ZeRO, FSDP, KV-cache, quantization, batching
  - Can calculate: Transformer parameters, FLOPs, memory requirements
  - Can discuss: 3 key papers (Megatron-LM, ZeRO, vLLM)
- ✅ **Statistical Testing**: 90% ready (from 65%)
- ⏸️ **Advanced RAG**: Deferred to Week 3 or later
- ⏸️ **ML Evaluation**: Deferred to Week 3 or later

---

### Week 3: System Design + Advanced RAG ⚠️ **SUBJECT TO ADJUSTMENT**

**Goal**: Strengthen system design from 70% → 85%, add Advanced RAG (deferred from Week 2)

**Context**: Week 3 plan is flexible based on Week 2 progress. If LLM Systems takes longer or additional gaps emerge, this schedule will adjust.

---

**Day 1: PyTorch Basics (2-3 hours)** ⭐ **NEW - For OpenAI/Anthropic/DeepMind interviews**

**Rationale**: Basic PyTorch literacy needed for research-heavy ML roles. Goal is reading comprehension, not production coding.

**Study** (1 hour):
- [ ] PyTorch Quickstart Tutorial (30 min)
  - URL: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
  - Focus: Tensors, Dataset/DataLoader, nn.Module structure
- [ ] Autograd Mechanics (30 min)
  - URL: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
  - Understand: .backward(), .grad, computational graph

**Practice** (1.5 hours):
- [ ] Implement logistic regression in PyTorch (60 min)
  - Define nn.Module class
  - Forward pass, loss calculation, optimizer step
  - Compare to your Week 1 Day 2 numpy implementation
- [ ] Read PyTorch FSDP code (30 min)
  - URL: https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py
  - Goal: Recognize patterns from Week 2 Day 3 (all-gather, reduce-scatter)
  - Don't need to understand every line—just high-level structure

**Target**: Basic PyTorch literacy (can read code in interviews, understand distributed patterns)

---

**Day 2-3: ML Technology Deep Dives (4-5 hours)**

**Pre-study**: Topic Coverage Check (20 min)
- [ ] List ML infra technologies: Kafka, Flink, Airflow, Feature stores, Model serving, etc.
- [ ] Self-assess: know/unsure/dunno for each

**Learning**:
- [ ] Kafka basics (1 hour)
  - Use cases: Event streaming, real-time data pipelines
  - Core concepts: Topics, partitions, producers, consumers
- [ ] Flink basics (1 hour)
  - Use cases: Stream processing, stateful computations
  - When to use vs Spark Streaming
- [ ] Airflow basics (1 hour)
  - Use cases: Workflow orchestration, ML pipelines
  - DAGs, operators, scheduling
- [ ] Feature stores (1-1.5 hours)
  - Architectures: Feast, Tecton
  - Online vs offline stores, feature serving

**Practice**:
- [ ] Design: Feature store for recommendation system (30 min)
  - Real-time features (last hour clicks) + batch features (user demographics)
  - Serving architecture

---

**Day 4-5: System Design Practice (3-4 hours)**

**Practice Problems** (3-4 problems × 45-60 min each):
- [ ] Problem 1: Design YouTube recommendation system
  - Focus: Candidate generation, ranking, serving architecture
- [ ] Problem 2: Design fraud detection system
  - Focus: Real-time inference, feature engineering, model monitoring
- [ ] Problem 3: Design search ranking system
  - Focus: Retrieval, ranking stages, personalization
- [ ] Problem 4 (optional): Design ad click prediction system
  - Focus: Feature engineering, two-tower model, online learning

**For each problem**:
- [ ] Write full solution (data, features, model, serving, monitoring)
- [ ] Draw architecture diagram
- [ ] Compare with sample solutions (ML-System-Design-Questions.md)

**Memorize**:
- [ ] Common architectures: Two-tower, cascade, lambda architecture
- [ ] Scale numbers: QPS targets (1K, 10K, 100K+), latency (p50, p99), throughput

---

**Day 6-7: Advanced RAG Architectures (2-3 hours)** ⭐ **MOVED FROM WEEK 2**

**Pre-study**: Topic Coverage Check (20 min) - if needed
- [ ] List RAG subtopics: Retrieval methods (sparse, dense, hybrid), Reranking, FiD, ColBERT, DPR, etc.
- [ ] Self-assess: know/unsure/dunno for each

**Study**:
- [ ] Read: FiD paper ("Leveraging Passage Retrieval with Generative Models for Open Domain QA") (1 hour)
  - Focus: Architecture (encode independently, decode jointly), advantages over concatenation
- [ ] Read: Hybrid retrieval overview (30 min)
  - Sparse (BM25) + Dense (embeddings) fusion
  - Reciprocal Rank Fusion (RRF) algorithm

**Practice**:
- [ ] Implement: Simple hybrid retrieval (1 hour)
  - BM25 (using `rank_bm25` library) + Sentence-Transformers
  - RRF fusion: score(d) = Σ 1/(k + rank_r(d))
- [ ] Explore: LlamaIndex or LangChain hybrid retrieval examples (30 min)

**Target**: Bring advanced RAG from 30% → 75%

---

**Week 3 Note**: Schedule is flexible. If Week 2 LLM Systems extends into Week 3 Days 1-2, push System Design and Advanced RAG accordingly. Priority is quality over strict timeline.

---

### Week 4: Progress Check & Adaptive Planning ⚠️ **SUBJECT TO ADJUSTMENT**

**Goal**: Assess gap closure progress and decide next steps

**Context**: Week 4 serves as checkpoint and buffer. Actual activities depend heavily on Week 2-3 progress and any emerging gaps.

---

**Day 1-2: Gap Re-assessment (2-3 hours)**

**Activity**:
- [ ] Re-test on weak areas from Day 6-7 (select 30-40 questions)
  - LLM systems (Q182-189) - expect 60-70% → target 80%+
  - Statistical testing (Q181-182) - expect 90%+
  - Advanced RAG (Q177-179) - expect 75%+
  - System design (Q145-160) - expect 85%+
- [ ] Calculate improvement: Before vs after Week 2-3 study
- [ ] Identify remaining gaps

**Decision Point**: Based on re-assessment results
- **If critical gaps closed (80%+ across all areas)**: ✅ Proceed to Option A (projects or ML evaluation)
- **If gaps remain (< 80% in any critical area)**: 🔄 Proceed to Option B (continue gap closure)

---

**Day 3-7: Option A - Projects or Additional Topics (if gaps closed)**

**Choice 1: Start Projects**
- [ ] Image Classification with transfer learning (ResNet/EfficientNet)
- [ ] Deploy to HuggingFace Spaces or Gradio
- [ ] Clean code, README, evaluation metrics

**OR Choice 2: ML Evaluation & Problem Reframing (2-3 hours)**
- [ ] Read: BERTScore paper (Zhang et al., 2020) - focus on formula, when to use
- [ ] Watch: Fairness in ML talk - demographic parity, equalized odds
- [ ] List 10 problem reframing examples from production ML experience
- [ ] Create: Personal evaluation metrics cheat sheet

**OR**

**Day 3-7: Option B - Continue Gap Closure (if gaps remain)**

**Flexible study based on remaining gaps**:
- [ ] More LLM systems topics (if < 70%): Flash Attention, pipeline parallelism, gradient checkpointing, data loading optimization
- [ ] More statistical tests (if < 90%): Chi-square, ANOVA, non-parametric tests
- [ ] More RAG architectures (if < 75%): ColBERT, DPR, query expansion, reranking strategies
- [ ] More system design practice (if < 85%): 2-3 additional problems

---

**Phase 1 Revised Deliverables** (End of Week 4):
- ✅ Critical gaps closed (target: 80%+ in all high-priority areas)
- ✅ Interview readiness: 80-85% overall (up from 75%)
- ✅ Strong system design skills (can design 3-5 ML systems)
- 🔄 Projects: 0-1 projects (deferred to Phase 2 if needed - this is acceptable)

**Key Success Metric**: Confidence to start applying to mid-level ML roles by Week 5-6

**Key Insight**: **Gap closure > Projects** for senior roles. Projects demonstrate skills but don't close knowledge gaps.

---

## Phase 2: Modern AI & LLMs (Weeks 5-6)

⚠️ **SUBJECT TO CHANGE** based on Week 4 progress check
- If gaps remain after Week 4: Continue gap closure, defer Phase 2
- If gaps closed (85%+ readiness): Proceed with Phase 2 as planned OR compress/skip if ready to interview

**Original Goal**: Get current with LLM technology and build modern AI applications
**Time**: 10-20 hours total

**Gap Analysis Update**: Day 6-7 revealed we're already strong on LLM *applications* (75-85%) but weak on LLM *systems* (30%). Phase 2 may need to focus on systems-level topics instead of application-level.

### Week 5: LLM Fundamentals + Prompt Engineering

**Learning (3-4 hours)**:
- [ ] DeepLearning.AI: "ChatGPT Prompt Engineering for Developers" (2 hours)
- [ ] DeepLearning.AI: "Building Systems with ChatGPT API" (1-2 hours)
- [ ] HuggingFace Course Chapter 1: Transformer models
- [ ] Read: "Attention is All You Need" paper (at least abstract + intro)

**Hands-on (3-4 hours)**:
- [ ] Experiment with OpenAI API / Anthropic API
- [ ] Build simple chatbot with system prompts
- [ ] Try few-shot learning and prompt optimization
- [ ] Start Project 4: LLM-based application

**Resources**:
- DeepLearning.AI Short Courses: https://www.deeplearning.ai/short-courses/
- HuggingFace Course: https://huggingface.co/learn/nlp-course/
- OpenAI Cookbook: https://cookbook.openai.com/

---

### Week 6: RAG, Vector DBs & Fine-tuning

**Learning (3-4 hours)**:
- [ ] DeepLearning.AI: "LangChain for LLM Application Development"
- [ ] DeepLearning.AI: "Vector Databases from Embeddings to Applications"
- [ ] HuggingFace Course Chapter 2: Using transformers
- [ ] Read about RAG (Retrieval-Augmented Generation)

**Hands-on (4-5 hours)**:
- [ ] Complete Project 4: RAG system or fine-tuned model
  - Option A: Document Q&A system with vector DB
  - Option B: Fine-tune small model (BERT/DistilBERT) for specific task
  - Option C: Multi-agent system with LangChain
- [ ] Polish Project 4 for portfolio

**Phase 2 Deliverable**:
- ✅ 1 modern LLM application on GitHub
- ✅ Can discuss: transformers, attention, embeddings, RAG, fine-tuning
- ✅ Familiar with LangChain, vector DBs, prompt engineering

---

## Phase 3: Interview Prep (Weeks 7-10)

⚠️ **SUBJECT TO CHANGE** based on Week 4-6 progress
- May start earlier (Week 5-6) if gaps close fast
- System design already practiced in Week 3, ML coding validated in Week 1
- Main focus: Mock interviews and application preparation

**Goal**: Master interview formats and build interview-specific skills
**Time**: 28-40 hours total (may be compressed to 2-3 weeks)

### Week 7: ML System Design (Part 1)

**Learning (2-3 hours)**:
- [ ] Read "Machine Learning System Design Interview" (Chapters 1-3)
- [ ] Study ML system design framework
- [ ] Watch example system design interviews on YouTube

**Practice (4-5 hours)**:
- [ ] Practice problem: Design a video recommendation system
- [ ] Practice problem: Design a search ranking system
- [ ] Write out full solutions with diagrams
- [ ] Review sample solutions and compare

**Topics to master**:
- Data collection and labeling
- Feature engineering pipeline
- Model selection and training
- Evaluation metrics and A/B testing
- Deployment and serving architecture
- Monitoring and retraining

---

### Week 8: ML System Design (Part 2) + ML Coding Prep

**System Design Practice (3-4 hours)**:
- [ ] Practice problem: Design a fraud detection system
- [ ] Practice problem: Design a feed ranking system (e.g., Facebook/LinkedIn)
- [ ] Record yourself explaining solutions (simulate interview)

**ML Coding Practice (3-4 hours)**:
- [ ] Implement linear regression from scratch (no sklearn)
- [ ] Implement logistic regression from scratch
- [ ] Implement k-means clustering from scratch
- [ ] Implement decision tree from scratch (optional)
- [ ] Practice on LeetCode ML problems (if available)

**Resources**:
- ML System Design questions: See `ML-System-Design-Questions.md`
- ML Coding questions: See `ML-Coding-Questions.md`

---

### Week 9: ML Theory + Algorithm Implementation

**Theory Review (3-4 hours)**:
- [ ] Review bias-variance tradeoff
- [ ] Review regularization (L1, L2, dropout)
- [ ] Review optimization algorithms (SGD, Adam, etc.)
- [ ] Review evaluation metrics (precision, recall, F1, AUC-ROC, etc.)
- [ ] Review common ML algorithms and when to use them
- [ ] Go through 30+ theory questions in `ML-Theory-Questions.md`

**Coding Practice (3-4 hours)**:
- [ ] **Implement neural network with backpropagation from scratch** (2-3 hours priority)
  - **Rationale**: Week 2 Day 9 practice exercises revealed need for hands-on coding beyond conceptual derivation
  - Forward pass: Input → Hidden layer(s) → Output
  - Backward pass: Derive all gradients (∂L/∂W, ∂L/∂b) using chain rule
  - Manual gradient check: Compare analytical gradients to numerical gradients
  - Test on simple dataset (XOR or MNIST)
  - **Goal**: Solidify backprop understanding with actual implementation
- [ ] Implement cross-validation from scratch
- [ ] Practice feature engineering problems
- [ ] Practice data preprocessing tasks (handling missing data, scaling, encoding)

---

### Week 10: Mock Interviews + Portfolio Polish

**Mock Interviews (4-5 hours)**:
- [ ] Schedule 2-3 mock interviews on Pramp or interviewing.io
- [ ] Do at least one ML system design mock
- [ ] Do at least one ML coding mock
- [ ] Record feedback and areas for improvement

**Portfolio Work (3-4 hours)**:
- [ ] Final polish on all 4 projects
- [ ] Write comprehensive READMEs with:
  - Problem statement
  - Approach and methodology
  - Results and evaluation
  - Technologies used
  - Future improvements
- [ ] Add visualizations and demo gifs/screenshots
- [ ] Ensure code is clean and well-commented

**Resume & LinkedIn (1-2 hours)**:
- [ ] Update resume with ML projects
- [ ] Update LinkedIn with skills and projects
- [ ] Prepare 2-minute elevator pitch

---

## Phase 4: Active Job Search (Weeks 11-12+)

⚠️ **SUBJECT TO CHANGE** - May start as early as Week 8-9 if readiness hits 85%+
- Gap analysis showed 75% readiness after Week 1
- Target: 85%+ after Weeks 2-4 (gap closure)
- Can start applying to mid-level roles while continuing senior-level prep

**Goal**: Apply to jobs, interview, and iterate based on feedback
**Time**: 10-20+ hours/week

**Flexible Strategy**:
- **Week 8-10**: Start applications to mid-level ML Engineer roles (already 75% ready)
- **Week 10-12+**: Apply to senior roles after closing LLM systems gap
- Use early interviews as practice, iterate based on feedback

### Week 11-12: Applications + Interviews

**Applications (2-3 hours/week)**:
- [ ] Apply to 10-15 positions per week
- [ ] Target: ML Engineer, Applied Scientist, AI Engineer roles
- [ ] Track applications in spreadsheet

**Interview Practice (3-4 hours/week)**:
- [ ] Continue mock interviews
- [ ] Review and practice weak areas
- [ ] Practice explaining projects clearly and concisely
- [ ] Prepare behavioral stories using STAR format

**Ongoing Learning (2-3 hours/week)**:
- [ ] Stay current: read ML papers, blog posts
- [ ] Follow ML on Twitter/LinkedIn
- [ ] Participate in ML communities (Reddit, Discord)
- [ ] Continue Fast.AI or other courses if time permits

**After Each Interview**:
- [ ] Document questions asked
- [ ] Identify knowledge gaps
- [ ] Study gaps immediately
- [ ] Update interview prep materials

---

## 📊 Revised Timeline Summary (Updated Day 8 - Week 2 Start)

**Current Status** (End of Week 1 + Day 8):
- ✅ **Interview Readiness**: 75% (B+)
- ✅ **Strong Areas**: ML fundamentals (95%), classical ML (90%), deep learning (85%), NLP/transformers (85%)
- ❌ **Critical Gaps**: LLM systems (0% know, 82% dunno), statistical testing (65%), advanced RAG (30%)
- 🎯 **Target**: 80-85% readiness for ML Engineer roles by Week 4

**Revised Timeline** (Updated based on LLM Systems topic check):

| Week | Focus | Goal | Readiness Target | Notes |
|------|-------|------|------------------|-------|
| **Week 1** ✅ | Algorithm implementations + gap analysis | Validate skills, identify gaps | 75% → Baseline | Completed |
| **Week 2** 🔄 | **5 days LLM systems + 2 days statistics** | Interview-ready in LLM systems (60-70%) | 75% → 78% | **Extended from 2 to 5 days** |
| **Week 3** ⚠️ | System design + Advanced RAG | Strengthen system design to 85% | 78% → 82% | **Subject to adjustment** |
| **Week 4** ⚠️ | Progress check + adaptive (gaps OR projects) | Close remaining gaps OR start projects | 82% → 85% | **Checkpoint + buffer** |
| **Week 5-6** ⚠️ | Adaptive (projects OR advanced topics) | Build portfolio OR continue learning | Maintain 85%+ | Depends on Week 4 results |
| **Week 7-10** | Mock interviews + applications | Practice + iterate | 85%+ → 90% | Subject to change |
| **Week 8+** | Start job search (mid-level roles) | Apply while studying | - | If 85%+ achieved |
| **Week 10+** | Apply to senior roles | Interview at target companies | 90%+ | Subject to change |

**Key Decisions Made**:
1. ✅ **Day 8 Topic Coverage Check** revealed 82% dunno in LLM Systems → foundational learning needed
2. ✅ **Week 2 extended to 5 days** for LLM Systems (was 2 days) - focus on 24 high-impact topics
3. ✅ **Advanced RAG moved to Week 3** (was Week 2 Day 4-5)
4. ✅ **Week 3-4 marked subject to adjustment** based on Week 2 progress
5. ✅ **Gap closure > Projects** - will reassess at Week 4 checkpoint

**Success Metrics**:
- **Week 2 (Day 14)**: LLM systems 60-70% (from 0%), statistics 90% (from 65%)
- **Week 4 (Day 28)**: 80-85%+ readiness across all critical areas
- **Week 6**: Portfolio with 1-2 projects (if time permits and gaps closed)
- **Week 10+**: Actively interviewing at target companies (if 85%+ achieved)

---

## Weekly Schedule Template

### For 7-8 hours/week:

**Weekdays (3-4 hours)**:
- Monday: 1 hour - Watch lectures/read materials
- Tuesday: 1 hour - Coding exercises or practice problems
- Wednesday: 1 hour - Watch lectures/read materials
- Thursday: 1-2 hours - Coding exercises or practice problems

**Weekend (4-5 hours)**:
- Saturday: 2-3 hours - Project work
- Sunday: 2 hours - Project work or interview practice (later phases)

### For 10 hours/week:

Add:
- Friday: 1 hour - Review week's learnings, plan next week
- Weekend: +1-2 hours more on projects

---

## Key Milestones & Checkpoints

**End of Week 4**:
- ✅ 3 ML projects completed
- ✅ Comfortable with ML pipeline end-to-end
- ✅ Can train, evaluate, and compare models

**End of Week 6**:
- ✅ 4 projects total (including 1 LLM project)
- ✅ Can discuss modern AI/LLM concepts
- ✅ Portfolio ready to show

**End of Week 8**:
- ✅ Can design ML systems for common problems
- ✅ Can implement ML algorithms from scratch
- ✅ Ready for technical interviews

**End of Week 10**:
- ✅ Completed mock interviews
- ✅ Resume and LinkedIn updated
- ✅ Ready to apply

**Week 11+**:
- ✅ Actively interviewing
- ✅ Iterating based on feedback

---

## Resources Summary

### Primary Courses:
1. **Fast.AI Practical Deep Learning** - https://course.fast.ai/
2. **DeepLearning.AI Short Courses** - https://www.deeplearning.ai/short-courses/
3. **HuggingFace NLP Course** - https://huggingface.co/learn/nlp-course/

### Supplementary:
4. **Stanford CS229** - https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU
5. **DeepLearning.AI ML Specialization** (Coursera) - If you need more structure

### Books & Readings:
- "Machine Learning System Design Interview" by Ali Aminian & Alex Xu
- "Designing Machine Learning Systems" by Chip Huyen
- Papers: "Attention is All You Need", "BERT", "GPT" papers (at least abstracts)

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

## Next Steps

1. **Start today** with Week 1, Day 1 tasks
2. **Set up tracking** - Use this document or create a Notion/spreadsheet
3. **Block calendar** - Schedule your 7-10 hours/week now
4. **Join communities** - Get support and accountability
5. **Tell someone** - Accountability partner or mentor

---

**Last Updated**: 2025-10-31
**Created for**: Senior SWE → ML Engineer transition
**Timeline**: 12 weeks (adjustable)

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
