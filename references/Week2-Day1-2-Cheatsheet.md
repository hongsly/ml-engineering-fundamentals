# Week 2 Day 1-2: Distributed Training Cheat Sheet
**Topics**: Megatron-LM (Tensor Parallelism) + ZeRO (Memory Optimization)

---

## **1. Three Main Parallelism Strategies**

| Type | What It Does | Memory Reduction | Communication | Best For |
|------|-------------|------------------|---------------|----------|
| **Data Parallelism (DP)** | Replicate model, split data | None (replicates everything) | 2Ψ per step (all-reduce gradients) | Models that fit in 1 GPU |
| **Model Parallelism (MP)** | Split model layers/parameters | Nm× (MP degree) | High (per layer, ~2Ψ per layer) | Models too large for 1 GPU |
| **Pipeline Parallelism (PP)** | Split model into stages (horizontal) | Np× (PP degree) | Low (only between stages) | Very large models + reduce bubble |

**Key Insight**: DP is compute-efficient but memory-inefficient. MP is memory-efficient but compute-inefficient (especially cross-node).

---

## **2. Megatron-LM: Tensor Parallelism**

### **What It Does:**
Splits **within a layer** across GPUs (intra-layer parallelism, a form of MP)

### **Two Patterns:**

#### **A. MLP Block (Column-Parallel → Row-Parallel):**

**Formulation**: `Y = GeLU(X @ W1)`, then `Z = Dropout(Y @ W2)`

```
Input X: [batch, seq, 1024]

--- Step 1: First Linear + GeLU (Column-Parallel) ---
W1 [1024, 4096] split by COLUMNS (output dimension):

GPU-0: W1_left  [1024, 2048]  (first 2048 output neurons)
GPU-1: W1_right [1024, 2048]  (last 2048 output neurons)

GPU-0: Y1 = GeLU(X @ W1_left)  = [batch, seq, 2048]
GPU-1: Y2 = GeLU(X @ W1_right) = [batch, seq, 2048]

→ GeLU applied INDEPENDENTLY on each GPU (no communication!)
→ [Y1 | Y2] = [batch, seq, 4096] (implicit concatenation)

--- Step 2: Second Linear + Dropout (Row-Parallel) ---
W2 [4096, 1024] split by ROWS (input dimension):

GPU-0: W2_top [2048, 1024]  (top 2048 rows)
GPU-1: W2_bot [2048, 1024]  (bottom 2048 rows)

GPU-0: Z1 = Y1 @ W2_top = [batch, seq, 1024]  (partial sum)
GPU-1: Z2 = Y2 @ W2_bot = [batch, seq, 1024]  (partial sum)

→ All-Reduce: Z = Z1 + Z2 = [batch, seq, 1024]
→ Dropout(Z)  (applied after all-reduce)
```

**Communication**: 1 all-reduce (after row-parallel W2, before dropout)

**Why column-parallel for W1**: Allows GeLU to be applied independently on each GPU (no communication before activation)

#### **B. Self-Attention (Split Heads → All-Reduce After Projection):**
```
Input X: [batch, seq, 1024]

GPU-0: Handles heads 0-7   → Attn1 [batch, seq, 512]
GPU-1: Handles heads 8-15  → Attn2 [batch, seq, 512]

W_O projection (split by ROWS, not columns!):
GPU-0: W_O1 [512, 1024] → Y1 = Attn1 @ W_O1 = [batch, seq, 1024]  (partial)
GPU-1: W_O2 [512, 1024] → Y2 = Attn2 @ W_O2 = [batch, seq, 1024]  (partial)

→ All-Reduce: Y_full = Y1 + Y2  [batch, seq, 1024]
```

**Why All-Reduce Needed**: Y1 and Y2 are **partial sums** (not concatenation). Dropout + residual connection require the **full tensor** Y_full.

**Communication**: 1 all-reduce (after attention output projection)

### **Key Metrics:**
- **Largest model**: 8.3B parameters (Megatron-LM paper)
- **Efficiency**: Good within node (NVLink 300-600 GB/s), degrades cross-node (InfiniBand 12.5 GB/s)
- **Communication (Forward only)**: 2 all-reduce per transformer block (attention + MLP)

### **Backward Pass Communication:**

**Total**: 4 all-reduces per transformer layer (2 forward + 2 backward)

#### **f and g Operators (Conjugate Pairs):**

```python
# f operator: Identity forward, all-reduce backward
class f(torch.autograd.Function):
    def forward(ctx, x):
        return x  # No communication
    def backward(ctx, grad):
        all_reduce(grad)  # Sum gradients across GPUs
        return grad

# g operator: All-reduce forward, identity backward
class g(torch.autograd.Function):
    def forward(ctx, x):
        all_reduce(x)  # Sum activations across GPUs
        return x
    def backward(ctx, grad):
        return grad  # No communication
```

#### **MLP Block Backward:**

```
Forward:
  X (replicated) → f → [W1_left | W1_right] (column-parallel) → GeLU → Y
  Y (partitioned: Y1, Y2) → [W2_top | W2_bot] (row-parallel) → g → Z (replicated)

Backward:
  ∂L/∂Z (replicated) → g.backward (identity) → [W2_top^T | W2_bot^T]
  → ∂L/∂Y (partitioned: ∂L/∂Y1, ∂L/∂Y2)
  → GeLU' → [W1_left^T | W1_right^T] → f.backward (all-reduce) → ∂L/∂X (replicated)
```

**Why all-reduce in backward?**
- **After W2 (row-parallel)**: No all-reduce needed (g.backward is identity, gradients already partitioned)
- **After W1 (column-parallel)**: All-reduce needed (f.backward sums partial gradients)
  - GPU-0: ∂L/∂X₁ = ∂L/∂Y1 @ W1_left^T (partial gradient w.r.t. replicated X)
  - GPU-1: ∂L/∂X₂ = ∂L/∂Y2 @ W1_right^T (partial gradient w.r.t. replicated X)
  - ∂L/∂X = ∂L/∂X₁ + ∂L/∂X₂ (all-reduce sum, needed for previous layer)

**Key Insight**: Activations are **replicated** within tensor-parallel group (not partitioned like in data parallelism). When weights are column-partitioned, each GPU computes a partial gradient w.r.t. the replicated input → need all-reduce to sum partials.

#### **Self-Attention Backward:**

```
Forward:
  X → [Q, K, V] (column-parallel) → Attention (partitioned heads) → W_O (row-parallel) → g → Y

Backward:
  ∂L/∂Y → g.backward (identity) → W_O^T (row-parallel) → Attention backward
  → [Q^T, K^T, V^T] (column-parallel) → f.backward (all-reduce) → ∂L/∂X
```

**Communication**: 1 all-reduce (after Q/K/V projection backprop, before previous layer)

---

## **3. ZeRO: Zero Redundancy Optimizer**

### **Core Idea:**
Eliminate memory redundancy in Data Parallelism by **partitioning** instead of **replicating** model states.

### **Three Stages (Cumulative):**

| Stage | What It Partitions | Memory Reduction | Communication Volume | When to Use |
|-------|-------------------|------------------|----------------------|-------------|
| **Pos (Stage 1)** | Optimizer states | 4× | 2Ψ (same as DP) | Always (free memory, no overhead) |
| **Pos+g (Stage 2)** | + Gradients | 8× | 2Ψ (same as DP) | Always (free memory, no overhead) |
| **Pos+g+p (Stage 3)** | + Parameters | Nd× (DP degree) | 3Ψ (1.5× more) | When Nd is large (>64 GPUs) |

**Memory Formula:**
- **Baseline DP**: (2 + 2 + K)Ψ = 16Ψ bytes (for Adam, K=12)
  - fp16 params: 2Ψ
  - fp16 gradients: 2Ψ
  - fp32 optimizer states (params + momentum + variance): 12Ψ
- **Pos**: 4Ψ + KΨ/Nd = 4Ψ + 12Ψ/Nd
- **Pos+g**: 2Ψ + 14Ψ/Nd
- **Pos+g+p**: 16Ψ/Nd (for large Nd ≈ 0!)

### **ZeRO-DP Communication Analysis:**

#### **Stage 2 (Pos+g):**
```
Baseline DP:
  Reduce-scatter gradients: Ψ
  All-gather parameters: Ψ
  Total: 2Ψ

ZeRO Pos+g:
  Reduce-scatter gradients: Ψ  (only reduce to owning GPU)
  All-gather parameters: Ψ     (after optimizer step)
  Total: 2Ψ (SAME as baseline!)
```

#### **Stage 3 (Pos+g+p):**
```
ZeRO Pos+g+p:
  Reduce-scatter gradients: Ψ
  All-gather parameters (forward): Ψ   (broadcast before each layer)
  All-gather parameters (backward): Ψ  (broadcast again in reverse)
  Total: 3Ψ (1.5× baseline)
```

**Key Insight**: Pos+g has **zero communication overhead**. Pos+g+p trades 50% more communication for Nd× memory reduction.

---

## **4. ZeRO-R: Residual Memory Optimization**

### **Three Optimizations:**

| Optimization | What It Does | Memory Saving | When to Use |
|--------------|-------------|---------------|-------------|
| **Pa (Partitioned Activation Checkpointing)** | Partition activations across MP GPUs | Nm× (MP degree) | When using MP + large batch |
| **CB (Constant Buffers)** | Use fixed-size buffers (not model-size-dependent) | Constant overhead | Always |
| **MD (Memory Defragmentation)** | Pre-allocate contiguous memory for activations/gradients | Reduces fragmentation | Always |

**Pa Details:**
- MP replicates activations (each GPU needs full activations for its partition)
- Pa partitions activations, uses all-gather to reconstruct on-demand
- **Communication overhead**: <10% of MP communication (1 all-gather per transformer block vs 2 all-reduces)
- **Pa+cpu**: Offload partitioned activations to CPU (for extreme memory pressure)

---

## **5. 3D Parallelism (ZeRO + MP + PP)**

### **Memory Reduction:**
```
Total reduction = Nd × Nm × Np

Example (1024 GPUs = 64 DP × 16 MP × 1 PP):
  - ZeRO Pos+g+p: 64× (DP degree)
  - Tensor Parallel: 16× (MP degree)
  - Total: 1024× memory reduction!
```

### **Communication Patterns:**

| Parallelism | When | Who Talks to Whom | Volume |
|-------------|------|-------------------|--------|
| **ZeRO-DP (Pos+g)** | Backward: reduce-scatter gradients + all-gather params after optimizer | All GPUs in same DP group | 2Ψ |
| **ZeRO-DP (Pos+g+p)** | Forward: all-gather params per layer<br>Backward: reduce-scatter gradients + all-gather params after optimizer | All GPUs in same DP group | 3Ψ (1Ψ forward + 2Ψ backward) |
| **Tensor Parallel (MP)** | Every layer (forward + backward) | All GPUs in same MP group | ~4Ψ per layer (2 forward + 2 backward) |
| **Pipeline Parallel (PP)** | Between stages (forward + backward) | Adjacent stages only | Activations/gradients |

**Key Insight**: Adding PP to DP+MP naturally staggers DP communication - each pipeline stage finishes backward at different times, so DP synchronization happens at different times → less network contention → higher effective bandwidth. Without PP, all DP groups synchronize simultaneously, causing network congestion.

---

## **6. When to Use What?**

| Model Size | Configuration | Reasoning |
|------------|---------------|-----------|
| **<1.5B** | Pure DP | Fits in single GPU (32GB) |
| **1.5B-13B** | ZeRO Pos+g (pure DP) | ZeRO eliminates memory redundancy, no MP needed |
| **13B-100B** | ZeRO Pos+g + MP (intra-node) | Activation memory requires MP, ZeRO for model states |
| **100B-1T** | ZeRO Pos+g+p + MP + PP (3D) | All three dimensions needed |

**Decision Factors:**
1. **Activation memory bottleneck?** → Add MP (within node)
2. **Batch size too large?** → Add MP (reduces DP degree)
3. **Model too large for Pos+g?** → Add Pos+g+p (trades 50% comm for Nd× memory)
4. **Still OOM?** → Add PP (split into stages)

---

## **7. Key Interview Questions & Answers**

### **Q1: Why does self-attention need all-reduce after W_O projection?**
**A**: Because W_O is split by **rows** (not columns):
- GPU-0: Y1 = Attn1 @ W_O1 (partial sum)
- GPU-1: Y2 = Attn2 @ W_O2 (partial sum)
- Need Y_full = Y1 + Y2 (all-reduce) before dropout/residual

### **Q2: What's the communication overhead of ZeRO?**
**A**:
- Pos+g: **0% overhead** (2Ψ, same as baseline DP)
- Pos+g+p: **50% overhead** (3Ψ vs 2Ψ baseline)
- Trade-off: 50% more comm for Nd× memory reduction

### **Q3: Why is MP inefficient cross-node?**
**A**: MP requires all-reduce **per layer** (100+ times per step for 50-layer model):
- Intra-node (NVLink): 300-600 GB/s → Fast ✅
- Cross-node (InfiniBand): 12.5 GB/s → Slow ❌
- ZeRO only communicates **once per step** → Cross-node friendly

### **Q4: When should you use MP instead of ZeRO?**
**A**:
1. **Activation memory bottleneck** (very large models, long sequences)
2. **Batch size constraint** (DP alone causes batch too large for convergence)
3. **Within a single node** (MP is efficient with NVLink)

### **Q5: How does ZeRO achieve memory efficiency without hurting compute efficiency?**
**A**:
- **Partitions** model states (eliminates redundancy) → Memory efficient
- **Dynamic communication schedule** (all-gather on-demand) → Maintains computational granularity
- **Same communication volume as DP** (Pos+g) → No efficiency loss

### **Q6: What's the difference between tensor parallelism and pipeline parallelism?**
**A**:
- **Tensor Parallel (MP)**: Splits **within layers** (intra-layer), high communication (per layer)
- **Pipeline Parallel (PP)**: Splits **across layers** (horizontal), low communication (only between stages)

### **Q7: Why is all-reduce needed in backward pass after column-parallel layers?**
**A**: Because activations are **replicated** (not partitioned) within tensor-parallel group:
- Each GPU has full input X, but only a partition of weights (W1_left or W1_right)
- Backward: Each GPU computes ∂L/∂X_partial w.r.t. its weight partition
- Previous layer needs **full gradient** ∂L/∂X = sum of all partials → all-reduce
- Row-parallel layers don't need backward all-reduce (g.backward is identity)

---

## **8. Quick Reference Formulas**

### **Memory Consumption (Mixed-Precision Adam):**
```
Baseline DP:           16Ψ bytes
ZeRO Pos:             4Ψ + 12Ψ/Nd
ZeRO Pos+g:           2Ψ + 14Ψ/Nd
ZeRO Pos+g+p:         16Ψ/Nd
Tensor Parallel:      16Ψ/Nm
3D (ZeRO+MP+PP):      16Ψ/(Nd × Nm × Np)
```

### **Communication Volume (per training step):**
```
Baseline DP:          2Ψ (Ψ = total model parameters)
ZeRO Pos:            2Ψ (same as baseline)
ZeRO Pos+g:          2Ψ (same as baseline)
ZeRO Pos+g+p:        3Ψ (1.5× overhead)

Tensor Parallel:     4 all-reduces per layer (2 forward + 2 backward)
                     Volume = 4 × batch × seq × hidden per layer
                     Example: batch=32, seq=512, hidden=1024, 50 layers
                              = 4 × 32 × 512 × 1024 × 2 bytes × 50
                              = 12.8 GB per training step

                     Note: TP communicates activations (depends on batch/seq/hidden),
                           NOT parameters (unlike DP/ZeRO)
```

### **Strong Scaling:**
```
Definition: Fixed model size, increase GPUs → reduce training time

Ideal: 2× GPUs → 2× speedup
ZeRO: Can achieve SUPER-LINEAR speedup!
  Why? More GPUs → Less memory per GPU → Larger batch per GPU → Better throughput
```

---

## **9. Common Pitfalls**

❌ **"ZeRO eliminates the need for MP"**
✅ **Correct**: ZeRO reduces model states memory, but MP still needed for activation memory in very large models

❌ **"Pos+g+p has 2× communication overhead"**
✅ **Correct**: 1.5× overhead (3Ψ vs 2Ψ)

❌ **"Tensor parallelism splits W_O by columns"**
✅ **Correct**: W_O is split by **rows** (input dimension), that's why we need all-reduce (not concatenation)

❌ **"3D parallelism = just use all three techniques"**
✅ **Correct**: 3D parallelism requires careful configuration based on model size, hardware topology, and batch size constraints

---

## **10. Real-World Examples**

| Model | Size | Configuration | Company |
|-------|------|---------------|---------|
| **GPT-2** | 1.5B | Pure DP | OpenAI (2019) |
| **Megatron-LM** | 8.3B | MP=8 (intra-node) | NVIDIA (2019) |
| **Turing-NLG** | 17B | ZeRO Pos+g + MP=16 | Microsoft (2020) |
| **GPT-3** | 175B | ZeRO + MP=8 + PP=16 | OpenAI (2020) |
| **Megatron-Turing NLG** | 530B | ZeRO + MP=8 + PP=64 | Microsoft+NVIDIA (2021) |

---

**Study Tips**:
- Draw the MLP/attention splitting patterns from memory (forward + backward)
- Practice deriving gradients for column-parallel and row-parallel layers
- Trace gradient flow: identify where all-reduce is needed and why
- Practice calculating memory savings for different Nd/Nm/Np
- Explain communication trade-offs (volume vs frequency)
- Understand **when** to add each parallelism dimension
