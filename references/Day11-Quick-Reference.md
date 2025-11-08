# Day 11 Quick Reference: Inference Optimization

**Week 2, Day 4** | Focus: KV-cache, Quantization, Continuous batching, Speculative decoding, MQA/GQA, PagedAttention

---

## **1. KV-Cache**

### **What It Is:**
Caches the attention Key (K) and Value (V) vectors for previously generated tokens instead of recomputing them from scratch at each decoding step.

### **Problem Solved:**
Reduces computation from O(n²) to O(n) per token:
- **Without KV-cache**: Each step recomputes K and V for all previous tokens (1+2+3+...+n = O(n²))
- **With KV-cache**: Each step only computes K and V for the new token (O(n) total)

### **Memory Formula:**
```
Per token: 2 × H × L × 2 bytes (FP16)
Per sequence: 2 × H × L × 2 × S bytes

Example (OPT-13B):
- H = 5120, L = 40, FP16 (2 bytes)
- Per token: 2 × 5120 × 40 × 2 = 819,200 bytes ≈ 800 KB
- Per sequence (2048 tokens): 800 KB × 2048 ≈ 1.6 GB
```

Where:
- **2** = Key + Value
- **H** = Hidden size
- **L** = Number of layers
- **S** = Sequence length

### **Trade-off:**
**Memory space vs. computation speed** - Saves massive compute but consumes significant GPU memory (30% in OPT-13B serving)

---

## **2. Quantization**

### **What It Is:**
Uses lower precision numbers (INT8, INT4) for model weights and/or activations instead of high precision (FP16/FP32).

**Two main approaches:**
- **GPTQ**: Per-layer quantization using Hessian matrix to find optimal adjustments that preserve overall accuracy
- **AWQ**: Keeps 1-2% salient weights (those contributing most to large activations) at full precision, aggressively quantizes all other weights

### **Problem Solved:**
- Reduces memory usage (2-4× smaller models)
- Improves computation speed (can load more values into registers at once)
- Enables running larger models on limited hardware

### **Trade-off:**
**Performance vs. prediction accuracy** - Lower precision = faster/smaller, but potential quality degradation (though GPTQ/AWQ minimize this)

---

## **3. Continuous Batching**

### **What It Is:**
Schedules requests at the **iteration level** rather than request level. After each iteration, completed requests are removed from the batch and new requests are added immediately.

**Contrast with static batching:**
- **Static**: Wait for entire batch to finish before starting new batch
- **Continuous**: Add new request as soon as any request finishes (per iteration)

### **Problem Solved:**
- Better GPU parallelism - no waiting for the longest sequence to complete
- Lower queueing delay - new requests only wait one iteration, not entire batch duration
- Higher throughput - GPU stays busy processing requests

### **Example:**
```
Static batching (batch=4):
Iteration 1: [A, B, C, D]
Iteration 2: [A, B, C, D]
...
Iteration 200: [D only] ← A,B,C finished, GPU underutilized
Must wait for D to finish before starting next batch

Continuous batching:
Iteration 1: [A, B, C, D]
Iteration 50: [B, C, D, E] ← A finished, E added immediately
Iteration 100: [C, D, E, F] ← B finished, F added
```

### **Trade-off:**
**Higher throughput vs. complexity** - Requires dynamic scheduling logic and special GPU kernels (more complex than static batching)

---

## **4. Speculative Decoding**

### **What It Is:**
Uses a faster **draft model** (small, 1-2B params) to generate multiple candidate tokens (4-5 tokens), then the **large model** verifies all candidates in a single forward pass using parallel attention.

**Verification process:**
1. Draft model generates: `["blue", "and", "the", "grass"]`
2. Large model takes prompt + 4 draft tokens as input
3. Computes probabilities for all 4 positions in ONE forward pass (parallel)
4. Accept/reject each token independently, resample if rejected

### **Problem Solved:**
**Decoding speedup** - Achieves 2-3× speedup by amortizing large model cost over multiple tokens

**Why it works:**
- Draft model: 1B params (2 GB) → 175× faster memory transfer than 175B model
- Parallel verification: Transformer's causal attention allows verifying 4 tokens in one pass

### **Trade-off:**
**Speedup vs. complexity and batch size:**
- ✅ **Best for small batches (1-4)**: Low GPU utilization → draft model fills idle time
- ❌ **Less effective for large batches (64)**: Higher variance in acceptance rates + ragged tensor overhead
- ❌ Need to load extra draft model (memory overhead)
- ❌ Can slow down if draft disagrees often (wasted work)

### **The Ragged Tensor Problem (Batch Processing Challenge):**
When sequences accept different numbers of draft tokens, they end up at different positions:
```
After verification (draft generated 5 tokens for each):
Seq A: Accepts all 5 → now at position 5
Seq B: Rejects at token 2, accepts 1 → now at position 1
Seq C: Accepts 4 tokens → now at position 4
```

**Problem**: GPUs require rectangular tensors, but sequences are now misaligned (position IDs, attention masks, KV-cache)

**Three solutions from research:**
1. **Masking** (BSP approach): Handle ragged tensors directly → ❌ Corrupted outputs (non-contiguous position IDs)
2. **Rollback**: Truncate all sequences to minimum accepted (all → position 1) → ❌ Wasteful, throughput collapses
3. **Dynamic Padding**: Realign via left padding to maintain right alignment → ✓ Viable, but 40% overhead for synchronization

**Why small batches benefit MORE:**
1. Low GPU utilization (draft model fills idle time)
2. Lower variance in acceptance rates (less raggedness to manage)
3. Less overhead synchronizing ragged tensors across many sequences

---

## **5. Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)**

### **What It Is:**
Reduces KV-cache memory by **sharing** Key and Value heads across multiple Query heads:

- **MHA (Multi-Head Attention)**: 32 Q heads, 32 K heads, 32 V heads (standard)
- **MQA (Multi-Query Attention)**: 32 Q heads, **1 K head, 1 V head** (all queries share same K/V)
- **GQA (Grouped-Query Attention)**: 32 Q heads, **4 K heads, 4 V heads** (queries grouped, share K/V within group)

### **Problem Solved:**
Multi-head attention (MHA) leads to **large KV-cache** (32 K/V heads × all tokens). MQA/GQA dramatically reduce memory:
- MQA: 32× reduction in KV-cache
- GQA (8 groups): 8× reduction in KV-cache

### **Trade-off:**
**Memory/speed gains vs. potential accuracy loss:**
- MQA: Maximum memory savings, but can hurt quality
- GQA: **Best compromise** - significant savings (8×) with minimal accuracy loss
- MHA: Highest quality, but largest memory footprint

---

## **6. PagedAttention (vLLM)**

### **What It Is:**
Manages KV-cache in **fixed-size blocks** (like OS virtual memory paging). Physical blocks can be:
- **Non-contiguous**: Block 0, Block 7, Block 3 (any order)
- **Shared**: Multiple sequences can reference same physical block (copy-on-write)

**Key innovation:**
- Logical blocks (per sequence) map to physical blocks via block table
- Allocate blocks **on-demand** as tokens are generated (no pre-allocation)

### **Problem Solved:**
**GPU memory fragmentation** in existing systems:
- Traditional: Pre-allocate max length (2048 slots) per request → **80% waste**
- vLLM: Allocate 16-token blocks on-demand → **96.3% utilization** (near-zero waste)

### **Memory Savings Example:**
```
4 requests (max 2048 tokens each):
Traditional (FasterTransformer): 4 × 2048 = 8192 slots
- Actual usage: A=2000, B=100, C=500, D=1500
- Wasted: 48 + 1948 + 1548 + 548 = 4092 slots (50% waste!)

vLLM (block size=16):
- A: 125 blocks, B: 7 blocks, C: 32 blocks, D: 94 blocks = 258 blocks
- Wasted: Only last block of each request (at most 15 tokens × 4 = 60 slots)
- Waste: 28 slots (0.3%!)
```

### **Trade-off:**
**Improved memory utilization vs. potential computation overhead:**
- ✅ 2-4× higher throughput (more requests fit in memory)
- ✅ Enables memory sharing (parallel sampling, beam search)
- ❌ 20-26% higher attention kernel latency (block table lookup, non-contiguous memory access)
- ❌ Copy-on-write overhead for shared blocks

---

## **Comparison Table**

| Technique | Primary Benefit | Memory Impact | Compute Impact | Best Use Case |
|-----------|----------------|---------------|----------------|---------------|
| **KV-Cache** | O(n²)→O(n) speedup | +30% memory | -90% compute | All autoregressive generation |
| **Quantization** | 2-4× smaller | -50-75% memory | +10-30% speed | Limited GPU memory |
| **Continuous Batching** | Higher throughput | Neutral | Better utilization | High-traffic serving |
| **Speculative Decoding** | 2-3× speedup | +20% (draft model) | Variable | Low-latency, small batch |
| **MQA/GQA** | Lower KV-cache | -8-32× cache | Slightly faster | Memory-constrained serving |
| **PagedAttention** | Near-zero waste | 4-5× more requests | -20% kernel speed | High-concurrency serving |

---

## **Interview Q&A**

### **Q: Explain KV-cache and its memory cost.**
**A**: "KV-cache stores the attention Key and Value vectors for previously generated tokens, avoiding O(n²) recomputation. Memory cost is 2×H×L×2 bytes per token, where H is hidden size and L is layers. For OPT-13B, that's 800 KB per token, so a 2048-token sequence requires 1.6 GB just for KV-cache - which is why efficient memory management like PagedAttention is critical."

### **Q: What's the difference between GPTQ and AWQ quantization?**
**A**: "GPTQ uses Hessian-based per-layer quantization, adjusting weights sequentially to compensate for previous quantization errors and preserve overall accuracy. AWQ takes an activation-aware approach - it identifies the 1-2% of weights that contribute most to large activations and keeps those at full precision, then aggressively quantizes the rest. AWQ is often faster to apply and preserves quality well for LLMs."

### **Q: Why is speculative decoding more effective for small batches?**
**A**: "Small batches have low GPU utilization - matrix-vector operations don't saturate the GPU, so the fast draft model (1-2B params with 100× less memory transfer) fills idle time. The large model then verifies multiple tokens in parallel. For large batches, three issues arise: (1) GPU already busy with matrix-matrix operations, (2) higher variance in acceptance rates creates ragged tensors where sequences end up at different positions, and (3) synchronizing these ragged tensors (position IDs, attention masks, KV-cache) can consume 40% of computation. Research shows three approaches - masking (corrupts outputs), rollback (wastes verified tokens), or dynamic padding (40% overhead) - making large batch speculative decoding significantly less efficient."

### **Q: How does PagedAttention achieve near-zero memory waste?**
**A**: "PagedAttention divides KV-cache into fixed-size blocks (typically 16 tokens) and allocates them on-demand, like OS virtual memory paging. Unlike traditional systems that pre-allocate max sequence length (e.g., 2048 slots), vLLM only allocates blocks as tokens are generated. This eliminates internal fragmentation (no unused pre-allocated space) and external fragmentation (all blocks are same size). Memory waste is limited to the last unfilled block per sequence - for a 16-token block size, that's at most 15 tokens, compared to 1000+ tokens wasted in traditional systems."

### **Q: When would you use MQA vs GQA?**
**A**: "MQA (one shared K/V head) gives maximum memory savings but can hurt model quality. GQA groups queries (e.g., 8 groups for 32 heads) so you share K/V within each group - this provides 8× memory reduction while maintaining quality close to full MHA. I'd use GQA as the default for new models (best compromise), MQA only if memory is extremely constrained, and MHA for maximum quality if memory allows."

---

## **Common Pitfalls**

❌ **"KV-cache reduces memory usage"**
✅ KV-cache reduces **compute** but **increases memory** (trades memory for speed)

❌ **"Speculative decoding always improves latency"**
✅ Only helps for small batches; large batches already have good GPU utilization

❌ **"PagedAttention is slower because of non-contiguous memory"**
✅ 20% kernel overhead, but 2-4× overall throughput due to fitting more requests

❌ **"Continuous batching = larger batch size"**
✅ Continuous batching = **dynamic** scheduling at iteration level (batch size can vary)

❌ **"MQA means one query head"**
✅ MQA means one **K/V head** shared across all query heads

---

## **Key Numbers to Remember**

### **KV-Cache Memory:**
- OPT-13B: **800 KB per token**
- GPT-3 (175B): **4.5 MB per token**

### **Memory Savings:**
- PagedAttention: **96.3%** utilization vs 20.4% (FasterTransformer)
- MQA: **32× reduction** in KV-cache (32 heads → 1)
- GQA: **8× reduction** (32 heads → 4)
- Quantization: **2-4× reduction** (FP16 → INT8/INT4)

### **Speedup:**
- KV-cache: **O(n²) → O(n)** per token
- Speculative decoding: **2-3× speedup** (small batch)
- vLLM throughput: **2-4× improvement** vs Orca/FasterTransformer

---

## **Study Tips**

- **Draw it out**: Sketch MHA vs MQA vs GQA head patterns
- **Calculate examples**: Practice KV-cache memory for different models
- **Connect techniques**: PagedAttention + Continuous batching work together in vLLM
- **Trade-off thinking**: Every optimization has a cost - memory, compute, or complexity
- **Production context**: Know which techniques are research vs production-ready (continuous batching = Orca paper, PagedAttention = vLLM paper)
