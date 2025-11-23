# Day 25: Communication Patterns & Dynamic Batching

**Date**: Day 25 (Week 4, Day 4)
**Topics**: Communication volume formulas, Dynamic batching, QPS/latency calculations

---

## 1. Communication Volume Formulas (TP/DP/PP)

### Ring All-Reduce Implementation

**Key insight**: When papers say "2P" for DP, they mean **physical network traffic** (ring all-reduce implementation), not logical tensor size.

**Ring all-reduce has 2 phases:**
1. Reduce-scatter: Each node gets 1/N of final result → P data moved
2. All-gather: Broadcast results to all nodes → P data moved
3. **Total**: 2P physical network traffic

### Communication Volume by Parallelism Type

| Parallelism | Logical Volume | Physical Network Traffic | Per What |
|-------------|----------------|-------------------------|----------|
| **DP/FSDP** | P (gradients) | **2P** | Per training step |
| **TP** | B×S×H per all-reduce | **2×B×S×H** per all-reduce | Per all-reduce |
| **PP** | B×S×H | **B×S×H** | Per stage boundary (p2p, no ring) |

### Tensor Parallelism (TP) - Detailed

**Per layer (Megatron-LM style):**
- Forward pass: 2 all-reduces
- Backward pass: 2 all-reduces
- **Total**: 4 all-reduces per layer

**Communication volume:**
- **Logical**: 4×B×S×H per layer
- **Physical** (with ring all-reduce): **8×B×S×H per layer**
- **Full model**: 8×B×S×H×L (where L = number of layers)

**Example calculation:**
```
B=16, S=2048, H=4096, L=32 layers
Logical: 4 × 16 × 2048 × 4096 × 32 = 17.2 TB per training step
Physical: 8 × 16 × 2048 × 4096 × 32 = 34.4 TB per training step
```

### Data Parallelism (DP/FSDP)

**Communication:**
- All-reduce gradients once per step
- **Logical**: P (parameter size)
- **Physical**: **2P** (ring all-reduce overhead)

**Example:**
```
P = 7B parameters × 4 bytes = 28 GB
Physical traffic: 2 × 28 GB = 56 GB per training step
```

### Pipeline Parallelism (PP)

**Communication:**
- Point-to-point (no ring all-reduce)
- Send activations forward, gradients backward
- **Volume**: B×S×H per stage boundary
- **No 2× multiplier** (direct send, not collective)

**Example:**
```
B=16, S=2048, H=4096, 4 pipeline stages
Volume per boundary: 16 × 2048 × 4096 = 134 MB
Total (3 boundaries): 3 × 134 MB = 402 MB per microbatch
```

---

## 2. Dynamic Batching: QPS vs Latency

### Key Distinction: Latency ≠ Throughput

**Latency** (user experience):
- How long ONE request takes
- Includes waiting time + inference time
- Varies per request (some wait 0ms, some wait 50ms)

**Throughput (QPS)** (system capacity):
- How many requests per second
- Based on GPU processing time ONLY
- Does NOT include batch collection time

### QPS Calculation Formula

```
QPS per GPU = Batch Size / Inference Time
```

**Example:**
- Batch size: 50 requests
- Inference time: 100ms per batch
- QPS per GPU: 50 / 0.1s = **500 QPS**

**Note**: At steady state (continuous load), GPU never idles - always has requests in queue.

### Latency Calculation

**Average latency** = (Batch Collection Time / 2) + Inference Time

**Why divide by 2?** On average, a request arrives halfway through batch collection.

**Example:**
```
Per-GPU arrival rate: 500 QPS
Batch size: 50
Collection time: 50 / 500 = 100ms
Average wait: 100ms / 2 = 50ms
Inference: 100ms
Average latency: 50ms + 100ms = 150ms
```

### Batch Collection Time Calculation

**Use PER-GPU QPS, not total QPS!**

```
Per-GPU QPS = Total QPS / Number of GPUs
Collection Time = Batch Size / Per-GPU QPS
```

**Example:**
```
Total load: 30,000 QPS
GPUs: 60
Per-GPU QPS: 30,000 / 60 = 500 QPS
Batch size: 50
Collection time: 50 / 500 = 100ms
```

**Why per-GPU?** Each GPU has its own queue after load balancing.

### Cold Start vs Steady State

**Cold start (empty queue):**
- First request waits full collection time (100ms)
- Total latency: 100ms (wait) + 100ms (inference) = 200ms

**Steady state (continuous load):**
- Queue always has waiting requests
- GPU finishes batch → immediately grabs next batch
- Collection time ≈ 0ms (queue is full)
- Average latency: ~50ms (avg wait) + 100ms (inference) = 150ms

---

## 3. Multi-Stage Ranking with Batching

### The Core Challenge

**Problem**: Larger batch size → Higher throughput → Lower cost
**But**: Collecting batch takes time → Adds latency

### Why Later Stages Can Use Bigger Batches

**Key insight**: Later stages process FEWER candidates per request → Lighter per-request workload

| Stage | Candidates/Request | Batch Size | Why |
|-------|-------------------|------------|-----|
| 1 | 1M → 1000 | 1 | ANN search, no batching |
| 2 | 200 | 10-20 | Keep latency low |
| 3 | 50 | 50-100 | Fewer items → can batch more requests |

### Stage 3 Example (Detailed)

**Setup:**
- Input: 20 candidates per request (not 200!)
- Model: Large transformer
- Total load: 30,000 QPS

**Option A: Small batch, fast inference**
```
Batch size: 20 requests
Per-request work: 20 × 20 = 400 items
Inference: 40ms
GPUs needed: 30,000 / (20/0.04) = 60 GPUs
Per-GPU QPS: 500
Collection time: 20 / 500 = 40ms
Average latency: 20ms + 40ms = 60ms ❌ (exceeds 50ms budget)
```

**Option B: Optimized model**
```
Batch size: 20 requests
Inference: 30ms (TensorRT optimization)
GPUs needed: 30,000 / (20/0.03) = 45 GPUs
Per-GPU QPS: 667
Collection time: 20 / 667 = 30ms
Average latency: 15ms + 30ms = 45ms ✅
```

**Option C (Most common): Skip Stage 3**
```
Make Stage 2 slightly more complex
Inference: 80ms per request, batch size 50
QPS per GPU: 50 / 0.08 = 625 QPS
GPUs needed: 48
Per-GPU QPS: 625
Collection time: 50 / 625 = 80ms
Average latency: 40ms + 80ms = 120ms ✅
```

### Progressive Ranking Architecture

```
Stage 1: Candidate Generation (1000 candidates)
  - Two-tower, ANN
  - 20ms, no batching

Stage 2: Lightweight Ranking (Top 200)
  - Simple MLP
  - 50ms per request, batch size 50
  - Per-GPU: 50 / 0.05 = 1000 QPS
  - GPUs: 30 (30,000 / 1000)
  - Collection: 50ms
  - Latency: 25ms + 50ms = 75ms

Stage 3: Heavy Ranking (Top 50)
  - Complex model
  - 40ms inference, batch size 20
  - Per-GPU: 20 / 0.04 = 500 QPS
  - GPUs: 60
  - Collection: 40ms
  - Latency: 20ms + 40ms = 60ms

Total: 20 + 75 + 60 = 155ms ✅
```

---

## 4. Common Mistakes to Avoid

### ❌ Mistake 1: Including wait time in QPS
```
Wrong: "100ms wait + 100ms inference = 200ms → 5 QPS"
Right: QPS = 50 / 0.1s = 500 QPS (based on inference time only)
```

### ❌ Mistake 2: Using total QPS for collection time
```
Wrong: "30k QPS, 50 batch → 50/30000 = 1.7ms collection"
Right: Per-GPU = 30k/60 = 500 QPS → 50/500 = 100ms collection
```

### ❌ Mistake 3: Forgetting ring all-reduce overhead
```
Wrong: "TP communicates B×S×H per all-reduce"
Right: "TP communicates 2×B×S×H per all-reduce (ring all-reduce)"
Or: "TP logical volume: B×S×H, physical: 2×B×S×H"
```

### ❌ Mistake 4: Thinking bigger batch always helps latency
```
Wrong: "Batch 100 requests → lower latency"
Right: Larger batch → higher throughput BUT longer collection wait
```

---

## 5. Quick Reference Formulas

### Communication (Physical Network Traffic)
```
DP/FSDP:  2P per step
TP:       8×B×S×H per layer (4 all-reduces × 2× ring overhead)
PP:       B×S×H per stage boundary (no ring overhead)
```

### QPS and Latency
```
QPS per GPU = Batch Size / Inference Time
GPUs needed = Total QPS / QPS per GPU
Per-GPU QPS = Total QPS / Number of GPUs
Collection Time = Batch Size / Per-GPU QPS
Average Latency = (Collection Time / 2) + Inference Time
```

### Multi-Stage Serving
```
Stage 1: Few candidates → Small batch or no batch
Stage 2: Medium candidates → Medium batch (10-50)
Stage 3: Many candidates → Larger batch (50-100)
```

---

## 6. Interview Tips

### When discussing communication:
✅ "TP has 4 all-reduces per layer: 2 forward, 2 backward"
✅ "Physical traffic is 2× logical due to ring all-reduce: 8×B×S×H per layer"
✅ "DP communicates 2P per step (all-reduce gradients)"

### When discussing serving:
✅ "QPS is based on GPU processing time, not including batch wait"
✅ "Batch collection time uses per-GPU arrival rate"
✅ "Later stages can use bigger batches because per-request work is smaller"
✅ "Trade-off: Larger batch → higher throughput, longer average latency"

### Red flags to avoid:
❌ "TP communicates B×S×H" (missing 2× overhead)
❌ "Collection time is negligible at 30k QPS" (need per-GPU rate!)
❌ "QPS = 1 / total_latency" (confuses throughput with latency)

---

## 7. Related Topics

- **Day 8-9**: Megatron-LM, ZeRO (foundational concepts)
- **Day 10**: GPU roofline, 3D parallelism (where these formulas apply)
- **Day 11**: Inference optimization (continuous batching, KV-cache)
- **Day 20**: YouTube recommendations (system design with batching)
- **Day 24**: System design serving (latency budgets, multi-stage)

---

## 8. Practice Problems

### Problem 1: Communication Volume
```
Model: 13B parameters, D=5120, L=40 layers
Training: B=32, S=4096, TP=8, DP=64
Calculate: (a) DP communication per step, (b) TP communication per step
```

**Answer:**
```
(a) DP: 2P = 2 × 13B × 4 bytes = 104 GB per step
(b) TP per layer: 8 × 32 × 4096 × 5120 = 5.37 GB
    TP total: 5.37 GB × 40 = 214.7 GB per step
```

### Problem 2: Serving with Batching
```
Total QPS: 50,000
Stage 2: Batch size 50, inference 80ms
Calculate: (a) GPUs needed, (b) Average latency
```

**Answer:**
```
(a) QPS per GPU = 50 / 0.08 = 625 QPS
    GPUs = 50,000 / 625 = 80 GPUs

(b) Per-GPU QPS = 50,000 / 80 = 625 QPS
    Collection = 50 / 625 = 80ms
    Average latency = 40ms + 80ms = 120ms
```

### Problem 3: Latency Budget
```
Total budget: 200ms
Stage 1: 20ms
Stage 2: 80ms (batch 50, inference 80ms)
Stage 3: Budget remaining = ?
Can you use batch size 30 with 50ms inference?
```

**Answer:**
```
Remaining: 200 - 20 - 80 = 100ms

Stage 3 with batch 30, inference 50ms:
QPS per GPU = 30 / 0.05 = 600 QPS
Assume 50k QPS total → 84 GPUs
Per-GPU QPS = 595 QPS
Collection = 30 / 595 = 50ms
Latency = 25ms + 50ms = 75ms ✅ (within 100ms budget)
```

---

**Created**: 2025-11-22 (Day 25, Week 4 Day 4)
**Topics Mastered**: Communication formulas clarified, Dynamic batching understood
**Gap Closed**: llm_comm_patterns (75% → review with this sheet → target 95%+)
