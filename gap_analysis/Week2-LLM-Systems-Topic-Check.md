# Week 2: LLM Systems Optimization - Topic Coverage Check

**Date**: Day 8 (Week 2, Day 1)
**Purpose**: Comprehensive topic inventory before beginning LLM Systems study
**Context**: Gap analysis (Day 6-7) tested 6 specific areas with 30% avg score. This check ensures we cover the full landscape, not just tested areas.

---

## 📋 **Instructions**

For each subtopic below, mark your current knowledge level:
- ✅ **Know**: Can explain confidently in an interview (2-3 min), ready to use in practice
- 🟡 **Unsure**: Heard of it, vague understanding, need review/refresh
- ❌ **Dunno**: No idea, never learned, or completely forgot

**After completing the check**:
1. Count scores by area (summary at end)
2. Identify priority gaps (❌ and 🟡 in high-impact areas)
3. Create focused 3-day study plan

---

## 🔍 **Area 1: Distributed Training Fundamentals**

**Cross-reference with gap analysis**: Q182 (Strong scaling - 0%)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Data parallelism - Basic concept, gradient synchronization | [ ] | [x] | [ ] | |
| Model parallelism - Splitting layers across devices | [ ] | [ ] | [x] | |
| Pipeline parallelism - Micro-batching, bubble overhead | [ ] | [ ] | [x] | |
| Tensor parallelism - Splitting individual layers (attention heads, FFN) | [ ] | [ ] | [x] | |
| 3D parallelism - Combining data/model/pipeline parallelism | [ ] | [ ] | [x] | |
| **Strong scaling** - Fixed problem size, increase devices | [ ] | [ ] | [x] | **Gap Q182** |
| **Weak scaling** - Fixed workload per device, increase devices | [ ] | [ ] | [x] | |
| **Scaling efficiency** - Communication overhead vs computation | [ ] | [x] | [ ] | |

**Area 1 Score**: ___/8 Know, _2_/8 Unsure, _6_/8 Dunno

---

## 🔍 **Area 2: Memory Optimization Strategies**

**Cross-reference with gap analysis**: Q184 (Memory sharding - 0%), Q183 (Memory bandwidth - 0%)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Activation checkpointing (gradient checkpointing) - Trade compute for memory | [ ] | [ ] | [x] | |
| ZeRO optimization - ZeRO Stage 1/2/3, optimizer state partitioning | [ ] | [ ] | [x] | |
| **Memory sharding** - Sharding calculations, FSDP parameters | [ ] | [ ] | [x] | **Gap Q184** |
| Mixed precision training - FP16, BF16, FP32 master weights | [ ] | [x] | [ ] | |
| CPU offloading - When to use, bandwidth considerations | [ ] | [ ] | [x] | |
| Flash Attention - Memory-efficient attention mechanism | [ ] | [ ] | [x] | |
| Paged Attention (vLLM) - Memory management during inference | [ ] | [ ] | [x] | |
| **Memory bandwidth bottlenecks** - Arithmetic intensity | [ ] | [ ] | [x] | **Gap Q183** |
| **LoRA mechanics** - Full activations, tiny optimizer states, memory benefits | [ ] | [ ] | [x] | **NEW: PEFT** |
| **QLoRA** - 4-bit quantization, paged optimizers, double quantization | [ ] | [ ] | [x] | **NEW: PEFT** |
| **PEFT trade-offs** - Memory vs accuracy, rank selection, when to use | [ ] | [ ] | [x] | **NEW: PEFT** |

**Area 2 Score**: ___/11 Know, _1_/11 Unsure, _10_/11 Dunno

---

## 🔍 **Area 3: Hardware & Performance Optimization**

**Cross-reference with gap analysis**: Q183 (Memory bandwidth - 0%), Q185 (Communication costs - 0%)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| GPU architecture basics - SM, CUDA cores, tensor cores | [ ] | [ ] | [x] | |
| TPU architecture - Systolic arrays, MXU | [ ] | [ ] | [x] | |
| Memory hierarchy - HBM, L2 cache, shared memory, registers | [ ] | [x] | [ ] | |
| Roofline model - Compute-bound vs memory-bound operations | [ ] | [x] | [ ] | |
| **Arithmetic intensity** - FLOPs per byte, optimizing for hardware | [ ] | [ ] | [x] | **Gap Q183** |
| Kernel fusion - Reducing memory access | [ ] | [ ] | [x] | |
| GPU utilization metrics - MFU (model FLOPs utilization), TFLOPs | [ ] | [ ] | [x] | |
| **Interconnect bandwidth** - NVLink, InfiniBand, communication costs | [ ] | [ ] | [x] | **Gap Q185** |

**Area 3 Score**: ___/8 Know, _2_/8 Unsure, _6_/8 Dunno

---

## 🔍 **Area 3.5: Data Loading & Preprocessing Bottlenecks** ⭐ NEW

**Why added**: Often the actual bottleneck in training, not GPU/model. Practical systems knowledge that differentiates candidates.

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| **CPU bottlenecks** - Tokenization, data augmentation as limiters | [ ] | [ ] | [x] | **NEW** |
| **DataLoader optimization** - num_workers, prefetch_factor, pin_memory | [ ] | [ ] | [x] | **NEW** |
| **Data formats** - webdataset, MosaicML streaming, Arrow/Parquet | [ ] | [ ] | [x] | **NEW** |
| **Pre-tokenization vs on-the-fly** - Trade-offs, storage vs flexibility | [ ] | [ ] | [x] | **NEW** |

**Area 3.5 Score**: ___/4 Know, ___/4 Unsure, _4_/4 Dunno

---

## 🔍 **Area 4: Parallelism Strategies & Trade-offs**

**Cross-reference with gap analysis**: Q185 (FSDP vs model parallelism communication - 0%)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| FSDP (Fully Sharded Data Parallel) - How it works, when to use | [ ] | [ ] | [x] | |
| Megatron-LM - Tensor + pipeline parallelism implementation | [ ] | [ ] | [x] | |
| DeepSpeed - ZeRO implementation, stages | [ ] | [ ] | [x] | |
| Communication patterns - All-reduce, reduce-scatter, all-gather | [ ] | [ ] | [x] | |
| **Communication costs** - FSDP vs model parallelism | [ ] | [ ] | [x] | **Gap Q185** |
| Hybrid strategies - When to combine different parallelism types | [ ] | [ ] | [x] | |
| Gradient accumulation - Effective batch size vs memory | [ ] | [ ] | [x] | |
| Asynchronous vs synchronous training - Trade-offs | [ ] | [ ] | [x] | |

**Area 4 Score**: ___/8 Know, ___/8 Unsure, _8_/8 Dunno

---

## 🔍 **Area 5: Inference Optimization**

**Cross-reference with gap analysis**: Q187 (4 methods for throughput - 0%: KV-cache, quantization, continuous batching, speculative decoding)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| **KV-cache** - How it works, memory requirements, O(n²)→O(n) | [ ] | [ ] | [x] | **Gap Q187** |
| **Quantization** - INT8, INT4, GPTQ, AWQ, effect on throughput | [ ] | [x] | [ ] | **Gap Q187** |
| **Continuous batching (Orca)** - Dynamic request joining | [ ] | [ ] | [x] | **Gap Q187** |
| **Speculative decoding** - Draft model + verification, 2-3× speedup | [ ] | [ ] | [x] | **Gap Q187** |
| Multi-query attention (MQA) - Reducing KV-cache size | [ ] | [ ] | [x] | |
| Grouped-query attention (GQA) - Balance between MHA and MQA | [ ] | [ ] | [x] | |
| Paged attention (vLLM) - Non-contiguous KV-cache | [ ] | [ ] | [x] | |
| Batching strategies - Static vs dynamic batching | [ ] | [ ] | [x] | |
| Request scheduling - Priority, fairness, throughput optimization | [ ] | [ ] | [x] | |
| Serving frameworks - vLLM, TensorRT-LLM, TGI (Text Generation Inference) | [ ] | [ ] | [x] | |

**Area 5 Score**: ___/10 Know, _1_/10 Unsure, _9_/10 Dunno

---

## 🔍 **Area 6: Transformer Architecture & Parameters**

**Cross-reference with gap analysis**: Q189 (QKV projections + INT8 KVs - 25%)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Transformer parameter counting - Attention, FFN, embeddings | [ ] | [x] | [ ] | |
| **QKV projections** - Parameter calculation, dimension relationships | [ ] | [x] | [ ] | **Gap Q189** |
| Multi-head attention parameters - Per-head dimensions | [ ] | [ ] | [x] | |
| Feed-forward network sizing - Typical 4× hidden size | [ ] | [x] | [ ] | |
| Layer normalization parameters - Affine parameters | [ ] | [ ] | [x] | |
| Position embeddings - Learned vs sinusoidal vs RoPE/ALiBi | [ ] | [x] | [ ] | |
| Vocabulary size impact - Embedding matrix size | [ ] | [x] | [ ] | |
| **INT8 KV-cache** - Quantization for inference | [ ] | [ ] | [x] | **Gap Q189** |
| Activation functions - GELU, SwiGLU parameter count impact | [ ] | [ ] | [x] | |

**Area 6 Score**: ___/9 Know, _5_/9 Unsure, _4_/9 Dunno

---

## 🔍 **Area 7: Performance Calculation & Analysis**

**Cross-reference with gap analysis**: Q183 (Memory bandwidth calculations - 0%)

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| FLOPs calculation - Forward pass, backward pass, total training FLOPs | [ ] | [ ] | [x] | |
| Memory calculation - Model weights, optimizer states, activations, gradients | [ ] | [x] | [ ] | |
| Throughput calculation - Tokens/sec, samples/sec | [ ] | [x] | [ ] | |
| Latency calculation - Time per token, time to first token (TTFT) | [ ] | [ ] | [x] | |
| Batch size optimization - Finding optimal batch size for hardware | [ ] | [ ] | [x] | |
| Scaling laws - Chinchilla laws, compute-optimal training | [ ] | [ ] | [x] | |
| Training time estimation - Given FLOPs, hardware, batch size | [ ] | [ ] | [x] | |
| Cost estimation - GPU-hours, cloud pricing | [ ] | [ ] | [x] | |

**Area 7 Score**: ___/8 Know, _2_/8 Unsure, _6_/8 Dunno

---

## 🔍 **Area 8: Advanced Topics (Good to Know)**

**Cross-reference with gap analysis**: Not directly tested, but important for comprehensive understanding

| Subtopic | Know | Unsure | Dunno | Notes |
|----------|------|--------|-------|-------|
| Mixture of Experts (MoE) - Routing, load balancing, sparse training | [ ] | [ ] | [x] | |
| Long context optimization - Sparse attention, FlashAttention-2, context compression | [ ] | [ ] | [x] | |
| Gradient checkpointing strategies - Selective checkpointing | [ ] | [ ] | [x] | |
| Communication compression - Gradient compression, PowerSGD | [ ] | [ ] | [x] | |
| Fault tolerance - Checkpointing strategies, elastic training | [ ] | [ ] | [x] | |
| Multi-node training - Challenges beyond single-node | [ ] | [ ] | [x] | |
| Custom kernels - Triton, CUDA for LLM operations | [ ] | [ ] | [x] | |
| Profiling tools - PyTorch Profiler, NSight, TensorBoard | [ ] | [x] | [ ] | |
| **RLHF systems** - 4 models (actor, critic, reference, reward) orchestration | [ ] | [x] | [ ] | **NEW** |
| **Multi-model orchestration** - Memory management for simultaneous models | [ ] | [ ] | [x] | **NEW** |

**Area 8 Score**: ___/10 Know, _2_/10 Unsure, _8_/10 Dunno

---

## 📊 **Summary Scorecard**

Fill in after completing the topic check:

| Area | Know | Unsure | Dunno | Priority |
|------|------|--------|-------|----------|
| 1. Distributed Training Fundamentals | ___/8 | ___/8 | ___/8 | |
| 2. Memory Optimization Strategies | ___/11 | ___/11 | ___/11 | |
| 3. Hardware & Performance Optimization | ___/8 | ___/8 | ___/8 | |
| 3.5 Data Loading & Preprocessing | ___/4 | ___/4 | ___/4 | |
| 4. Parallelism Strategies & Trade-offs | ___/8 | ___/8 | ___/8 | |
| 5. Inference Optimization | ___/10 | ___/10 | ___/10 | |
| 6. Transformer Architecture & Parameters | ___/9 | ___/9 | ___/9 | |
| 7. Performance Calculation & Analysis | ___/8 | ___/8 | ___/8 | |
| 8. Advanced Topics | ___/10 | ___/10 | ___/10 | |
| **TOTAL** | **___/76** | **___/76** | **___/76** | |

**Overall Readiness**: ____% (Know / 76 × 100)

---

## 🎯 **Priority Matrix for Study Planning**

After scoring, use this to prioritize:

### **Critical Priority** (Must study Day 1-2):
- ❌ Dunno in Areas 1-5 (foundational + high-impact)
- ❌ Dunno that appeared in gap analysis (Q182-189)

### **High Priority** (Study Day 2-3):
- 🟡 Unsure in Areas 1-5
- ❌ Dunno in Areas 6-7 (calculations are important)

### **Medium Priority** (Study if time permits):
- 🟡 Unsure in Areas 6-7
- ❌ Dunno in Area 8 (advanced topics, good to know)

### **Low Priority** (Skip for now):
- 🟡 Unsure in Area 8
- Already ✅ Know topics (light review only)

---

## 📅 **Recommended Timeline Adjustment**

**Original Plan**: 2 days (4 hours) for LLM Systems
**Realistic Assessment**: **3 days (6 hours)** for comprehensive coverage

### **Why 3 Days?**
- 76 subtopics is substantial (even just ❌ + 🟡 will be 30-50 topics)
- Gap analysis showed 5/6 areas at 0% → need foundational learning, not just review
- Practical exercises (calculations, vLLM exploration) take time

### **Proposed Split**:

**Day 1-2 (4 hours total)**:
- Focus: Critical gaps (strong scaling, memory optimization, parallelism)
- Resources: Megatron-LM paper, ZeRO paper, NVIDIA talks
- Hands-on: FLOPs/memory calculations

**Day 3 (2 hours)**:
- Focus: Inference optimization (KV-cache, quantization, batching, speculative decoding)
- Resources: vLLM docs, TensorRT-LLM, practical exploration
- Hands-on: Try vLLM locally or in Colab

**Total**: 3 days for LLM Systems → Week 2 schedule shifts accordingly

---

## 🔄 **Next Steps**

1. ✅ **Complete this topic check** (15-20 min) - mark Know/Unsure/Dunno for all 76 subtopics
2. ✅ **Fill in summary scorecard** - calculate overall readiness percentage
3. ✅ **Share results** - I'll create focused 3-day study plan based on your gaps
4. ✅ **Adjust Week 2 plan** - Shift other topics (statistics, RAG) accordingly

---

## 📝 **Gap Analysis Cross-Reference**

From Day 6-7, the following questions revealed LLM systems gaps:

- **Q182**: Strong scaling (0%) → Area 1
- **Q183**: Memory bandwidth bottleneck, arithmetic intensity (0%) → Areas 2, 3
- **Q184**: Memory sharding calculations (0%) → Area 2
- **Q185**: FSDP vs model parallelism communication (0%) → Area 4
- **Q187**: 4 methods for throughput (0%) → Area 5
- **Q189**: QKV projections, INT8 KVs (25%) → Area 6

**Key Insight**: These 6 questions are entry points, but each area has 8-11 subtopics. The 189 questions identified the gaps, but didn't cover all the concepts within each gap area.
