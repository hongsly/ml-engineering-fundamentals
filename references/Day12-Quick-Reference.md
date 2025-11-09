# Day 12 Quick Reference: Transformer Calculations & Memory

**Week 2, Day 5** | Focus: Parameter counting, memory calculations, batch size optimization

---

## **1. Transformer Parameter Counting**

### **Attention Layer (per layer):**
```
Attention params = 4H²

Breakdown:
- W_Q: H × H
- W_K: H × H
- W_V: H × H
- W_O: H × H (output projection)
Total: 4H²

NOTE: n_heads SPLITS H, doesn't multiply!
```

### **Feed-Forward Network (per layer):**
```
FFN params = 2 × H × d_ff
             = 2 × H × (4H)  (typically d_ff = 4H)
             = 8H²
```

### **Embeddings:**
```
Token embeddings: V × H
Position embeddings: max_seq_len × H (if learned)
```

### **Total Model Parameters:**
```
Total = V × H + L × (4H² + 8H²)
      = V × H + 12H²L
```

**Example - GPT-3:**
- V = 50,257, H = 12,288, L = 96
- Total = 50,257 × 12,288 + 12 × (12,288)² × 96
- Total ≈ 175B parameters

---

## **2. Memory Calculations**

### **Model Memory (Inference):**
```
Memory_model = P × bytes_per_param

FP16: 2 bytes per param
FP32: 4 bytes per param
INT8: 1 byte per param

Example (7B model, FP16):
Memory = 7B × 2 bytes = 14 GB
```

### **Training Memory (per GPU):**
```
Total = Model + Optimizer + Gradients + Activations

Model:     P × 2 bytes (FP16)
Optimizer: P × 8 bytes (Adam, FP32 momentum + variance)
          +P × 4 bytes (FP32 master copy if mixed precision)
          = P × 12 bytes total for Adam + mixed precision
Gradients: P × 2 bytes (FP16)
---------
Model states: P × 16 bytes

Activations: Variable (depends on s, b, h, L)
```

**Example - 7B model:**
```
Model states: 7B × 16 bytes = 112 GB
Cannot fit on A100 (80GB) → Need parallelism!
```

---

## **3. Activation Memory**

### **Formula (Full Gradient Checkpointing):**
```
memory_activation = 2 × s × b × h × L bytes

s: sequence length
b: batch size
h: hidden dimension
L: number of layers
```

### **Without Checkpointing:**
```
memory_activation ≈ sbhL × (10 + 24t + 5as/ht)

For intuition: ~10-20× higher than with checkpointing
```

### **Per Sample:**
```
memory_per_sample = 2 × s × h × L bytes

Example (Llama-2 7B: H=4096, L=32, s=4096):
= 2 × 4096 × 4096 × 32
= 1,073,741,824 bytes
≈ 1 GB per sample
```

---

## **4. ZeRO Memory Sharding**

### **ZeRO Stages:**
```
Stage 1: Shard optimizer states only
         Memory per GPU = (2P + 12P + 2P) / N = 16P / N
         Reduction: ~4× (for optimizer states)

Stage 2: Shard optimizer + gradients
         Memory per GPU = (2P + 12P/N + 2P/N) = (2P + 14P/N)
         Reduction: ~8× (cumulative)

Stage 3: Shard optimizer + gradients + model
         Memory per GPU = (2P/N + 12P/N + 2P/N) = 16P / N
         Reduction: N× (everything sharded)
```

### **Practical Example (7B model, ZeRO-3, N=4 GPUs):**
```
Model states per GPU = 16 × 7B / 4 = 28 GB
Available for activations = 80 - 28 = 52 GB

With 1 GB per sample:
Max batch size per GPU = 52 samples
Global batch size = 52 × 4 = 208 samples
```

---

## **5. Batch Size Optimization**

### **Process:**
1. Calculate model states memory
2. Apply parallelism (ZeRO-3 for memory efficiency)
3. Calculate available memory for activations
4. Calculate activation memory per sample
5. Determine max batch size

### **Example - 10B model on 8× A100 (80GB):**
```
Step 1: Model states = 16 × 10B = 160 GB (no parallelism)

Step 2: ZeRO-3 with N=8
        Per GPU = 160 GB / 8 = 20 GB

Step 3: Available = 80 - 20 = 60 GB per GPU

Step 4: Per sample (H=5120, L=40, s=2048):
        = 2 × 2048 × 5120 × 40
        = 838,860,800 bytes ≈ 0.84 GB

Step 5: Max batch = 60 / 0.84 ≈ 71 samples per GPU
        Global = 71 × 8 = 568 samples
```

---

## **6. Chinchilla Scaling Law**

### **Formula:**
```
D = 20P

D: Training data size (in tokens)
P: Number of parameters
```

### **What It Optimizes:**
**Minimizes training cost (GPU-hours) to reach target performance**

### **What It DOESN'T Optimize:**
- Total cost of ownership (training + serving)
- Inference cost (larger models cost more to serve)
- Time to market
- Model capabilities (few-shot learning, etc.)

### **Why Companies Violate:**
1. Inference cost dominates for popular models
2. Time to market (1000 GPUs × 1 hr >> 1 GPU × 1000 hrs in wall-clock time)
3. Model size constraints (edge deployment, latency)
4. Amortization (if serving billions of queries, training cost negligible)

---

## **7. Key Formulas**

### **Parameter Counting:**
```
Attention:  4H²
FFN:        8H² (assuming d_ff = 4H)
Total:      V×H + 12H²L
```

### **Memory (Training):**
```
Model states:  16P bytes (Adam + mixed precision)
Activations:   2sbhL bytes (full checkpointing)
```

### **ZeRO-3 Per GPU:**
```
Model states: 16P / N bytes
```

### **Batch Size:**
```
Max batch per GPU = (GPU_memory - model_states) / activation_per_sample
```

---

## **Interview Q&A**

### **Q: How many parameters in a transformer attention layer?**
**A**: "4H² parameters for the attention layer - H² each for Query, Key, Value projections, plus H² for the output projection. Common mistake is thinking it's 4H²×n_heads, but the heads split H, they don't multiply it. For example, if H=768 and n_heads=12, each head processes dimension 64, not the full 768."

### **Q: Can you train a 7B parameter model on a single A100 with 80GB?**
**A**: "No. Model states alone require 112 GB: 14 GB for the model (FP16), 84 GB for Adam optimizer (FP32 momentum, variance, and master copy), and 14 GB for gradients. You'd need at least 2 GPUs with ZeRO Stage 3 to fit the model states, which would take 56 GB per GPU. With 4 GPUs, you'd have 28 GB per GPU for model states, leaving 52 GB for activations, allowing a batch size of about 50-60 samples per GPU depending on sequence length."

### **Q: What's the Chinchilla scaling law and why don't companies follow it?**
**A**: "Chinchilla says D=20P - to minimize training cost for a target performance, train on 20× more tokens than parameters. For example, a 70B model should train on 1.4T tokens. Companies violate this because it only optimizes training cost, not total cost. For popular models, inference cost dominates - if you serve billions of queries, the training cost is amortized. Also, time-to-market matters: 1000 GPUs for 1 hour is much faster than 1 GPU for 1000 hours, even though the cost is the same. Finally, larger models may have better few-shot learning or other capabilities that justify the extra training cost."

### **Q: How do you calculate memory for activations?**
**A**: "With full gradient checkpointing, it's 2sbhL bytes, where s is sequence length, b is batch size, h is hidden dimension, and L is layers. For example, Llama-2 7B with s=4096, b=1, h=4096, L=32 gives 2×4096×4096×32 = 1 GB per sample. Without checkpointing, it's about 10-20× higher because you store all intermediate activations. The '2' factor is for FP16 precision (2 bytes per value)."

### **Q: How does ZeRO Stage 3 enable training large models?**
**A**: "ZeRO-3 shards the model parameters, gradients, and optimizer states across all GPUs. Memory per GPU becomes 16P/N instead of 16P, where N is the number of GPUs. For example, a 7B model requiring 112 GB can fit on 4× A100s (28 GB per GPU for model states). ZeRO-3 uses all-gather to reconstruct parameters layer-by-layer during forward/backward passes, then reduce-scatters gradients. The communication cost is 3P per step (1P forward + 2P backward), but the memory savings enable training models that wouldn't otherwise fit."

---

## **Common Pitfalls**

❌ **"Attention has 4H²×n_heads parameters"**
✅ n_heads splits H, so it's just 4H² total (each head processes H/n_heads dimensions)

❌ **"Optimizer memory is 2P (momentum + variance)"**
✅ For Adam with mixed precision: 12P (4P momentum + 4P variance + 4P master copy, all FP32)

❌ **"Chinchilla law says use 20× more data"**
✅ More precise: D=20P (20× as many tokens as parameters, not 20× more than some baseline)

❌ **"Pipeline parallelism doesn't reduce activation memory"**
✅ It DOES reduce by N× per GPU (blog posts saying otherwise are imprecise)

❌ **"Gradients are transient during backward pass"**
✅ They ARE computed layer-by-layer but STORED for all parameters until optimizer.step() (needed for gradient clipping)

---

## **Key Numbers to Remember**

### **Parameter Counts:**
- GPT-2: 124M (H=768, L=12)
- GPT-3: 175B (H=12288, L=96)
- Llama-2 7B: 7B (H=4096, L=32)

### **Memory Multipliers:**
- Model (FP16): 2P bytes
- Adam optimizer (mixed precision): 12P bytes
- Gradients (FP16): 2P bytes
- **Total model states: 16P bytes**

### **Activation Memory:**
- With full checkpointing: 2sbhL bytes
- Without checkpointing: ~10-20× higher

### **ZeRO Reductions:**
- Stage 1: 4× (optimizer only)
- Stage 2: 8× (optimizer + gradients)
- Stage 3: N× (everything sharded)

---

## **Study Tips**

- **Practice calculations**: Don't just memorize formulas - do examples (GPT-2, GPT-3, 7B models)
- **Show your work**: In interviews, write out the formula then plug in numbers
- **State assumptions**: "Assuming Llama-2 specs with H=4096, L=32, s=4096..."
- **Check units**: Convert bytes → GB (divide by 10⁹ or 2³⁰)
- **Explain trade-offs**: Every optimization has a cost - memory, compute, or complexity
