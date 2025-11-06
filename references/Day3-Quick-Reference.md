# ML Quick Reference - Day 3 Theory Refresh

**Date**: 2025-10-30
**Topics**: Optimizers, Evaluation Metrics (ROC-AUC), Attention/Transformers

---

## 1. Optimizers

### **Momentum**
```python
v_dw = beta * v_dw + (1 - beta) * dw
w = w - alpha * v_dw
```

**Key Points**:
- Adds "velocity" to gradient descent
- Smooths out oscillations, faster convergence
- **Default**: β = 0.9
- **When to use**: Works better than plain SGD, especially with noisy gradients

---

### **RMSprop**
```python
s_dw = beta2 * s_dw + (1 - beta2) * (dw ** 2)
w = w - alpha * dw / (sqrt(s_dw) + epsilon)
```

**Key Points**:
- **Adaptive learning rate per parameter**
- Divides by root mean square of gradients
- **Numerical stability**: Add ε (epsilon) to denominator to prevent division by zero
- **When to use**: Good for non-stationary objectives, different scales across parameters

---

### **Adam** (Adaptive Moment Estimation)
```python
# Iteration t:
v_dw = beta1 * v_dw + (1 - beta1) * dw          # Momentum
s_dw = beta2 * s_dw + (1 - beta2) * (dw ** 2)   # RMSprop

# Bias correction (important for early iterations)
v_dw_corrected = v_dw / (1 - beta1 ** t)
s_dw_corrected = s_dw / (1 - beta2 ** t)

# Update
w = w - alpha * v_dw_corrected / (sqrt(s_dw_corrected) + epsilon)
```

**Key Points**:
- **Combines**: Momentum + RMSprop
- **Default hyperparameters**:
  - β₁ = 0.9 (momentum)
  - β₂ = 0.999 (RMSprop)
  - α: learning rate
  - ε = 1e-8 (numerical stability)
- **Bias correction**: Compensates for initialization at 0
  - Important in first ~10 iterations
  - Prevents slow warm-up
  - Correction factor: (1 - β^t) → approaches 1 as t increases
- **When to use**: **Default choice** for most ML problems

**Bias Correction Intuition**:
- Problem: v_dw starts at 0 → early iterations biased toward 0
- Solution: Divide by (1 - β^t) to compensate
- Iteration 1: Divide by 0.1 → 10x correction
- Iteration 10+: Divide by ~0.65 → minimal correction
- Result: No slow start, full speed from iteration 1

---

## 2. Evaluation Metrics

### **ROC Curve** (Receiver Operating Characteristic)

**Axes**:
- **Y-axis**: TPR (True Positive Rate) = Recall = Sensitivity
  - `TPR = TP / (TP + FN)`
  - "**Of all actual positives**, how many did we catch?"

- **X-axis**: FPR (False Positive Rate) = 1 - Specificity
  - `FPR = FP / (FP + TN)`
  - "**Of all actual negatives**, how many did we incorrectly flag?"

**What it shows**:
- Trade-off between catching positives (TPR) vs false alarms (FPR)
- Summarizes confusion matrices across **all thresholds**
- Each point = one threshold setting

---

### **AUC** (Area Under the ROC Curve)

**Interpretation**:
- **Range**: 0 to 1
- **0.5**: Random classifier (diagonal line)
- **0.7-0.8**: Acceptable
- **0.8-0.9**: Good
- **0.9+**: Excellent
- **1.0**: Perfect classifier

**Meaning**: Probability that the model ranks a random positive example higher than a random negative example

**When to use** ✅:
- Moderate class imbalance
- Need threshold-independent metric
- Comparing multiple models
- Threshold tuning matters

**When NOT to use** ❌:
- **Extreme class imbalance** (use Precision-Recall curve instead)
- Specific costs for FP vs FN (use cost-sensitive metrics)
- Balanced classes (accuracy is fine)

---

### **Precision-Recall Curve**

**Alternative to ROC**:
- **Y-axis**: Precision = TP / (TP + FP)
- **X-axis**: Recall = TP / (TP + FN)

**When to use** ✅:
- **High base rate of negatives** (rare disease, fraud detection)
- Focus on positive class performance
- Large number of true negatives would skew ROC/AUC

**Why better for rare events**:
- Not affected by large number of true negatives
- Precision focuses on positive predictions
- More informative when positives are rare

---

## 3. Attention Mechanism & Transformers

### **Attention (Original - Encoder-Decoder)**

**Concept**: Weighted sum based on relevance

**Mechanism**:
1. Decoder output computes **similarity** with each encoder output (dot product)
2. Apply **softmax** to get normalized attention weights
3. Compute **weighted sum** of encoder outputs
4. Feed both decoder output + weighted sum into fully connected layer

**Key idea**: Model can "focus" on relevant parts of input instead of relying on fixed encoding

---

### **Self-Attention (Transformers)**

**Formula**:
```python
# 1. Create Q, K, V from input embeddings X
Q = X @ W_Q  # Query: "What am I looking for?"
K = X @ W_K  # Key: "What do I contain?"
V = X @ W_V  # Value: "What information do I provide?"

# 2. Compute attention scores (similarity)
scores = (Q @ K.T) / sqrt(d_k)  # d_k = dimension of keys
# Division by sqrt(d_k) for numerical stability

# 3. Softmax to get attention weights
attention_weights = softmax(scores)  # Shape: (seq_len, seq_len)

# 4. Weighted sum of values
output = attention_weights @ V
```

**Intuition**:
- Each token attends to **all tokens in the same sequence**
- Example: "The cat sat on the mat"
  - "sat" (query) computes similarity with all tokens (keys)
  - High similarity with "cat" (subject) and "mat" (object)
  - Takes weighted sum of their values → understands context

**Key difference from original attention**:
- Original: Decoder attends to encoder (different sequences)
- Self-attention: Same sequence attends to itself

---

### **Transformers** (BERT, GPT)

**Architecture**:
- Uses **self-attention** instead of RNNs/LSTMs
- **Multi-head attention**: Multiple attention mechanisms in parallel
- **Position encoding**: Adds positional information (no sequential processing)

**Advantages over RNNs** ✅:
- **Captures long-range dependencies** (no vanishing gradient)
- **Parallel processing** (see below)
- **Faster training** (can process all tokens at once)

**Disadvantages** ❌:
- High memory cost (attention matrix is O(n²))
- High compute cost

---

### **Parallel Processing: BERT vs GPT**

#### **BERT (Encoder, Bidirectional)**:
- **Input**: Full sentence "The cat sat on the mat"
- **Processing**: ALL tokens processed in **PARALLEL**
  - Token "The" sees all tokens simultaneously
  - Token "cat" sees all tokens simultaneously
  - Token "sat" sees all tokens simultaneously
- **Output**: Contextualized embeddings for all tokens **at once**
- **Use case**: Understanding (classification, NER, QA)

**Example**: BERT can be used for quality assessment - it processes entire segments in parallel to understand context bidirectionally.

#### **GPT (Decoder, Autoregressive)**:
- **Generation**: ONE token at a time (sequential)
  - Step 1: "The" → generate "cat"
  - Step 2: "The cat" → generate "sat"
  - Step 3: "The cat sat" → generate "on"
- **Within each step**: Attention over seen tokens is **parallel**
  - "The cat sat" all attend to each other in parallel
- **Use case**: Generation (text completion, chatbots)

**Key Insight**: Both use parallel attention internally (vs RNN which is fully sequential token-by-token). The difference is:
- BERT: Sees full sentence → parallel encoding
- GPT: Generates sequentially → but attention within seen tokens is parallel

---

## 4. Interview Quick Answers

### **"Explain Adam optimizer"**
> "Adam combines momentum and RMSprop for adaptive learning rates per parameter. It maintains exponentially weighted averages of gradients (momentum, β₁=0.9) and squared gradients (RMSprop, β₂=0.999). It includes bias correction to prevent slow start in early iterations. Adam is the default choice for most problems because it adapts well to different parameter scales and converges reliably."

### **"When to use AUC-ROC vs Precision-Recall?"**
> "AUC-ROC is good for moderate class imbalance and threshold-independent model comparison. However, for extreme imbalance like rare disease detection, Precision-Recall is better because it focuses on the positive class and isn't influenced by the large number of true negatives. ROC can look deceptively good when the negative class dominates."

### **"Explain attention in transformers"**
> "Attention allows the model to focus on relevant parts of the input. In transformers, each token computes queries, keys, and values. The query represents 'what am I looking for?', keys represent 'what do I contain?', and values hold the actual information. We compute attention scores as the similarity between queries and keys (normalized by softmax), then take a weighted sum of values. This allows capturing long-range dependencies and enables parallel processing, unlike sequential RNNs."

### **"What's the difference between BERT and GPT?"**
> "Both use transformers, but BERT is an encoder (bidirectional) while GPT is a decoder (autoregressive). BERT sees the full sentence and processes all tokens in parallel to create contextualized embeddings. GPT generates text one token at a time, though it still uses parallel attention over already-generated tokens. BERT is better for understanding tasks, GPT is better for generation."

---

**Last Updated**: 2025-10-30 (Day 3)
**Status**: Theory refresh complete, ready for K-NN implementation
