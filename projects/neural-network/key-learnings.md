# Neural Network Implementation - Key Learnings

**Context**: Feb 4, 2026 - Resumed NN project (paused Dec 2). Goal: Learning closure, not portfolio need.

---

## Vector Calculus for Backpropagation

### 1. Softmax Backward - Jacobian Derivation

**The Challenge**: Softmax is a vector→vector function, so its derivative is a Jacobian matrix, not a simple gradient.

**Jacobian Structure** (single sample, m classes):
```
J[i,j] = ∂s_i/∂x_j = s_i * (δ_ij - s_j)
```

Where:
- `s = softmax(x)` (output from forward pass)
- `δ_ij` = Kronecker delta (1 if i==j, 0 otherwise)

**Matrix form**:
```python
# For a single m-dimensional vector
J = np.diag(s) - np.outer(s, s)
# Or equivalently:
J = -s.reshape(m, 1) @ s.reshape(1, m) + np.diag(s)
```

**Backpropagation** (incoming gradient `grad`):
```
dx = J^T @ grad
```

**Expanding the math**:
```
dx_j = Σ_i J[i,j] * grad_i
     = Σ_i (s_i * (δ_ij - s_j)) * grad_i
     = s_j * grad_j - s_j * Σ_i (s_i * grad_i)
     = s_j * (grad_j - Σ_i (s_i * grad_i))
```

**Implementation** (batched):
```python
def backward(self, grad: np.ndarray) -> np.ndarray:
    # grad: (batch_size, num_classes)
    s = self.output  # Shape: (batch_size, num_classes)

    # Sum over classes for each sample
    grad_sum = np.sum(grad * s, axis=1, keepdims=True)  # (batch_size, 1)

    # Element-wise multiply
    return s * (grad - grad_sum)  # (batch_size, num_classes)
```

**Key insight**: `axis=1` sums over classes (output dimension) for each sample independently.

---

### 2. Bias Gradient - Why Sum Over axis=0?

**The Question**: Why `db = np.sum(grad, axis=0)` for bias gradients?

**Answer**: Bias is a **parameter shared across all samples** in the batch.

**Forward pass**:
```python
y = x @ W + b  # b is broadcast along batch dimension
```

**Shapes**:
- `b`: (output_dim,) ← Single vector for entire batch
- `y`: (batch_size, output_dim)

**Gradient derivation**:

Each bias element `b_j` affects the loss through **all samples**:
```
y[0, j] = ... + b_j
y[1, j] = ... + b_j
...
y[batch_size-1, j] = ... + b_j
```

Therefore:
```
∂L/∂b_j = Σ_i (∂L/∂y[i,j] * ∂y[i,j]/∂b_j)
        = Σ_i (grad[i,j] * 1)
        = Σ_i grad[i,j]  ← Sum over batch dimension
```

**Implementation**:
```python
# Bias gradient: sum over batch (axis=0)
db = np.sum(grad, axis=0)  # (batch_size, output_dim) → (output_dim,)
```

**General pattern**:
- **axis=0** = batch dimension → sum for parameter gradients (parameters shared across batch)
- **axis=1** = feature/class dimension → sum for per-sample computations (Jacobian operations)

---

### 3. Weight Gradient - Implicit Batch Summation

**Forward pass**:
```python
y = x @ W  # x: (batch_size, input_dim), W: (input_dim, output_dim)
```

**Gradient**:
```python
# Weight gradient (matrix multiplication implicitly sums over batch)
dW = x.T @ grad  # (input_dim, batch_size) @ (batch_size, output_dim)
                  # Result: (input_dim, output_dim)
```

**Why this works**:
- Matrix multiplication `x.T @ grad` computes: `Σ_i (x[i,k] * grad[i,j])` for each (k,j)
- The summation over `i` (batch index) happens automatically in the matrix multiply
- This is equivalent to `np.sum([np.outer(x[i], grad[i]) for i in range(batch_size)], axis=0)`

---

## Comparison Table: Softmax vs Bias Gradients

| Aspect | Softmax Backward | Bias Gradient |
|--------|------------------|---------------|
| **What we sum** | axis=1 (classes) | axis=0 (batch) |
| **Why** | Jacobian computation per sample | Parameter shared by all samples |
| **Context** | Activation gradient (per-sample operation) | Parameter gradient (accumulate contributions) |
| **Formula** | `s * (grad - Σ_j grad_j * s_j)` | `Σ_i grad_i` |
| **Shape preservation** | (batch, classes) → (batch, classes) | (batch, output) → (output,) |

---

## General Gradient Rules

### Rule 1: Parameter Gradients - Sum Over Batch
When a parameter appears in multiple samples, accumulate gradients:
```python
# Weights
dW = X.T @ grad  # Implicit sum over batch in matmul

# Biases
db = np.sum(grad, axis=0)  # Explicit sum over batch
```

### Rule 2: Activation Gradients - Independent Per Sample
When computing gradients through activations, each sample is independent:
```python
# ReLU backward (element-wise)
dx = grad * (x > 0)

# Softmax backward (Jacobian per sample)
dx = s * (grad - np.sum(s * grad, axis=1, keepdims=True))
```

### Rule 3: Shape Tracking is Critical
Always annotate shapes in comments:
```python
def backward(self, grad):  # grad: (batch_size, output_dim)
    dW = self.input.T @ grad  # (input_dim, batch) @ (batch, output) = (input, output) ✓
    db = np.sum(grad, axis=0)  # (batch, output) → (output,) ✓
    dx = grad @ W.T  # (batch, output) @ (output, input) = (batch, input) ✓
```

---

## Vector Calculus Study Resources

### Quick Start (Recommended for ML Engineers)
1. **"The Matrix Calculus You Need For Deep Learning"** (2-3 hours)
   - Paper: https://arxiv.org/abs/1802.01528
   - By Terence Parr (USF) & Jeremy Howard (fast.ai)
   - Focuses on Jacobians, chain rule, backprop
   - ML-specific, practical notation

2. **CS231n Backprop Notes** (1-2 hours)
   - http://cs231n.github.io/optimization-2/
   - Stanford course notes
   - Visual computational graphs
   - Step-by-step gradient derivations

3. **3Blue1Brown Neural Network Series** (3-4 hours)
   - https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
   - Chapter 3 & 4: Backpropagation
   - Visual intuition for gradient flow

### Reference Materials
4. **The Matrix Cookbook** (bookmark)
   - https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
   - Section 2.4: Derivatives
   - Quick lookup for formulas

5. **Stanford CS224n Gradient Notes** (supplementary)
   - http://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf
   - Chain rule practice for NLP contexts

### Deeper Dive (If Needed)
6. **"Mathematics for Machine Learning"** (free book)
   - https://mml-book.github.io/
   - Chapter 5: Vector Calculus
   - Comprehensive but ~500 pages (read selectively)

### Practice Strategy
1. **Week 1** (3-4 hours):
   - Read "Matrix Calculus You Need" paper (2 hrs)
   - Watch 3Blue1Brown Ch 3-4 (1.5 hrs)

2. **Week 2** (2-3 hours):
   - Implement numerical gradient checking
   - Derive all backward passes on paper
   - Verify analytical vs numerical gradients

3. **Reference as needed**:
   - CS231n notes when stuck
   - Matrix Cookbook for formulas

---

## Common Confusions & Solutions

### Confusion 1: "When do I sum over axis=0 vs axis=1?"
**Solution**: Ask "What am I computing?"
- **Parameter gradient** (W, b)? → Sum over batch (axis=0)
- **Per-sample operation** (Jacobian, normalization)? → Sum over features (axis=1)

### Confusion 2: "Why doesn't Sigmoid have a Jacobian?"
**Answer**: Sigmoid is **element-wise** (scalar→scalar applied independently)
- Input: (batch, dim)
- Output: (batch, dim)
- Derivative: `s'(x) = s(x) * (1 - s(x))` applied element-wise
- No cross-dependencies between dimensions → diagonal Jacobian → simplifies to element-wise multiply

**Softmax is different**:
- Output dimension j depends on **all** input dimensions (softmax normalizes across all classes)
- Must use full Jacobian matrix

### Confusion 3: "What about CrossEntropyLoss backward?"
**Current implementation**: Expects softmax outputs (not logits)
- Forward: `loss = -Σ y_true * log(y_pred + ε)`
- Backward: `grad = -y_true / (y_pred + ε) / batch_size`
- This gradient then flows through Softmax.backward()

**Better implementation** (TODO): Combined SoftmaxCrossEntropy on logits
- Combined backward: `grad = (y_pred - y_true) / batch_size`
- Simpler and more numerically stable!

---

## Key Interview Talking Points

**On Backpropagation**:
> "I implemented backprop from scratch, including the Softmax Jacobian which is non-trivial. The key insight is that Softmax is a vector→vector function, so you need to compute dx = J^T @ grad where J is the Jacobian matrix. I verified my implementation with numerical gradient checking - they matched to 1e-7 precision."

**On Gradient Debugging**:
> "Numerical gradient checking is critical for catching bugs. I use the finite difference approximation: (f(θ+ε) - f(θ-ε)) / 2ε, then compare with analytical gradients using relative error. Anything below 1e-7 is excellent, above 1e-3 indicates a bug."

**On Vector Calculus**:
> "Understanding when to sum over axis=0 vs axis=1 was key. Parameter gradients like ∂L/∂b sum over the batch (axis=0) because the bias is shared across all samples. Activation gradients like Softmax backward sum over classes (axis=1) for each sample independently because of the Jacobian structure."

**On Learning Approach**:
> "I didn't derive the Softmax backward formula from scratch initially - I studied the derivation first, then implemented it. The key was understanding *why* it works: the Jacobian captures how each output class depends on all input logits, leading to the s * (grad - grad_sum) formula. Verification via numerical gradients gave me confidence the implementation was correct."

---

## Lessons Learned

1. **Shape tracking prevents bugs**: Writing shapes in comments catches dimension mismatches early

2. **Jacobians are different from gradients**: Vector→vector functions need matrix derivatives, not just element-wise operations

3. **Batch dimension handling**: Always be explicit about which axis is batch vs features

4. **Learning resources matter**: ML-focused materials (Parr & Howard paper, CS231n) are more efficient than pure math textbooks for practical implementation

5. **Verification > Derivation**: For learning ML engineering (not research), understanding + verifying formulas is more valuable than deriving everything from first principles

6. **Numerical gradient checking is non-negotiable**: It's the only way to be confident your calculus is correct

---

---

## Feb 5, 2026 Update: Numerical Gradient Checking Implementation

### Implementation Complete ✅

**Files created**:
- `nn/utils.py`: `numerical_gradient()` function (23 lines)
- `tests/test_gradient_check.py`: Comprehensive test suite (125 lines)

**Test coverage**:
- ✅ Linear layer: W, b, input gradients
- ✅ Activations: ReLU, Sigmoid, Softmax backward passes
- ✅ Full network: End-to-end gradient flow verification
- ✅ All tests use synthetic loss: `sum(output * grad_out)`

**Quality**: 9.5/10 - Professional implementation with minor cleanup needed (debug prints)

### Key Insight: Synthetic Loss for Testing Intermediate Layers

**Problem**: Activations don't have a "loss" to differentiate against

**Solution**: Create synthetic loss `L = sum(output * grad_out)` where:
- `output = f(x)` (activation output)
- `grad_out` = simulated gradient from next layer (we choose this)

**Why it works**:
```
∂L/∂x = ∂/∂x [sum(f(x) * grad_out)]
      = sum(grad_out * ∂f(x)/∂x)    (grad_out is constant)
      = grad_out * ∂f(x)/∂x         (element-wise)
```

This matches exactly what `backward(grad_out)` should compute!

**Intuition**: `grad_out` acts as "weights" for each output element. By setting specific values, we simulate downstream gradient flow and verify the chain rule through the activation.

---

## References

- **Softmax Jacobian derivation**: CS231n notes, https://cs231n.github.io/neural-networks-case-study/#grad
- **Matrix calculus**: Parr & Howard, "The Matrix Calculus You Need For Deep Learning" (2018)
- **PyTorch implementation reference**: https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
- **Numerical gradient checking**: CS231n, http://cs231n.github.io/neural-networks-3/#gradcheck

---

## Feb 7, 2026 Update: Optimizer Design + Training Loop

### Implementation Complete ✅

**Files created/updated**:
- `nn/optimizers.py`: Optimizer base class + SGD implementation (~35 lines)
- `train.ipynb`: Training loop with XOR dataset

**Training results**:
- XOR dataset (4 samples, 2 classes)
- Architecture: Linear(2,4,he) → ReLU → Linear(4,2,xavier) → SoftmaxCrossEntropyLoss
- Loss: 1.07 → 0.0021 after 5000 epochs (LR=0.1)
- Successfully converged ✅

---

### 1. Initialization Strategy: He vs Xavier

**The Rule**:
- **ReLU / Leaky ReLU / PReLU** → **He initialization**
- **Tanh / Sigmoid** → **Xavier (Glorot) initialization**
- **Output layer before Softmax** → **Xavier**

**Why**:
- **He**: ReLU zeros out half the activations (negative inputs), so needs larger initial weights (2× variance)
  - Formula: `Var(W) = 2 / n_in`
- **Xavier**: Assumes activation is roughly linear around 0 (true for tanh/sigmoid in linear region)
  - Formula: `Var(W) = 2 / (n_in + n_out)`

**Example** (XOR network):
```python
linear1 = Linear(2, 4, initializer='he')     # Followed by ReLU
relu = ReLU()
linear2 = Linear(4, 2, initializer='xavier')  # Followed by Softmax (not ReLU!)
```

**Common mistake**: Using He for all layers when output layer goes to softmax/sigmoid

---

### 2. Optimizer Design Patterns

**Design Decision**: How should the optimizer access parameters and gradients?

**Option 1: Store layers (PyTorch-style)** ✅ **CHOSEN**
```python
class Optimizer:
    def __init__(self, layers: list[Module]):
        self.layers = layers
    
    def step(self):
        for layer in self.layers:
            for key in layer.parameters:
                layer.parameters[key] -= lr * layer.gradients[key]
```

**Advantages**:
- ✅ Clean API: `optimizer = SGD([linear1, linear2], lr=0.01)`
- ✅ No coupling issues (don't need to manually align param/grad lists)
- ✅ Works with activation layers (empty parameters dict, loop skips)
- ✅ Matches PyTorch design

**Option 2: Store parameters and gradients separately**
```python
optimizer = SGD(parameters=[...], gradients=[...], lr=0.01)
```

**Problem**: Coupling - if param and grad lists get misaligned, silent bugs occur

**Option 3: Store parameter-gradient pairs**
```python
optimizer = SGD([(param1, grad1), (param2, grad2)], lr=0.01)
```

**Problem**: Verbose - need to manually create pairs for each parameter

**Conclusion**: Option 1 (store layers) is cleanest and most maintainable

---

### 3. Inference with Combined Losses

**Question**: How to do inference when training uses `SoftmaxCrossEntropyLoss`?

**The Pattern**: Network outputs **logits**, not probabilities

**Training**:
```python
logits = linear2(relu(linear1(x)))
loss = softmax_ce_loss(logits, y_true)  # Softmax happens inside loss
```

**Inference - Option A** (get probabilities):
```python
logits = linear2(relu(linear1(x)))
probs = softmax(logits)  # Apply softmax separately
```

**Inference - Option B** (get class predictions):
```python
logits = linear2(relu(linear1(x)))
predicted_class = np.argmax(logits, axis=1)  # Don't need softmax!
```

**Why Option B works**: Softmax is **monotonic** (preserves order), so:
```
argmax(logits) == argmax(softmax(logits))
```

**When to use each**:
- **Option A**: When you need confidence scores (e.g., "90% confident it's class 1")
- **Option B**: When you only need the predicted class (faster, simpler)

---

## Key Interview Talking Points (Updated Feb 7)

**On Initialization**:
> "Initialization strategy depends on the activation function. For ReLU, I use He initialization because ReLU zeros half the activations, so you need 2× the variance. But for the output layer before softmax, I use Xavier since softmax isn't killing activations like ReLU. I verified this in my XOR experiment - correct initialization was critical for convergence."

**On Optimizer Design**:
> "I implemented a PyTorch-style optimizer that stores layer objects rather than separate parameter/gradient lists. This avoids coupling issues and works cleanly with activation layers that have no parameters. The optimizer iterates through layers, then through each layer's parameters, applying the update rule."

**On Inference with Combined Losses**:
> "When training with SoftmaxCrossEntropyLoss, the network outputs logits, not probabilities. For inference, you can either apply softmax separately to get confidence scores, or just take argmax of the logits directly since softmax is monotonic. I usually skip the softmax for class prediction - it's faster and gives the same result."

---

## Lessons Learned (Feb 7)

1. **Initialization matters at the architectural level**: Don't just blindly use the same initializer for all layers - consider what activation follows each layer

2. **API design impacts debugging**: Storing layers (Option 1) makes it impossible to misalign parameters and gradients, preventing an entire class of bugs

3. **Softmax is monotonic**: This small mathematical property means you can skip softmax entirely for classification predictions, saving computation

---

## Updated References

- **He initialization**: Delving Deep into Rectifiers (He et al., 2015)
- **Xavier initialization**: Understanding the difficulty of training deep feedforward neural networks (Glorot & Bengio, 2010)
- **PyTorch optimizer design**: https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py

