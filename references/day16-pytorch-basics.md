# Day 16 Quick Reference: PyTorch Basics

**Week 3, Day 2** | Focus: Model definition, training loop, autograd, FSDP patterns

---

## 1. nn.Module Pattern

```python
import torch
from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        return self.sigmoid(self.linear(X)).reshape(-1)
```

**Key points:**
- Subclass `nn.Module`
- `__init__`: Define layers/parameters
- `forward()`: Define computation (no backward needed - autograd handles it!)
- Alternative: `nn.Sequential` for simple stacking

---

## 2. Training Loop Pattern

```python
# Setup
model = LogisticRegression(input_dim=10)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass + optimization
    loss.backward()           # Compute gradients
    optimizer.step()          # Update parameters
    optimizer.zero_grad()     # Clear gradients for next iteration
```

**Both orders work:**
- `backward() → step() → zero_grad()` (shown above)
- `zero_grad() → backward() → step()` (also valid)

**Key methods:**
- `.backward()`: Computes gradients via backprop, stores in `.grad` attribute
- `.step()`: Updates parameters using computed gradients
- `.zero_grad()`: Clears accumulated gradients

---

## 3. Loss Functions

### BCELoss vs BCEWithLogitsLoss

| Loss Function | Input Range | Model Output Activation | Use Case |
|--------------|-------------|-------------------------|----------|
| `nn.BCELoss()` | [0, 1] | Requires `nn.Sigmoid()` | When model outputs probabilities |
| `nn.BCEWithLogitsLoss()` | (-∞, ∞) | No sigmoid needed | **Preferred** (numerically stable) |

**Why BCEWithLogitsLoss is better:**
- Numerically stable (combines sigmoid + BCE in one operation)
- Avoids numerical overflow/underflow
- Use this in practice!

---

## 4. Evaluation Mode

```python
# Evaluation/inference
with torch.no_grad():
    y_pred = model(X_test)
    accuracy = (y_pred >= 0.5) == y_test).float().mean()
```

**`torch.no_grad()` benefits:**
- Stops gradient computation (saves memory)
- Faster inference
- Use for: Evaluation, prediction, freezing layers

---

## 5. Model Parameters

```python
# Access all trainable parameters
for param in model.parameters():
    print(param.shape, param.requires_grad)

# Used with optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**Key points:**
- `model.parameters()` returns iterator of all parameters with `requires_grad=True`
- Each parameter has `.grad` attribute after `.backward()`
- Optimizer updates all parameters in one `.step()` call

---

## 6. Autograd Mechanics

**Computational graph:**
```
X → Linear → Sigmoid → BCELoss → Scalar loss
                ↓
            .backward()
                ↓
    Gradients stored in .grad for each parameter
```

**Key concepts:**
- PyTorch builds computational graph during forward pass
- `.backward()` traverses graph in reverse, computing gradients
- Gradients **accumulate** (add to existing `.grad`) → need `.zero_grad()`
- `.grad` attribute only exists for tensors with `requires_grad=True`

---

## 7. FSDP Internals (ZeRO Stage 3)

**Pattern recognized from Week 2 distributed training:**

### Forward Pass
1. **Unshard (all-gather)**: Gather sharded parameters from all GPUs
2. **Compute**: Run forward pass with full parameters
3. **Reshard**: Free memory by resharding parameters

### Key Concepts
- **Stream synchronization**:
  - Communication stream: All-gather parameters (background)
  - Compute stream: Matrix operations (foreground)
  - Must **wait** for communication to finish before using parameters
  - Prevents race conditions (using incomplete data)

- **Prefetching (pipelining)**:
  - While computing layer N, start all-gathering parameters for layer N+1
  - Overlaps communication with computation
  - Hides communication latency behind compute time

**CUDA Streams** = Parallel task queues on GPU (advanced GPU programming)

---

## 8. Common Patterns

### Data Type Conversion
```python
# NumPy → PyTorch
X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
X_tensor = torch.as_tensor(X_numpy, dtype=torch.float32)  # Shares memory

# PyTorch → NumPy
X_numpy = X_tensor.numpy()
X_numpy = X_tensor.detach().numpy()  # If requires_grad=True
```

### GPU Usage
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X = X.to(device)
```

---

## 9. Interview Q&A

**Q: What are the three key methods in PyTorch training loop?**
A:
1. `loss.backward()` - Computes gradients via backprop
2. `optimizer.step()` - Updates parameters using gradients
3. `optimizer.zero_grad()` - Clears accumulated gradients

**Q: When to use BCELoss vs BCEWithLogitsLoss?**
A: Use `BCEWithLogitsLoss` (takes raw logits, numerically stable). Only use `BCELoss` if model already outputs probabilities via sigmoid.

**Q: What does torch.no_grad() do?**
A: Disables gradient computation. Use for evaluation/inference to save memory and speed up computation.

**Q: How does PyTorch autograd work?**
A: Builds computational graph during forward pass, then `.backward()` traverses graph in reverse to compute gradients using chain rule. Gradients stored in `.grad` attribute of each parameter.

**Q: What is FSDP?**
A: Fully Sharded Data Parallel = ZeRO Stage 3. Shards parameters across GPUs, all-gathers before use, reshards after. Enables training models larger than single GPU memory.

---

## 10. Key Formulas

**No formulas for Day 16** - Implementation-focused day

**Memory cost:**
- Model parameters: P × 4 bytes (FP32) or P × 2 bytes (FP16)
- Optimizer states (Adam): P × 12 bytes (momentum + variance + param copy)
- Gradients: P × 4 bytes

---

## 11. Common Pitfalls

1. **Forgetting `.zero_grad()`**: Gradients accumulate across iterations
2. **Using BCELoss with raw logits**: Use `BCEWithLogitsLoss` instead
3. **Not using `torch.no_grad()` for evaluation**: Wastes memory, slower
4. **Forgetting `.backward()` before `.step()`**: Optimizer has no gradients to use
5. **Wrong tensor dtype**: Use `dtype=torch.float32` for inputs/labels

---

## 12. Study Resources

- **PyTorch Quickstart**: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- **Autograd Tutorial**: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- **FSDP Source**: https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py

---

**Created**: 2025-11-12 (Day 16)
**Implementation**: `notebooks/day16-pytorch.ipynb`
**Knowledge Check**: 92.5% (A-) - Strong fundamentals, FSDP concepts directional
