# ML Gradient Formulas - Quick Reference

**Created**: Based on Day 1 assessment (2025-10-27)
**Purpose**: Quick reference for implementing ML algorithms from scratch

---

## Common Pattern

Most ML algorithms follow this pattern:
```python
1. Forward pass: ŷ = f(X, w, b)
2. Calculate loss: L = loss_function(y, ŷ)
3. Backward pass: compute ∂L/∂w and ∂L/∂b
4. Update: w = w - lr * ∂L/∂w, b = b - lr * ∂L/∂b
```

---

## Linear Regression

### Model:
```
ŷ = Xw + b
```

### Loss (Mean Squared Error):
```
L = (1/n) * Σ(y - ŷ)²
  = (1/2n) * (y - ŷ)ᵀ(y - ŷ)    [vectorized]
```

### Gradients:
```python
∂L/∂w = (1/n) * Xᵀ(ŷ - y)       # Shape: (m,)
∂L/∂b = (1/n) * Σ(ŷ - y)       # Shape: scalar

# Code:
dL_dw = X.T @ (y_pred - y) / n
dL_db = np.sum(y_pred - y) / n
```

**Key insight**: Gradient is proportional to prediction error (ŷ - y)

---

## Logistic Regression

### Model:
```
z = Xw + b
ŷ = σ(z) = 1 / (1 + e^(-z))    [sigmoid]
```

### Loss (Binary Cross-Entropy):
```
L = -(1/n) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

### Gradients:
```python
∂L/∂w = (1/n) * Xᵀ(ŷ - y)       # Same as linear regression!
∂L/∂b = (1/n) * Σ(ŷ - y)       # Same as linear regression!

# Code:
dL_dw = X.T @ (y_pred - y) / n
dL_db = np.sum(y_pred - y) / n
```

**Key insight**: Despite different loss function, gradient has same form as linear regression! The sigmoid and log cancel out perfectly.

### Numerical Stability:
```python
# Prevent log(0):
epsilon = 1e-15
y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
```

---

## Softmax Regression (Multi-class)

### Model:
```
z = Xw + b                      # Shape: (n, K) for K classes
ŷ = softmax(z)
ŷ_k = e^(z_k) / Σe^(z_j)
```

### Loss (Categorical Cross-Entropy):
```
L = -(1/n) * Σ Σ y_ik * log(ŷ_ik)
where y is one-hot encoded
```

### Gradients:
```python
∂L/∂w = (1/n) * Xᵀ(ŷ - y)       # Same pattern again!
∂L/∂b = (1/n) * Σ(ŷ - y)

# Code (y is one-hot encoded):
dL_dw = X.T @ (y_pred - y) / n
dL_db = np.sum(y_pred - y, axis=0) / n
```

---

## Neural Network (Single Hidden Layer)

### Model:
```
z1 = X @ W1 + b1
a1 = σ(z1)                      # Hidden layer activation
z2 = a1 @ W2 + b2
ŷ = σ(z2)                       # Output activation
```

### Loss (Binary Cross-Entropy):
```
L = -(1/n) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

### Gradients (Backpropagation):
```python
# Output layer:
dL_dz2 = ŷ - y                  # Shape: (n, output_size)
dL_dW2 = (1/n) * a1.T @ dL_dz2
dL_db2 = (1/n) * np.sum(dL_dz2, axis=0)

# Hidden layer:
dL_da1 = dL_dz2 @ W2.T
dL_dz1 = dL_da1 * σ'(z1)        # Element-wise multiply
       = dL_da1 * a1 * (1 - a1) # Sigmoid derivative
dL_dW1 = (1/n) * X.T @ dL_dz1
dL_db1 = (1/n) * np.sum(dL_dz1, axis=0)

# Code:
# Forward pass
z1 = X @ W1 + b1
a1 = sigmoid(z1)
z2 = a1 @ W2 + b2
y_pred = sigmoid(z2)

# Backward pass
dz2 = y_pred - y
dW2 = a1.T @ dz2 / n
db2 = np.sum(dz2, axis=0) / n

da1 = dz2 @ W2.T
dz1 = da1 * a1 * (1 - a1)       # sigmoid derivative
dW1 = X.T @ dz1 / n
db1 = np.sum(dz1, axis=0) / n
```

**Key insight**: Chain rule! Each layer's gradient depends on the layer after it.

---

## Common Activation Functions & Derivatives

### Sigmoid:
```python
σ(z) = 1 / (1 + e^(-z))
σ'(z) = σ(z) * (1 - σ(z))

# Code:
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):  # a = σ(z)
    return a * (1 - a)
```

### ReLU:
```python
ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0, else 0

# Code:
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

### Tanh:
```python
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
tanh'(z) = 1 - tanh²(z)

# Code:
def tanh(z):
    return np.tanh(z)

def tanh_derivative(a):  # a = tanh(z)
    return 1 - a**2
```

### Softmax:
```python
softmax(z_i) = e^(z_i) / Σe^(z_j)

# Derivative is complex, but in practice:
# For cross-entropy loss: ∂L/∂z = ŷ - y (one-hot)

# Code:
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

---

## Matrix Dimension Cheat Sheet

**Setup**:
- `n` = number of samples
- `m` = number of features
- `k` = number of output classes

**Common shapes**:
```python
X:      (n, m)     # Data matrix
y:      (n,)       # Labels (or (n, 1))
w:      (m,)       # Weights (or (m, k) for multi-class)
b:      scalar     # Bias (or (k,) for multi-class)
ŷ:      (n,)       # Predictions (or (n, k))
```

**Key operations**:
```python
X @ w:              (n, m) @ (m,) = (n,)
X.T @ (ŷ - y):      (m, n) @ (n,) = (m,)
np.sum(ŷ - y):      (n,) → scalar
```

**Rule of thumb**:
- Forward pass: X @ w (matrix-vector multiply)
- Gradient: X.T @ error (transpose-error multiply)

---

## Common Loss Functions

### Mean Squared Error (MSE):
```python
L = (1/n) * Σ(y - ŷ)²
  = (1/2n) * ||y - ŷ||²        [sometimes with 1/2 for cleaner gradient]

# Code:
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
    # or: return np.sum((y_true - y_pred) ** 2) / (2 * len(y_true))
```

### Binary Cross-Entropy:
```python
L = -(1/n) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]

# Code:
def binary_cross_entropy(y_true, y_pred):
    n = len(y_true)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n
```

### Categorical Cross-Entropy:
```python
L = -(1/n) * Σ Σ y_ik * log(ŷ_ik)
where y is one-hot encoded

# Code:
def categorical_cross_entropy(y_true, y_pred):  # y_true is one-hot
    n = len(y_true)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / n
```

### Sparse Categorical Cross-Entropy:
```python
# When y is integer labels (not one-hot)
L = -(1/n) * Σ log(ŷ[y_i])

# Code:
def sparse_categorical_cross_entropy(y_true, y_pred):  # y_true is integers
    n = len(y_true)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(np.log(y_pred[np.arange(n), y_true])) / n
```

---

## Gradient Descent Variants

### Batch Gradient Descent:
```python
# Use entire dataset
dw = (1/n) * X.T @ (y_pred - y)
w = w - learning_rate * dw
```

### Stochastic Gradient Descent (SGD):
```python
# Use one sample at a time
for i in range(n):
    xi = X[i:i+1]
    yi = y[i:i+1]
    y_pred_i = predict(xi)
    dw = xi.T @ (y_pred_i - yi)
    w = w - learning_rate * dw
```

### Mini-Batch Gradient Descent:
```python
# Use small batches
batch_size = 32
for i in range(0, n, batch_size):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    y_pred_batch = predict(X_batch)
    dw = (1/batch_size) * X_batch.T @ (y_pred_batch - y_batch)
    w = w - learning_rate * dw
```

---

## Common Mistakes to Avoid

### 1. Wrong Matrix Order:
```python
# WRONG:
dw = (y_pred - y) @ X           # ❌ Shape mismatch or wrong result

# CORRECT:
dw = X.T @ (y_pred - y) / n     # ✅
```

### 2. Forgetting to Average:
```python
# WRONG:
loss = -np.sum(y * np.log(y_pred))  # ❌ Total loss, not mean

# CORRECT:
loss = -np.sum(y * np.log(y_pred)) / n  # ✅ Mean loss
```

### 3. Numerical Instability:
```python
# WRONG:
loss = -np.log(y_pred)          # ❌ log(0) = -inf

# CORRECT:
y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
loss = -np.log(y_pred)          # ✅ Safe
```

### 4. Wrong Sigmoid Derivative:
```python
# WRONG:
sigmoid_grad = sigmoid(z) * (1 - z)  # ❌ Using z instead of sigmoid(z)

# CORRECT:
a = sigmoid(z)
sigmoid_grad = a * (1 - a)      # ✅ Use activation, not input
```

### 5. Dimension Mismatch:
```python
# WRONG:
y_pred = X @ w                  # ❌ If w is 2D, result is 2D

# CORRECT:
y_pred = (X @ w).flatten()      # ✅ Or ensure w is (m,) not (m, 1)
# Or:
y_pred = X @ w + b              # Make sure b broadcasts correctly
```

---

## Quick Test - Can You Derive?

**Given**: Logistic Regression with sigmoid activation and binary cross-entropy loss

**Question**: Show that ∂L/∂w = (1/n) * X^T * (ŷ - y)

<details>
<summary>Click for solution</summary>

```
Start:
L = -(1/n) Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
ŷ = σ(z) = 1/(1 + e^(-z))
z = Xw + b

Step 1: ∂L/∂ŷ
∂L/∂ŷ = -(1/n) * [y/ŷ - (1-y)/(1-ŷ)]

Step 2: ∂ŷ/∂z (sigmoid derivative)
∂ŷ/∂z = ŷ(1 - ŷ)

Step 3: ∂z/∂w
∂z/∂w = X^T

Step 4: Chain rule
∂L/∂w = ∂L/∂ŷ * ∂ŷ/∂z * ∂z/∂w
      = -(1/n) * [y/ŷ - (1-y)/(1-ŷ)] * ŷ(1-ŷ) * X^T

Step 5: Simplify
= -(1/n) * [y(1-ŷ) - (1-y)ŷ] * X^T
= -(1/n) * [y - yŷ - ŷ + yŷ] * X^T
= -(1/n) * [y - ŷ] * X^T
= (1/n) * X^T * (ŷ - y)  ✓

Beautiful!
```
</details>

---

## Memory Aids

### "Same Pattern" Trio:
Linear Regression, Logistic Regression, and Softmax all have the same gradient:
```
∂L/∂w = (1/n) * X^T * (ŷ - y)
```
The only difference is the prediction function!

### "Chain Rule Triangle":
```
∂L/∂w = ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂w
        (loss)  (activ)  (linear)
```

### "Transpose Trick":
If forward pass is `X @ w`, gradient involves `X.T @ error`

---

**Pro tip**: Don't re-derive gradients during interviews. Memorize the common patterns and formulas. Understanding comes from practice, not from deriving on the spot.

**Last updated**: 2025-10-27 after Day 1 assessment
